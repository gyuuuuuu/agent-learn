"""
ReAct = Reason + Act，就是一个**带工具调用的 for 循环**。
    while 还没结束:
          让模型看一眼当前全部上下文，问它：下一步干嘛？
          如果它说"我要查 X"   → 你去查，把结果塞回上下文，继续循环
          如果它说"答案是 Y"   → 结束，返回 Y
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from tools import (
    MAX_OBSERVATION_CHARS,
    REGISTRY,
    execute_tool,
    render_tools_as_text,
    tool_schema,
)

# ---------------------------------------------------------------- 异常


class AgentError(RuntimeError):
    """agent 层的各种终止异常，统一基类，方便上层一把兜住。"""


class MaxStepsExceeded(AgentError):
    """模型一直在调工具，始终不给最终答案。[1]"""


class TokenBudgetExceeded(AgentError):
    """单次请求的 prompt 已经超出预算 —— 上下文被工具结果撑爆了。[1][4]"""


class ToolLoopDetected(AgentError):
    """同样的工具 + 同样的参数重复太多次。[5]"""


class ModelCallFailed(AgentError):
    """模型调用本身失败（鉴权、限流、超时等），重试也没救回来。"""


@dataclass
class StepRecord:
    index: int
    thought: str = ""
    tool_name: str = ""
    tool_args: str = ""
    observation: str = ""
    ok: bool = True
    error_kind: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw_observation_chars: int = 0


@dataclass
class AgentResult:
    answer: str
    steps: list[StepRecord] = field(default_factory=list)
    stopped_reason: str = "final_answer"

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    def trace_markdown(self) -> str:
        """把轨迹渲染成一张表 —— 这是你调 prompt 时的主要工作台。"""
        lines = [
            "| 步 | Thought 摘要 | 工具 | 参数 | 结果 | prompt | 耗时(s) |",
            "|---:|---|---|---|---|---:|---:|",
        ]
        for s in self.steps:
            thought = (s.thought or "").replace("\n", " ")[:36]
            args = (s.tool_args or "").replace("\n", " ")[:30]
            obs = (s.observation or "").replace("\n", " ")[:40]
            flag = "" if s.ok else f" ❌{s.error_kind}"
            lines.append(
                f"| {s.index} | {thought} | `{s.tool_name}` | `{args}` "
                f"| {obs}{flag} | {s.prompt_tokens} | {s.latency_s:.2f} |"
            )
        return "\n".join(lines)


@dataclass
class ParsedTurn:
    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    final: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final is not None

    @property
    def has_action(self) -> bool:
        return self.action is not None

_FENCE = re.compile(r"```(?:json|text)?\s*(.*?)\s*```", re.DOTALL)

def _after_colon(line: str) -> str:
    """取第一个冒号后面的内容。

    注意：不能写成 `for sep in (":", "：")` 依次尝试 ——
    `Action Input：{"expression": "23*47"}` 里的 JSON 内部含有半角冒号，
    先试半角就会在 "expression" 那里截断，切出一个残缺的 JSON。
    必须取**两个候选里位置更靠前**的那个。
    """
    idx = -1
    for sep in (":", "："):
        i = line.find(sep)
        if i != -1 and (idx == -1 or i < idx):
            idx = i
    return line[idx + 1 :].strip() if idx != -1 else ""

def _is_marker(line: str) -> bool:
    low = line.strip().lower().lstrip("*#- ").strip()
    return low.startswith(("thought", "action", "final"))

def parse_react(text: str) -> ParsedTurn:
    """解析模型输出的 Thought / Action / Action Input / Final Answer。

    为什么要有这个解析器？因为**文本式 ReAct 是这套范式的原始形态**，
    而它所有的痛苦，都是后来原生 tool_calls 想解决的问题。亲手写一遍解析，
    你会真正理解"为什么要有 function calling 标准"。

    模型实际会犯的错（这个解析器都做了容忍）：
      · 整段包在 ``` 代码块里
      · "Action: " 后面没空格、用了中文冒号
      · Action Input 的 JSON 跨了多行（美化输出）
      · 把 Thought 和 Action 挤在同一行
      · Final Answer 后面跟一大段多行正文
      · 干脆不写 Thought，直接 Action
    """
    raw = (text or "").strip()
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1).strip()

    turn = ParsedTurn()
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip().lstrip("*#- ").strip()
        low = stripped.lower()

        if low.startswith("thought"):
            turn.thought = _after_colon(stripped)
            i += 1
        elif low.startswith("action input"):
            head = _after_colon(stripped)
            body = [head] if head else []
            i += 1
            # Action Input 的 JSON 可能是多行的，收下直到下一个标记行
            while i < len(lines) and not _is_marker(lines[i]):
                body.append(lines[i])
                i += 1
            turn.action_input = "\n".join(body).strip()
        elif low.startswith("final"):
            first = _after_colon(stripped)
            rest = "\n".join(lines[i + 1 :]).strip()
            turn.final = (first + ("\n" + rest if rest else "")).strip()
            break
        elif low.startswith("action"):
            turn.action = _after_colon(stripped).strip().strip("`'\"")
            i += 1
        else:
            i += 1

    # 兜底：模型有时只写 Action 不写 Thought，有时把 Action 塞在 Thought 同一行
    if turn.action is None and not turn.is_final:
        mm = re.search(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)", raw)
        if mm:
            turn.action = mm.group(1)

    return turn


# ---------------------------------------------------------------- ReAct Agent
DEFAULT_SYSTEM = """你是一个善于使用工具的助手。
规则：
1. 需要事实、计算或文件内容时，必须调用工具，不要凭记忆编造。
2. 一次只调用必要的工具，拿到结果后再决定下一步。
3. 已经能回答时，直接给出最终答案，不要再调用工具。
4. 工具返回错误时，先读错误信息再决定怎么办；不要用同样的参数重复调用。
5. 最终答案要直接回答用户，不要复述你调用了哪些工具。
""".strip()

TEXT_MODE_SYSTEM = """你是一个用 ReAct 模式工作的助手。

每一轮你**只能**输出下面两种形式之一，且不要输出任何多余内容。

【形式一：需要调用工具】
Thought: <你的推理，一句话>
Action: <工具名>
Action Input: <JSON 参数对象>

【形式二：可以给出最终答案】
Thought: <你的推理>
Final Answer: <给用户的答案>

可用工具：
{tools}

严格要求：
- Action 的值必须是上面列表里的工具名，不要编造。
- Action Input 必须是合法的 JSON 对象，键名与工具定义完全一致，不要用单引号。
- 一次只允许调用一个工具。
- 已经能回答时，必须用 Final Answer，不要继续调用工具。
""".strip()

RETRYABLE_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "APIConnectionTimeoutError",
    "InternalServerError",
    "Timeout",
    "ConnectionError",
}

def _is_retryable(exc: Exception) -> bool:
    if type(exc).__name__ in RETRYABLE_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ModelReply:
    """把 SDK 的响应归一化成一个简单结构，mock 和真实响应都往这里塞。"""

    content: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_s: float = 0.0
    raw_message: Any = None


class ReactAgent:

    def __init__(
            self,
            client: Any,
            *,
            model: str,
            mode: str = "native",
            temperature: float = 0.0,
            max_steps: int = 8,
            max_prompt_tokens: int = 12_000,
            max_observation_chars: int = MAX_OBSERVATION_CHARS,
            max_repeat_calls: int = 3,
            max_retries: int = 2,
            system_prompt: str | None = None,
            verbose: bool = True,
    ) -> None:
        if mode not in {"native", "text"}:
            raise ValueError("mode 只能是 'native' 或 'text'")
        self.client = client
        self.model = model
        self.mode = mode
        self.temperature = temperature
        self.max_steps = max_steps
        self.max_prompt_tokens = max_prompt_tokens
        self.max_observation_chars = max_observation_chars
        self.max_repeat_calls = max_repeat_calls
        self.max_retries = max_retries
        self.verbose = verbose

        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif mode == "text":
            self.system_prompt = TEXT_MODE_SYSTEM.format(tools=render_tools_as_text())
        else:
            self.system_prompt = DEFAULT_SYSTEM

        self._on_event: Callable[[str, dict[str, Any]], None] | None = None

    def on_event(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        """注册一个回调函数，接收 agent 运行时的各种事件。

        事件类型：
          - "step_start"：每一步开始时触发，参数为 StepRecord
          - "step_end"：每一步结束时触发，参数为 StepRecord
          - "agent_end"：整个 agent 结束时触发，参数为 AgentResult
        """
        self._on_event = fn

    def _emit(self, kind: str, **payload: Any) -> None:
        if self._on_event:
            self._on_event(kind, payload)

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        # deepseek-reasoner 不接受 temperature 之类的采样参数
        if "reasoner" not in self.model:
            payload["temperature"] = self.temperature
        if self.mode == "native":
            payload["tools"] = tool_schema()
            # tool_choice="auto"：让模型自己决定要不要调工具。
            # 设成 "required" 会强迫它每轮都调，最后一步就再也出不来最终答案了。
            payload["tool_choice"] = "auto"
        return payload

    def _call_model(self, messages: list[dict[str, Any]]) -> ModelReply:
        payload = self._build_payload(messages)
        last_exc: Exception | None = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**payload)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise ModelCallFailed(
                        f"模型调用失败：{type(exc).__name__}: {exc}"
                    ) from exc
                delay = min(2.0**attempt, 8.0)
                self._log(f"  [retry] {type(exc).__name__}，{delay:.1f}s 后重试")
                time.sleep(delay)
        else:  # pragma: no cover
            raise ModelCallFailed(str(last_exc))

        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        message = choice.message
        usage = getattr(resp, "usage", None)

        return ModelReply(
            content=getattr(message, "content", None) or "",
            tool_calls=list(getattr(message, "tool_calls", None) or []),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=getattr(choice, "finish_reason", None) or "stop",
            latency_s=latency,
            raw_message=message,
        )

    def run(self, question: str) -> AgentResult:
        """执行一个 ReAct 循环，直到模型给出最终答案或触发异常。

        1. 先把用户问题和系统提示塞进上下文。
        2. 循环：
            a. 调模型，拿到 Thought / Action / Action Input / Final Answer
            b. 如果是 Final Answer，结束循环
            c. 如果是 Action，调用工具，把结果塞回上下文
        """
        result = AgentResult(answer="")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        repeat_counter: Counter[tuple[str, str]] = Counter()

        for index in range(1, self.max_steps + 1):
            self._emit("thinking", index=index)
            reply = self._call_model(messages)

            if reply.prompt_tokens > self.max_prompt_tokens:
                result.stopped_reason = "token_budget"
                result.answer = (
                    result.answer
                    or f"（上下文已达 {reply.prompt_tokens} token，超出预算 "
                    f"{self.max_prompt_tokens}，为避免继续烧钱主动中止）"
                )
                self._emit("stop", reason="token_budget", tokens=reply.prompt_tokens)
                return result

            if self.mode == "native":
                outcome = self._handle_native(index, messages, reply, result, repeat_counter)
            else:
                outcome = self._handle_text(index, messages, reply, result, repeat_counter)

            if outcome == "finished":
                return result
            if outcome == "aborted":
                return result

            sig = last_signature(result)
            if sig and repeat_counter[sig] >= self.max_repeat_calls:
                result.stopped_reason = "tool_loop"
                result.answer = (
                    result.answer
                    or f"（检测到重复调用：工具 `{sig[0]}` 用同样的参数被调用了 "
                    f"{repeat_counter[sig]} 次，已主动打断）"
                )
                self._emit("stop", reason="tool_loop", signature=sig)
                return result

        result.stopped_reason = "max_steps"
        result.answer = result.answer or (
            f"（已达到最大步数 {self.max_steps}，模型仍未给出最终答案。"
            "这通常意味着 max_steps 太小、工具描述太含糊，或者工具一直在报错）"
        )
        self._emit("stop", reason="max_steps")
        return result

            
    # 原生 tool_calls 分支
    def _handle_native(
            self,
            index: int,
            messages: list[dict[str, Any]],
            reply: ModelReply,
            result: AgentResult,
            repeat_counter: Counter[tuple[str, str]],
        ) -> str:
        if not reply.tool_calls:
            # 模型没调工具，直接给了最终答案
            rec = StepRecord(
                    index=index,
                    thought=reply.content,
                    prompt_tokens=reply.prompt_tokens,
                    completion_tokens=reply.completion_tokens,
                    latency_s=reply.latency_s,
                    finish_reason=reply.finish_reason,
            )
            result.steps.append(rec)
            result.answer = reply.content
            result.stopped_reason = "final_answer"
            self._emit("final", text=reply.content, record=rec)

            if reply.finish_reason == "length":
                # 被截断的"最终答案"不是最终答案。这个判断必须在代码里显式写，
                # 否则你会一边看着半截回答，一边以为模型就是这么简洁。
                self._log("  ⚠ finish_reason=length：最终答案被 max_tokens 截断了")
            return "finished"

        # 把 assistant 那条带 tool_calls 的消息原样写回历史。
        # 写回的 content 用接口返回的原文（可能是 None），不要自己造。
        messages.append(
            {
                "role": "assistant",
                "content": getattr(reply.raw_message, "content", None),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in reply.tool_calls
                ],
            }
        )

        for tc in reply.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            which = index if len(reply.tool_calls) == 1 else f"{index}.?"
            self._emit("tool_call", index=which, name=name, arguments=raw_args)

            signature = (name, str(raw_args))
            repeat_counter[signature] += 1
            _push_signature(result, signature)

            outcome = execute_tool(name, raw_args, max_chars=self.max_observation_chars)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": outcome.observation}
            )

            rec = StepRecord(
                index=index,
                thought=reply.content,
                tool_name=name,
                tool_args=str(raw_args),
                observation=outcome.observation,
                ok=outcome.ok,
                error_kind=outcome.error_kind,
                latency_s=round(reply.latency_s + outcome.latency_s, 3),
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                finish_reason=reply.finish_reason,
                raw_observation_chars=outcome.raw_chars,
            )
            result.steps.append(rec)
            self._emit("observation", index=which, record=rec)

        return "continue"


    # 文本 ReAct 分支
    def _handle_text(
            self,
            index: int,
            messages: list[dict[str, Any]],
            reply: ModelReply,
            result: AgentResult,
            repeat_counter: Counter[tuple[str, str]],     
    ) -> str:
        turn = parse_react(reply.content)
        self._emit("parsed", index=index, turn=turn)

        # 先处理 Final Answer。注意要放在 Action 前面 
        if turn.is_final:
            rec = StepRecord(
                index=index,
                thought=turn.thought,
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                latency_s=reply.latency_s,
                finish_reason=reply.finish_reason,
            )
            result.steps.append(rec)
            result.answer = turn.final
            result.stopped_reason = "final_answer"
            self._emit("final", text=turn.final, record=rec)
            return "finished"

        if not turn.has_action:
            # 模型既没给工具也没给答案 —— 最常见的"格式跑偏"。
            # 不要直接崩，把这条反馈塞回去，通常下一轮就纠回来了。
            messages.append({"role": "assistant", "content": reply.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Observation: 你的输出不符合约定格式。"
                        "请严格按以下两种形式之一输出，不要加任何其它内容：\n"
                        "Thought: ... / Action: <工具名> / Action Input: <JSON>\n"
                        "或 Thought: ... / Final Answer: ..."
                    ),
                }
            )
            rec = StepRecord(
                index=index,
                thought=turn.thought,
                tool_name="(格式错误)",
                observation="未解析到 Action 或 Final Answer，已提示模型纠正格式",
                ok=False,
                error_kind="format",
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                latency_s=reply.latency_s,
                finish_reason=reply.finish_reason,
            )
            result.steps.append(rec)
            self._emit("observation", index=index, record=rec)
            _push_signature(result, ("(format)", "format"))
            repeat_counter[("(format)", "format")] += 1
            return "continue"
        
        # 文本模式下没有 tool 角色，只能靠"对话伪装"：
        # assistant 说它要调工具，user 把结果当作 Observation 回给它。
        # 这就是原生 tool_calls 存在的全部理由 —— 手工协议非常脆。
        messages.append({"role": "assistant", "content": reply.content})
        name = turn.action or ""
        raw_args = turn.action_input or "{}"

        signature = (name, raw_args)
        repeat_counter[signature] += 1
        _push_signature(result, signature)

        outcome = execute_tool(name, raw_args, max_chars=self.max_observation_chars)
        messages.append(
            {"role": "user", "content": f"Observation: {outcome.observation}\n\n请继续。"}
        )

        rec = StepRecord(
            index=index,
            thought=turn.thought,
            tool_name=name,
            tool_args=raw_args,
            observation=outcome.observation,
            ok=outcome.ok,
            error_kind=outcome.error_kind,
            latency_s=round(reply.latency_s + outcome.latency_s, 3),
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            finish_reason=reply.finish_reason,
            raw_observation_chars=outcome.raw_chars,
        )
        result.steps.append(rec)
        self._emit("observation", index=index, record=rec)
        return "continue"


def _push_signature(result: AgentResult, sig: tuple[str, str]) -> None:
    """把最近的签名挂到 result 上，供 run() 做循环判定。"""
    result._last_signature = sig  # type: ignore[attr-defined]


def last_signature(result: AgentResult) -> tuple[str, str] | None:
    return getattr(result, "_last_signature", None)
