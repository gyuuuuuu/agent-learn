from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from dotenv import load_dotenv

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM = "你是一个简洁直接的技术助手，回答不超过三句话。"
ENV_PATH = Path(__file__).with_name(".env")
DEFAULT_TEMPERATURE = 1.0

# 结构化输出模型
class TagResult(BaseModel):
    title: str = Field(..., description="不超过 20 字的标题")
    tags: list[str] = Field(..., description="3 到 5 个标签，每个标签不超过 6 个字")

# 重试分类
RETRYABLE_EXC = {
    "RateLimitError",          # 429 限流
    "APITimeoutError",         # 超时
    "APIConnectionError",      # 连不上
    "APIConnectionTimeoutError",
    "InternalServerError",     # 5xx
    "Timeout",
    "ConnectionError",
}

def is_retryable_exception(exc: Exception) -> bool:
    if type(exc).__name__ in RETRYABLE_EXC:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in {408, 409, 425, 429, 500, 502, 503, 504}


# 记账
@dataclass
class CallRescord:
    seq: int
    tag: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish: str

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

class UsageLedger:
    def __init__(self):
        self.records: list[CallRescord] = []

    def add(self, tag: str, prompt: int, completion: int, latency: float, finish: str) -> None:
        seq = len(self.records) + 1
        self.records.append(CallRescord(seq, tag, prompt, completion, latency, finish))

    def _cost(self) -> str | None:
        # todo: 计算总费用
        return None

    def report(self) -> str:
        if not self.records:
            return "（还没有任何调用记录）"

        lines = [
            "| # | tag | prompt | completion | 合计 | 耗时(s) | finish |",
            "|---|-----|-------:|-----------:|-----:|--------:|--------|",
        ]
        for r in self.records:
            flag = " ⚠截断" if r.finish == "length" else ""
            lines.append(
                f"| {r.seq} | {r.tag} | {r.prompt_tokens} | {r.completion_tokens} "
                f"| {r.total} | {r.latency_s:.2f} | {r.finish}{flag} |"
            )
        lines.append(
            f"| — | **合计 {len(self.records)} 次** | **{sum(r.prompt_tokens for r in self.records)}** "
            f"| **{sum(r.completion_tokens for r in self.records)}** | "
            f"**{sum(r.total for r in self.records)}** | — | — |"
        )
        cost = self._cost()
        if cost:
            lines.append(f"\n估算成本：{cost}（按 .env 里填的单价计算）")
        return "\n".join(lines)


# 剥壳
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)

def strip_code_fence(text: str) -> str:
    """剥掉 ```json ... ``` 代码块"""
    text = (text or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        text = text[i : j + 1]
    return text

def json_instrunction(model: type[BaseModel]) -> str:
    """DeepSeek 的 json_object 模式要求提示词里出现 "json" 字样，且最好给出示例。"""
    shape = {name: str(f.annotation) for name, f in model.model_fields.items()}
    example = {"title": "示例标题", "tags": ["标签一", "标签二"]}
    return (
        "你的回复必须是合法 json，不要输出任何解释文字，不要使用 Markdown 代码块围栏。\n"
        f"json 的字段名必须与下面完全一致（大小写敏感）：\n{json.dumps(shape, ensure_ascii=False)}\n"
        f"输出示例：{json.dumps(example, ensure_ascii=False)}"
    )

# 客户端
class DeepSeekChat:
    def __init__(
            self,
            api_key: str,
            *,
            model: str = DEFAULT_MODEL,
            base_url: str = DEEPSEEK_BASE_URL,
            temperature: float = DEFAULT_TEMPERATURE,
            max_tries: int = 3,
            timeout: float = 30.0,
            verbose: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tries = max_tries
        self.verbose = verbose
        self.ledger = UsageLedger()
        self._last_payload: dict[str, Any] = {}
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _sampling(self, temperature: float | None = None) -> dict[str, Any]:
        if "reasoner" in self.model:
            return {}
        return {"temperature": self.temperature if temperature is None else temperature}

    @property
    def is_reasoner(self) -> bool:
        return "reasoner" in self.model

    def _create(self, payload: dict[str, Any], *, dry_run: bool = False) -> Any:
        payload.setdefault("model", self.model)
        self._last_payload = payload
        if dry_run:
            return None

        last_exec: Exception | None = None
        for attempt in range(self.max_tries + 1):
            try:
                return self.client.chat.completions.create(**payload)
            except Exception as exc:
                last_exc = exc
                if not is_retryable_exception(exc):
                    raise
                if attempt == self.max_retries:
                    break
                delay = min(2.0**attempt, 8.0) + random.uniform(0, 0.4)
                if self.verbose:
                    print(f"\n[retry] {type(exc).__name__}，{delay:.1f}s 后第 {attempt + 1} 次重试")
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def last_payload(self) -> dict[str, Any]:
        return self._last_payload

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tag: str = "chat",
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        dry_run: bool = False,
    ) -> str:
        payload: dict[str, Any] = {"messages": messages, **self._sampling(temperature)}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            # DeepSeek 只支持 json_object，不支持 json_schema 严格模式
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        resp = self._create(payload, dry_run=dry_run)
        if dry_run:
            return ""
        dt = time.perf_counter() - t0

        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", "unknown") or "unknown"
        usage = getattr(resp, "usage", None)
        self.ledger.add(
            tag,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            dt,
            finish,
        )
        return choice.message.content or ""

    def chat_stream(
            self,
            messages: list[dict[str, Any]],
            *,
            tag: str = "stream",
            temperature: float | None = None,
            max_tokens: int | None = None,
            dry_run: bool = False,
    ) -> Iterator[tuple[str, str]]:
        payload: dict[str, Any] = {
            "messages": messages,
            **self._sampling(temperature),
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        t0 = time.perf_counter
        try:
            stream = self._create(payload, dry_run=dry_run)
        except Exception as exc:
            msg = str(exc).lower()
            if "stream_options" in msg or "include_usage" in msg:
                if self.verbose:
                    print("\n[fallback] 该模型不支持 stream_options，降级为不带用量的流式")
                payload.pop("stream_options", None)
                stream = self._create(payload, dry_run=dry_run)
            else:
                raise

        if dry_run:
            return

        prompt_tokens = 0
        completion_tokens = 0   
        finish = "unknown"

        for chunck in stream:
            usage = getattr(chunck, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0

            choices = getattr(chunck, "choices", None) or []
            if not choices:
                continue
            ch = choices[0]
            if getattr(ch, "finish_reason", None):
                finish = ch.finish_reason
            delta = getattr(ch, "delta", None)
            if delta is None:
                continue

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ("reasoning", reasoning)
            content = getattr(delta, "content", None)
            if content:
                yield ("content", content)

        dt = time.perf_counter() - t0()
        self.ledger.add(tag, prompt_tokens, completion_tokens, dt, finish)


    def chat_json(
            self,
            messages: list[dict[str, Any]],
            model: type[BaseModel],
            *,
            tag: str = "json",
            max_repairs: int = 2,
            dry_run: bool = False,
    ) -> BaseModel:
        work = [dict(m) for m in messages]
        idx = 1 if work and work[0].get("role") == "system" else 0
        work.insert(idx, {"role": "system", "content": json_instrunction(model)})

        last_exc: Exception | None = None
        json_mode = True
        for attempt in range(max_repairs + 1):
            try:
                raw = self.chat(
                    work,
                    tag=f"{tag}#{attempt + 1}",
                    temperature=0.0,
                    json_mode=json_mode,
                    dry_run=dry_run,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if json_mode and ("response_format" in msg or "json_object" in msg):
                    # 有些模型（例如推理模型）不接受 response_format，降级为纯提示词约束
                    if self.verbose:
                        print("\n[fallback] 该模型不接受 response_format，降级为纯提示词约束")
                    json_mode = False
                    raw = self.chat(
                        work,
                        tag=f"{tag}#{attempt + 1}",
                        temperature=0.0,
                        json_mode=False,
                        dry_run=dry_run,
                    )
                else:
                    raise
            if dry_run:
                return None  # type: ignore[return-value]

            try:
                data = json.loads(strip_code_fence(raw))
                return model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_exc = exc
                if attempt >= max_repairs:
                    break
                # 把「模型自己写的错误输出」+「错误原因」一起回灌
                work = work + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "上一次的输出无法通过校验，原因：\n"
                            f"{exc}\n\n请只输出修正后的 json，不要任何解释，不要代码块围栏。"
                        ),
                    },
                ]
        raise ValueError(f"结构化输出连续 {max_repairs + 1} 次未通过校验：{last_exc}")


# ---------------------------------------------------------------- CLI

HELP = """
可用命令
  /help              显示这份帮助
  /system <文本>      重设系统提示（不带参数则打印当前值）
  /json <文本>        结构化输出演示：文字 -> {title, tags}
  /tokens            打印最近一次真实发出的请求体
  /stats             打印 Token 记账表
  /reset             清空对话历史
  /exit              退出
其他任何输入都当成普通提问，走流式对话。
""".strip()


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _force_utf8()
    load_dotenv(ENV_PATH)

    parser = argparse.ArgumentParser(description="Day 1：直连 DeepSeek 的问答 CLI")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL,
                        help="模型名，如 deepseek-chat / deepseek-reasoner")
    parser.add_argument("--key", help="直接传 API Key（不推荐，会留在命令历史里）")
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--no-stream", action="store_true", help="关掉流式，对比 usage 差异")
    parser.add_argument("--reasoning", action="store_true",
                        help="deepseek-reasoner 时显示推理过程（推理也计费）")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求体，不发请求")
    args = parser.parse_args()

    if args.dry_run:
        api_key = args.key or "sk-dry-run-placeholder"
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    bot = DeepSeekChat(
        api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    if bot.is_reasoner:
        print("提示：deepseek-reasoner 不支持 temperature 等采样参数，已自动省略。")
        print("      它的推理过程同样计费，想显示出来就加 --reasoning。\n")

    system_prompt = args.system

    def fresh() -> list[dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}]

    history = fresh()

    print(f"模型：{args.model}｜base_url：{args.base_url}｜temperature：{args.temperature}")
    print("输入 /help 看命令，/exit 退出。\n")

    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in {"/exit", "/quit"}:
            break
        if line == "/help":
            print(HELP + "\n")
            continue
        if line == "/reset":
            history = fresh()
            print("（历史已清空 —— 顺便提醒：模型从来没记住过任何东西，历史只活在这个 list 里）\n")
            continue
        if line == "/stats":
            print("\n" + bot.ledger.report() + "\n")
            continue
        if line == "/tokens":
            print("\n最近一次真实发出的请求体：\n")
            print(json.dumps(bot.last_payload(), ensure_ascii=False, indent=2, default=str))
            print()
            continue
        if line.startswith("/system"):
            arg = line[len("/system"):].strip()
            if arg:
                system_prompt = arg
                history = fresh()
                print("（系统提示已更新，历史已重置）\n")
            else:
                print(f"当前系统提示：{system_prompt}\n")
            continue

        if line.startswith("/json"):
            text = line[len("/json"):].strip() or (
                "上下文窗口决定了模型一次能看多少文本，多轮对话时历史会被反复重发。"
            )
            try:
                result = bot.chat_json(
                    [*history, {"role": "user", "content": f"请处理这段文字：{text}"}],
                    TagResult,
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    print("（dry-run：已生成请求体，用 /tokens 查看）\n")
                else:
                    print(f"\ntitle = {result.title}")
                    print(f"tags  = {result.tags}\n")
            except Exception as exc:  # noqa: BLE001
                print(f"\n结构化输出最终失败：{exc}\n")
            continue

        # ---------------- 普通提问 ----------------
        history.append({"role": "user", "content": line})
        t0 = time.perf_counter()
        print("AI > ", end="", flush=True)
        try:
            if args.no_stream:
                answer = bot.chat(history, tag="chat", dry_run=args.dry_run)
                print(answer)
            else:
                # 关键：reasoning 片段绝不能混进 answer，否则会把推理过程当成正文存进历史
                answer_parts: list[str] = []
                in_reasoning = False
                for kind, piece in bot.chat_stream(
                    history, tag="stream", dry_run=args.dry_run
                ):
                    if kind == "reasoning":
                        if args.reasoning:
                            if not in_reasoning:
                                print("\n[推理过程]", end="", flush=True)
                                in_reasoning = True
                            print(piece, end="", flush=True)
                        continue
                    if in_reasoning:
                        print("\n[推理结束]\nAI > ", end="", flush=True)
                        in_reasoning = False
                    answer_parts.append(piece)
                    print(piece, end="", flush=True)
                print()
                answer = "".join(answer_parts)
        except Exception as exc:  # noqa: BLE001
            history.pop()
            print(f"\n调用失败（已回滚本轮历史）：{type(exc).__name__}: {exc}\n")
            print("提示：401 是 Key 问题；402 通常是余额不足；400 多半是参数或模型名写错。\n")
            continue

        if args.dry_run:
            # dry-run 没有真实回答，不要把空内容写进历史
            history.pop()
            continue

        history.append({"role": "assistant", "content": answer})
        if bot.ledger.records:
            last = bot.ledger.records[-1]
            print(
                f"   ↳ {time.perf_counter() - t0:.2f}s｜"
                f"prompt={last.prompt_tokens} completion={last.completion_tokens}"
                f"｜finish={last.finish}｜历史 {len(history)} 条消息\n"
            )

    print("\n最终 Token 记账表：\n")
    print(bot.ledger.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())