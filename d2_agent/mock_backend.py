from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any


class MockAPIError(Exception):
    """形状与 openai 的 APIStatusError 对齐，带 status_code 以便验证重试分类。"""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code


def _estimate_tokens(obj: Any) -> int:
    """粗糙的 token 估算。绝对值不准没关系，**相对增长**才是你要看的。"""
    raw = json.dumps(obj, ensure_ascii=False, default=str)
    return max(1, len(raw) // 3)


# ---------------------------------------------------------------- 剧本


def _tool(name: str, thought: str = "", **args: Any) -> dict[str, Any]:
    return {"tool": name, "args": args, "thought": thought}


def _final(text: str, thought: str = "信息已经够了，可以回答。", finish: str = "stop") -> dict[str, Any]:
    return {"final": text, "thought": thought, "finish": finish}


def scenario_happy() -> list[dict[str, Any]]:
    return [
        _tool("get_current_time", "先确认今天的日期，我自身没有时钟。"),
        _tool("calculator", "再算这个式子。", expression="(23*47)+19"),
        _final("今天是 2026-10-09；(23*47)+19 = 1100。"),
    ]


def scenario_error_recovery() -> list[dict[str, Any]]:
    return [
        _tool("read_file", "先去读那份笔记。", path="notes/does_not_exist.md"),
        _tool("list_files", "读不到，那我先看看目录里到底有什么。"),
        _tool("read_file", "用正确的文件名再读一次。", path="notes/agent.md"),
        _final("沙箱里的 agent.md 讲的是 ReAct 循环的三个出口和工具错误的回灌。"),
    ]


def scenario_bad_arguments() -> list[dict[str, Any]]:
    return [
        # ① 单引号 —— 模型把 JSON 写成了 Python 字面量
        {
            "tool": "calculator",
            "raw_args": "{'expression': '2+2'}",
            "thought": "先算 2+2。",
        },
        # ② 参数名写错（expr 而不是 expression）
        {
            "tool": "calculator",
            "raw_args": '{"expr": "2+2"}',
            "thought": "改个写法再试。",
        },
        # ③ 漏掉必填参数
        {"tool": "calculator", "raw_args": "{}", "thought": "直接空参数试试。"},
        # ④ 修对了
        _tool("calculator", "看来参数名是 expression，再来。", expression="2+2"),
        _final("2+2 = 4。前三次都失败了：一次 JSON 语法错、一次参数名错、一次漏参数。"),
    ]


def scenario_unknown_tool() -> list[dict[str, Any]]:
    return [
        _tool("web_search", "这个我得联网查一下。", query="DeepSeek function calling"),
        _tool("calculator", "没有联网工具，那就先算个例子。", expression="7*6"),
        _final("我并没有联网工具（web_search 不存在），所以只能用现有工具。7*6 = 42。"),
    ]


def scenario_tool_loop() -> list[dict[str, Any]]:
    # 一模一样地重试 —— 模型卡住时最典型的表现
    return [
        _tool("get_current_time", "我看看现在几点。"),
        _tool("get_current_time", "再看一次。"),
        _tool("get_current_time", "还没有想要的信息，再看一次。"),
        _tool("get_current_time", "继续看。"),
        _tool("get_current_time", "接着看。"),
        _final("（本来这里才会出答案，但循环检测应该提前打断）"),
    ]


def scenario_runaway() -> list[dict[str, Any]]:
    # 每次参数都不同，绕开重复检测 —— 只能用步数上限兜底
    return [
        _tool("calculator", f"第 {i} 步试探。", expression=f"{i}+{i}")
        for i in range(1, 40)
    ]


def scenario_context_bloat() -> list[dict[str, Any]]:
    return [
        _tool("read_file", "先把 agent.md 读进来。", path="notes/agent.md"),
        _tool("read_file", "再把 rag.md 读进来。", path="notes/rag.md"),
        _tool("read_file", "再把 agent.md 读一遍做对比。", path="notes/agent.md"),
        _tool("search_notes", "搜一下所有提到 token 的地方。", keyword="token"),
        _final("两份笔记都读完了。注意每一步都在把前面积累的全部内容重发一遍。"),
    ]


def scenario_truncated_final() -> list[dict[str, Any]]:
    long_text = (
        "根据我读到的内容，ReAct 循环的关键点包括：第一，循环必须有多个出口，"
        "不能只依赖模型主动停止；第二，工具失败必须以观察值的形式回灌，"
        "让模型有机会自我修正；第三，assistant 的 tool_calls 消息必须完整写回历"
    )
    return [_final(long_text, thought="直接总结一下就好。", finish="length")]


def scenario_text_defect() -> list[dict[str, Any]]:
    return [
        # 模型忘了写 Action / Final Answer，直接开始说话
        {"raw_text": "我打算先算一下这个式子，看起来是个乘法。", "thought": "格式跑偏演示"},
        _tool("calculator", "好了，按格式来。", expression="6*7"),
        # Action Input 用单引号，且塞在代码块里
        {
            "raw_text": "Thought: 再确认一次。\n```\nAction: calculator\nAction Input: {'expression': '1+1'}\n```",
            "thought": "格式跑偏演示 2",
        },
        _tool("calculator", "规规矩矩写一次。", expression="1+1"),
        _final("6*7 = 42，1+1 = 2。这次演示了文本模式下两种典型的格式翻车。"),
    ]


SCENARIOS: dict[str, Any] = {
    "happy": scenario_happy,
    "error_recovery": scenario_error_recovery,
    "bad_arguments": scenario_bad_arguments,
    "unknown_tool": scenario_unknown_tool,
    "tool_loop": scenario_tool_loop,
    "runaway": scenario_runaway,
    "context_bloat": scenario_context_bloat,
    "truncated_final": scenario_truncated_final,
    "text_defect": scenario_text_defect,
}


def build_script(name: str) -> list[dict[str, Any]]:
    if name not in SCENARIOS:
        raise KeyError(f"没有这个剧本：{name}。可选：{sorted(SCENARIOS)}")
    return SCENARIOS[name]()


# ---------------------------------------------------------------- 渲染


def _render_final(thought: str, answer: str) -> str:
    return f"Thought: {thought}\nFinal Answer: {answer}"


def _render_tool_call(thought: str, name: str, raw_args: str) -> str:
    return f"Thought: {thought}\nAction: {name}\nAction Input: {raw_args}"


def _usage(prompt_tokens: int, completion_tokens: int) -> Any:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


# ---------------------------------------------------------------- 客户端


class _MockCompletions:
    def __init__(self, script: list[dict[str, Any]], *, seed: int = 42) -> None:
        self.script = list(script)
        self.pos = 0
        self.calls = 0
        self.rng = random.Random(seed)

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        if "model" not in payload:
            raise MockAPIError(400, "you must provide a model parameter")
        if "messages" not in payload:
            raise MockAPIError(400, "'messages' is a required property")

    def _next_step(self) -> dict[str, Any]:
        if self.pos < len(self.script):
            step = self.script[self.pos]
            self.pos += 1
            return step
        return {
            "final": "（mock 剧本已用尽，这是兜底回答）",
            "thought": "剧本跑完了。",
            "finish": "stop",
        }

    def create(self, **payload: Any) -> Any:
        self._validate(payload)
        self.calls += 1

        messages = payload["messages"]
        model = payload.get("model", "mock-model")
        native = payload.get("tools") is not None
        tools = payload.get("tools") or []

        # 关键：tools schema 每一轮都要重发，所以它也该计入 prompt 成本
        prompt_tokens = _estimate_tokens(messages) + _estimate_tokens(tools)
        step = self._next_step()

        if "raise" in step:
            raise MockAPIError(int(step["raise"]), str(step.get("message", "mock error")))

        thought = str(step.get("thought", ""))

        # ---- 最终答案 ----
        if step.get("final") is not None:
            answer = str(step["final"])
            finish = str(step.get("finish", "stop"))
            text = answer if native else _render_final(thought, answer)
            return self._completion(model, text, prompt_tokens, finish)

        # ---- 原样吐出（专门用来演格式跑偏）----
        if step.get("raw_text") is not None:
            return self._completion(model, str(step["raw_text"]), prompt_tokens, "stop")

        # ---- 工具调用 ----
        name = str(step.get("tool", ""))
        raw_args = step.get("raw_args")
        if raw_args is None:
            raw_args = json.dumps(step.get("args", {}), ensure_ascii=False)

        if native:
            tool_call = SimpleNamespace(
                id=f"call_{self.calls}",
                type="function",
                function=SimpleNamespace(name=name, arguments=raw_args),
            )
            completion_tokens = _estimate_tokens(raw_args) + 8
            return SimpleNamespace(
                id=f"mock-{self.calls}",
                model=model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            role="assistant",
                            # 真实厂商在 tool_calls 轮次里 content 是 None，这里如实照做
                            content=None,
                            tool_calls=[tool_call],
                        ),
                    )
                ],
                usage=_usage(prompt_tokens, completion_tokens),
            )

        text = _render_tool_call(thought, name, raw_args)
        return self._completion(model, text, prompt_tokens, "stop")

    def _completion(self, model: str, text: str, prompt_tokens: int, finish: str) -> Any:
        completion_tokens = max(1, _estimate_tokens(text))
        return SimpleNamespace(
            id=f"mock-{self.calls}",
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=finish,
                    message=SimpleNamespace(role="assistant", content=text),
                )
            ],
            usage=_usage(prompt_tokens, completion_tokens),
        )


class MockClient:
    """形状与 openai.OpenAI 一致，所以能直接塞进 ReactAgent(client=...)。"""

    def __init__(self, script: list[dict[str, Any]], *, seed: int = 42) -> None:
        self.chat = SimpleNamespace(completions=_MockCompletions(script, seed=seed))


if __name__ == "__main__":
    print("可用剧本：")
    for key, fn in SCENARIOS.items():
        print(f"  {key:18} {len(fn())} 步")
