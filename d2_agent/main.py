from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agent import (
    AgentError,
    AgentResult,
    ReactAgent,
)
from mock_backend import SCENARIOS, MockClient, build_script
from tools import REGISTRY, tool_schema
from dotenv import load_dotenv

load_dotenv()

ENV_PATH = Path(__file__).with_name(".env")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

class C:
    """极简 ANSI 着色。设了 NO_COLOR 或者输出不是终端就自动关掉。"""

    _on = sys.stdout.isatty() and not os.getenv("NO_COLOR")

    @classmethod
    def _wrap(cls, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls._on else text

    @classmethod
    def dim(cls, t: str) -> str:
        return cls._wrap("2", t)

    @classmethod
    def bold(cls, t: str) -> str:
        return cls._wrap("1", t)

    @classmethod
    def cyan(cls, t: str) -> str:
        return cls._wrap("36", t)

    @classmethod
    def green(cls, t: str) -> str:
        return cls._wrap("32", t)

    @classmethod
    def yellow(cls, t: str) -> str:
        return cls._wrap("33", t)

    @classmethod
    def red(cls, t: str) -> str:
        return cls._wrap("31", t)

    @classmethod
    def magenta(cls, t: str) -> str:
        return cls._wrap("35", t)


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def _clip(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + C.dim(f"  …（共 {len(text)} 字符）")


def make_printer(show_full_observation: bool = False):
    """把 agent 的事件流渲染成人能读的轨迹。"""

    def printer(kind: str, payload: dict[str, Any]) -> None:
        if kind == "thinking":
            print(C.dim(f"\n── 第 {payload['index']} 步 ── 让模型看当前上下文，问它下一步干嘛"))
            return

        if kind == "tool_call":
            print(f"  {C.cyan('Action')}     {C.bold(payload['name'])}")
            print(f"  {C.dim('Input')}      {payload['arguments']}")
            return

        if kind == "observation":
            rec = payload["record"]
            tag = C.green("Observation") if rec.ok else C.red(f"Observation ✗{rec.error_kind}")
            body = _clip(rec.observation, 100000 if show_full_observation else 400)
            print(f"  {tag} {body}")
            return

        if kind == "parsed":
            turn = payload["turn"]
            if turn.thought:
                print(f"  {C.magenta('Thought')}    {_clip(turn.thought, 200)}")
            return

        if kind == "final":
            pass  # 最终答案在打印结果时统一输出
            return

        if kind == "stop":
            reason = payload["reason"]
            text = {
                "max_steps": "到达步数上限 —— 模型一直没给最终答案",
                "tool_loop": "重复调用检测触发 —— 模型在用同样的参数原地打转",
                "token_budget": "超出 token 预算 —— 上下文被撑爆了",
            }.get(reason, reason)
            print(C.yellow(f"\n■ 循环中止：{text}"))
            return

    return printer

def print_result(result: AgentResult, agent: ReactAgent) -> None:
    print("\n" + C.bold("═" * 62))
    print(C.bold("最终答案"))
    print("═" * 62)
    print(result.answer or "（空）")

    print("\n" + C.bold("轨迹"))
    if not result.steps:
        print("  （没有产生任何步骤）")
    for s in result.steps:
        mark = C.green("✓") if s.ok else C.red("✗")
        thought = (s.thought or "").replace("\n", " ")[:40]
        print(
            f"  {s.index:>2} {mark} "
            f"{C.cyan(s.tool_name or '(无工具)'):<18} "
            f"{C.dim((s.tool_args or '').replace(chr(10), ' ')[:34]):<36} "
            f"{C.dim(f'prompt={s.prompt_tokens} {s.latency_s:.2f}s')}"
            + (C.dim(f"  「{thought}」") if thought else "")
        )

    print("\n" + C.bold("记账"))
    print(f"  循环步数        {len(result.steps)}")
    print(f"  停止原因        {result.stopped_reason}")
    print(f"  prompt tokens   合计 {result.prompt_tokens}")
    print(f"  completion      合计 {result.completion_tokens}")
    if result.steps:
        first = result.steps[0].prompt_tokens
        last = result.steps[-1].prompt_tokens
        print(
            f"  首轮 → 末轮     {first} → {last}"
            + C.dim(f"（涨了 {last - first}，因为每轮都在重发全部历史）")
        )
    if agent.mode == "native":
        print(
            C.dim(
                f"\n  提示：这一轮还额外重发了 {len(tool_schema())} 个工具的 schema，"
                "它们每轮都算钱。"
            )
        )


# ---------------------------------------------------------------- API Key


def ensure_api_key(argv_key: str | None) -> str:
    key = argv_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return key
    print("没有找到 API Key。")
    print("可以去 https://platform.deepseek.com/api_keys 申请一个。\n")
    try:
        key = input("请粘贴你的 DeepSeek API Key：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        raise SystemExit(1)
    if not key:
        raise SystemExit("Key 是空的，退出。")
    return key


# ---------------------------------------------------------------- 主流程
def build_agent(args: argparse.Namespace) -> tuple[ReactAgent, str, str]:
    """返回(agent, 描述, 问题)"""
    if args.mock:
        scenario = args.scenario or "happy"
        script = build_script(scenario)
        client = MockClient(script)
        model = f"mock:{scenario}"
    else:
        api_key = ensure_api_key(args.key)
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120.0)
        model = args.model

    agent = ReactAgent(
        client,
        model=model,
        mode=args.mode,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_prompt_tokens=args.token_budget,
        max_observation_chars=args.max_observation,
        verbose=not args.quiet,
    )
    agent.on_event(make_printer(show_full_observation=args.show_observation))
    return agent, model, args.question or ""


def main() -> int:
    _force_utf8()
    load_dotenv(ENV_PATH)

    parser = argparse.ArgumentParser(
        description="Day 2：手写 ReAct 循环",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py --mock --scenario happy \"帮我算 (23*47)+19，今天几号\"\n"
            "  python main.py --mock --scenario tool_loop\n"
            "  python main.py --mock --scenario text_defect --mode text\n"
            "  python main.py \"把 tools.md 的要点总结成三句话\"\n"
        ),
    )
    parser.add_argument("question", nargs="*", help="要问的问题；省略则进入交互模式")
    parser.add_argument("--mock", action="store_true", help="用离线剧本，不联网不花钱")
    parser.add_argument("--scenario", help="mock 剧本名，见 --list-scenarios")
    parser.add_argument("--list-scenarios", action="store_true", help="列出所有 mock 剧本")
    parser.add_argument("--mode", choices=["native", "text"], default="native",
                        help="native=原生 tool_calls；text=文本式 ReAct（看解析器的苦）")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL)
    parser.add_argument("--key", help="直接传 API Key（不推荐，会留在命令历史里）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="调工具的场景建议 0，减少参数幻觉")
    parser.add_argument("--max-steps", type=int, default=8, help="循环步数上限（硬护栏）")
    parser.add_argument("--token-budget", type=int, default=12000,
                        help="单次请求 prompt token 上限，超了主动中止")
    parser.add_argument("--max-observation", type=int, default=2000,
                        help="单条工具结果回灌前的截断长度")
    parser.add_argument("--show-observation", action="store_true", help="打印完整工具结果")
    parser.add_argument("--show-tools", action="store_true", help="打印模型看到的工具清单后退出")
    parser.add_argument("--quiet", action="store_true", help="只输出最终答案")
    parser.add_argument("--dry-run", action="store_true", help="只打印第一次请求体，不发请求")
    args = parser.parse_args()

    if args.list_scenarios:
        print("可用剧本：\n")
        for name, fn in SCENARIOS.items():
            print(f"  {C.bold(name):<24} {len(fn())} 步")
        print("\n用法：python main.py --mock --scenario <名字>")
        return 0

    if args.show_tools:
        print("模型看到的工具清单（原生 tool_calls 模式）：")
        print(json.dumps(tool_schema(), ensure_ascii=False, indent=2))
        print(f"\n工具函数所在模块共注册了 {len(REGISTRY)} 个工具。")
        return 0

    args.question = " ".join(args.question).strip()

    if args.dry_run:
        agent, _, _ = build_agent(argparse.Namespace(**{**vars(args), "mock": args.mock or True}))
        payload = agent._build_payload(
            [
                {"role": "user", "content": "（dry-run 占位问题）看看请求体长什么样"},
            ]
        )
        print("第一次请求的真实请求体：\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        print(
            C.dim(
                "\n注意两件事：\n"
                "  1. tools 字段把每个工具的描述和参数 schema 都发过去了 —— 这是成本的一部分\n"
                "  2. 这里还只有一条 user 消息，后面每轮都会把历史完整重发一遍"
            )
        )
        return 0

    agent, model, question = build_agent(args)

    if not args.mock and "reasoner" in model:
        print(
            C.red(
                "注意：deepseek-reasoner **不支持 Function Calling**。\n"
                "      官方文档明确列为 Not Supported Features，传 tools 进去不会生效。\n"
                f"      想跑 ReAct，请用 {DEFAULT_MODEL} 这类对话模型。\n"
            )
        )
        return 2

    header = (
        f"模式 {C.bold(args.mode)}｜模型 {C.bold(model)}｜"
        f"步数上限 {args.max_steps}｜token 预算 {args.token_budget}"
    )
    print(header)
    print(C.dim("输入问题开始；空行退出。\n"))

    def run_once(q: str) -> AgentResult | None:
        try:
            result = agent.run(q)
        except AgentError as exc:
            print(C.red(f"\nAgent 终止：{type(exc).__name__}: {exc}\n"))
            return None
        except Exception as exc:  # noqa: BLE001
            print(C.red(f"\n未预期的错误：{type(exc).__name__}: {exc}\n"))
            return None
        print_result(result, agent)
        return result

    if question:
        run_once(question)
        return 0

    while True:
        try:
            q = input(C.bold("你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        if q in {"/exit", "/quit"}:
            break
        if q == "/tools":
            print(json.dumps(tool_schema(), ensure_ascii=False, indent=2))
            continue
        run_once(q)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())