from __future__ import annotations

import ast
import json
import operator
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# 工具只能在这棵目录树里活动。真实项目里也建议这样 —— 显式白名单根目录。
SANDBOX_DIR = Path(__file__).with_name("sandbox")

MAX_OBSERVATION_CHARS = 2000

REGISTRY: dict[str, Tool] = {}

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    dangerous: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

def tool(name: str, description: str, parameters: dict[str, Any], dangerous: bool = False) -> Callable[[Callable[..., Any]], Tool]:
    """装饰器注册工具函数。"""
    def decorator(func: Callable[..., Any]) -> Tool:
        tool_instance = Tool(name=name, description=description, parameters=parameters, func=func, dangerous=dangerous)
        REGISTRY[name] = tool_instance
        return tool_instance
    return decorator

def get_tool(name: str) -> Tool:
    """根据名称获取工具实例。"""
    if name not in REGISTRY:
        raise ValueError(f"Tool '{name}' is not registered.")
    return REGISTRY[name]

def tool_schema() -> list[dict[str, Any]]:
    """返回所有注册工具的 schema 列表。"""
    return [tool.schema() for tool in REGISTRY.values()]

def render_tools_as_text() -> str:
    """将所有工具的描述渲染为文本。"""
    lines = []
    for tool in REGISTRY.values():
        lines.append(f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.parameters)}\n")
    return "\n".join(lines)


class ToolArgumentError(ValueError):
    """工具参数错误异常。"""
    pass

def parse_and_validate(tool_def: Tool, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
    """把模型给的 arguments 变成真正能用的 kwargs。

    模型的 arguments 是一个 **JSON 字符串**（不是对象！）。
    这是最容易踩的第一个坑，也是幻觉的高发区：

      · 给你单引号的 Python 字面量        {'expression': '1+1'}
      · 给你包了代码块                     ```json {...} ```
      · 把整个参数包成字符串 '"{\\"a\\": 1}"'
      · 参数名写错 / 编一个不存在的参数
    """
    if isinstance(raw_arguments, dict):
        data = raw_arguments
    else:
        text = (raw_arguments or "").strip()
        # 剥掉模型爱加的代码块围栏
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # 只截取最外层的一对花括号，掐掉前后寒暄
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            text = text[i : j + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(
                f"arguments 不是合法 JSON（{exc.msg}）。你给的是：{raw_arguments!r}。"
                "请只输出一个扁平的 JSON 对象，键名和类型必须与工具定义一致。"
            ) from exc

    if not isinstance(data, dict):
        raise ToolArgumentError(f"arguments 必须是 JSON 对象，你给的是 {type(data).__name__}。")

    props: dict[str, Any] = tool_def.parameters.get("properties", {})
    required = set(tool_def.parameters.get("required", []))

    missing = required - set(data)
    if missing:
        raise ToolArgumentError(
            f"缺少必填参数：{sorted(missing)}。该工具的参数定义是 "
            f"{json.dumps(tool_def.parameters, ensure_ascii=False)}"
        )

    extra = set(data) - set(props)
    if extra:
        # 严格模式：多余参数直接拒。宽松模式会让模型养成"多塞几个试试"的坏习惯。
        raise ToolArgumentError(
            f"传了未定义的参数：{sorted(extra)}。只允许：{sorted(props)}。"
        )

    return data

# ---------------------------------------------------------------- 工具 1：计算器


_SAFE_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """只走白名单语法树的求值器。

    为什么不用 eval()？因为表达式来自模型，而模型可以被提示词注入带偏。
    eval("__import__('os').system('rm -rf /')") 是真的能跑起来的。
    "模型不会那么坏"不是安全边界，"不支持" 才是。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"表达式里出现了不被允许的语法：{type(node).__name__}")


@tool(
    name="calculator",
    description=(
        "对数学表达式求值。支持 + - * / // % ** 和括号。"
        "只接受纯数字算式，不接受变量、函数调用或单位。"
        "需要多步计算时，一次只算一步，用上一步的结果组成下一步的表达式。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": '要计算的算式，例如 "(23*47)+19"',
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("表达式过长（超过 200 字符），请拆成多步计算。")
    tree = ast.parse(expression, mode="eval")
    value = _safe_eval(tree)
    # 防止 9**9**9 这种把 CPU 烧穿的输入
    if abs(value) > 1e15:
        raise ValueError(f"结果过大（{value:.3g}），疑似溢出，请检查算式。")
    if value.is_integer():
        return f"{expression} = {int(value)}"
    return f"{expression} = {value:.6g}"


# ---------------------------------------------------------------- 工具 2：时间


@tool(
    name="get_current_time",
    description="获取当前的本地日期与时间。模型自身没有时钟，任何和时间有关的问题都应该调用它。",
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_current_time() -> str:
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return now.strftime(f"%Y-%m-%d %H:%M:%S 星期{weekday}")


# ---------------------------------------------------------------- 工具 3：读文件


def _resolve_in_sandbox(raw_path: str) -> Path:
    """把模型给的路径解析到沙箱内，越界一律拒绝。

    这一段是整个工具层里最不能省的地方。三个必须挡的攻击：
      · 绝对路径            C:\\Windows\\System32\\config\\SAM
      · 相对回退            ../../../../etc/passwd
      · 软链接逃逸          sandbox/link -> /etc
    最后的 .resolve() + is_relative_to 就是用来把软链接也一起掐掉的。
    """
    if not raw_path or not raw_path.strip():
        raise ValueError("path 不能为空。")
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        raise ValueError(
            f"不接受绝对路径：{raw_path}。请使用相对沙箱根目录的路径，例如 notes/rag.md"
        )
    target = (SANDBOX_DIR / candidate).resolve()
    root = SANDBOX_DIR.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"路径越界，已被拒绝：{raw_path}。只允许访问 sandbox/ 目录内部。")
    return target


@tool(
    name="read_file",
    description=(
        "读取沙箱目录内的一个文本文件。路径必须相对于沙箱根目录，例如 notes/rag.md。"
        "文件不存在时会返回明确的错误，此时应该先用 list_files 看清有哪些文件，不要反复猜路径。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对于沙箱根目录的文件路径，例如 notes/agent.md",
            }
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = _resolve_in_sandbox(path)
    if not target.exists():
        available = sorted(
            str(p.relative_to(SANDBOX_DIR)).replace("\\", "/")
            for p in SANDBOX_DIR.rglob("*")
            if p.is_file()
        )
        raise FileNotFoundError(
            f"文件不存在：{path}。沙箱里现有的文件是：{available or '（空）'}"
        )
    if target.is_dir():
        raise IsADirectoryError(f"{path} 是目录，不是文件。用 list_files 列出它下面的内容。")
    text = target.read_text(encoding="utf-8", errors="replace")
    return f"【{path}】共 {len(text)} 字符\n{text}"


@tool(
    name="list_files",
    description="列出沙箱目录下的所有文件（相对路径）。不确定有哪些文件可用时先调用它。",
    parameters={
        "type": "object",
        "properties": {
            "subdir": {
                "type": "string",
                "description": "可选。只列出这个子目录，留空表示整个沙箱。",
            }
        },
        "required": [],
    },
)
def list_files(subdir: str = "") -> str:
    base = _resolve_in_sandbox(subdir) if subdir else SANDBOX_DIR.resolve()
    if not base.exists():
        raise FileNotFoundError(f"目录不存在：{subdir}")
    files = sorted(
        str(p.relative_to(SANDBOX_DIR)).replace("\\", "/")
        for p in base.rglob("*")
        if p.is_file()
    )
    if not files:
        return "（没有找到任何文件）"
    return "\n".join(files)


# ---------------------------------------------------------------- 工具 4：关键词检索


@tool(
    name="search_notes",
    description=(
        "在沙箱内的所有文本文件里做关键词检索，返回命中的文件名和所在行。"
        "想知道某个概念记在哪一篇里时用它，比逐个 read_file 便宜得多。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "要搜索的关键词，区分大小写"},
            "max_hits": {
                "type": "integer",
                "description": "最多返回多少条命中，默认 6",
            },
        },
        "required": ["keyword"],
    },
)
def search_notes(keyword: str, max_hits: int = 6) -> str:
    if not keyword.strip():
        raise ValueError("keyword 不能为空。")
    hits: list[str] = []
    for path in sorted(SANDBOX_DIR.rglob("*")):
        if not path.is_file():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if keyword in line:
                rel = str(path.relative_to(SANDBOX_DIR)).replace("\\", "/")
                hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
                if len(hits) >= max_hits:
                    return "\n".join(hits) + f"\n（已截断，只显示前 {max_hits} 条）"
    return "\n".join(hits) if hits else f"没有任何文件包含关键词「{keyword}」。"


# ---------------------------------------------------------------- 工具 5：联网（桩）


_ALLOWED_HOSTS = {"docs.python.org", "api-docs.deepseek.com"}


@tool(
    name="fetch_url",
    description=(
        "抓取一个网页的纯文本内容。仅允许白名单域名，且只支持 http/https。"
        "当前是离线桩实现，用于演示工具该有哪些护栏，不会真的发网络请求。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的完整 URL"},
            "max_chars": {
                "type": "integer",
                "description": "最多返回多少字符，默认 1500",
            },
        },
        "required": ["url"],
    },
    dangerous=True,
)
def fetch_url(url: str, max_chars: int = 1500) -> str:
    """真实实现需要的护栏，一条都不能少：

      · 协议白名单      只允许 http/https，否则 file:// 能读本机文件
      · 域名白名单      否则就是一条 SSRF 通道，能替你去打内网服务
      · 超时            否则一个不响应的站点就能把整个 agent 卡死
      · 响应体大小上限   否则一个 2GB 的响应直接把你的账单和内存一起干掉
      · 重定向次数上限   否则可以构造无限重定向
      · 内容类型检查     只收 text/*，二进制一律拒绝
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"只支持 http/https，收到的是：{url}")
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0].lower()
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"域名 {host} 不在白名单里。允许的域名：{sorted(_ALLOWED_HOSTS)}"
        )
    body = (
        f"（离线桩）这里是 {url} 的正文摘录。真实实现会在此处返回抓取到的页面文本，"
        "并在超过 max_chars 时截断。"
    )
    return body[:max_chars]


# ---------------------------------------------------------------- 执行入口


@dataclass
class ToolOutcome:
    """一次工具执行的结果。ok=False 表示失败，但**失败也是一种 observation**。"""

    ok: bool
    observation: str  # 准备回灌给模型的文本
    error_kind: str | None  # "unknown_tool" | "bad_arguments" | "tool_error" | None
    latency_s: float
    raw_chars: int  # 截断前的长度，用于观察"工具结果有多贵"


def execute_tool(
    name: str,
    raw_arguments: str | dict[str, Any],
    *,
    max_chars: int = MAX_OBSERVATION_CHARS,
) -> ToolOutcome:
    """执行一个工具。**永不抛异常** —— 所有失败都变成一段能回灌的文本。

    这是本文件里第二个关键设计。

    新手最常见的写法是让工具异常直接往上冒，结果整个 agent 崩掉，
    或者在 agent 里 try/except 之后只打一行日志、什么都没喂回去 —— 于是模型
    永远不知道自己调错了，会一直原样重试，直到 max_steps 耗尽。

    正确做法：把错误信息**当成工具的正常返回**喂回上下文。
    模型看到 "文件不存在，现有文件是 X/Y/Z"，下一轮大概率就改对了。
    """
    t0 = time.perf_counter()

    def done(ok: bool, text: str, kind: str | None) -> ToolOutcome:
        raw_len = len(text)
        if raw_len > max_chars:
            text = text[:max_chars] + f"\n…（结果已被截断，原始长度 {raw_len} 字符）"
        return ToolOutcome(ok, text, kind, time.perf_counter() - t0, raw_len)

    tool_def = get_tool(name)
    if tool_def is None:
        return done(
            False,
            f"错误：不存在名为 {name!r} 的工具。"
            f"可用工具只有：{sorted(REGISTRY)}。请不要编造工具名，"
            "如果需要的信息无法用现有工具获得，直接用 Final Answer 说明你做不到。",
            "unknown_tool",
        )

    try:
        kwargs = parse_and_validate(tool_def, raw_arguments)
    except ToolArgumentError as exc:
        return done(False, f"参数错误：{exc}", "bad_arguments")

    try:
        result = tool_def.func(**kwargs)
    except Exception as exc:  # noqa: BLE001 —— 这里就是要兜住一切
        return done(
            False,
            f"工具 {name} 执行失败：{type(exc).__name__}: {exc}",
            "tool_error",
        )

    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if not text.strip():
        # 空结果是个隐蔽的坑：模型会以为工具坏了，然后反复重试。
        text = "（工具执行成功，但返回内容为空）"
    return done(True, text, None)