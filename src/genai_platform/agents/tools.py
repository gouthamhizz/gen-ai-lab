"""STEP 8a - tools the agent may call. A tool = name + one-line description (shown to the LLM) + a Python function.

Security note: the calculator uses an AST whitelist, NEVER eval(). An LLM-controlled string passed to eval()
is remote code execution.
"""
from __future__ import annotations

import ast
import math
import operator as op
from dataclasses import dataclass
from typing import Callable

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: op.pow}
_UN = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "sin": math.sin, "cos": math.cos,
          "tan": math.tan, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    def ev(n: ast.AST):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    return ev(ast.parse(expr.strip().replace("^", "**"), mode="eval"))


def calculator(expression: str) -> str:
    try:
        v = safe_eval(expression)
        return str(int(v)) if isinstance(v, float) and v.is_integer() and abs(v) < 1e15 else f"{v:.10g}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def wikipedia_search(query: str) -> str:
    try:
        import wikipedia
    except ImportError:
        return "Error: the `wikipedia` package is not installed"
    try:
        return wikipedia.summary(query, sentences=4, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return f"'{query}' is ambiguous. Try one of: {', '.join(e.options[:6])}"
    except wikipedia.PageError:
        try:
            alts = wikipedia.search(query, results=4)
        except Exception:  # noqa: BLE001
            alts = []
        return f"No page for '{query}'." + (f" Did you mean: {', '.join(alts)}?" if alts else "")
    except Exception as e:  # noqa: BLE001 - network errors become observations, not crashes
        return f"Error contacting Wikipedia: {e}"


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.fn(arg)


def rag_tool(retriever, k: int = 4) -> Tool:
    def run(q: str) -> str:
        hits = retriever.retrieve(q, k)
        return "\n".join(f"[{i}] ({h.cite}) {h.text[:500]}" for i, h in enumerate(hits, 1)) or "No relevant passages."

    return Tool("rag_search", "Search the user's ingested documents (PDF/HTML). Input: a search query. "
                              "Use for anything about the user's own documents.", run)


def default_tools(retriever=None) -> list[Tool]:
    tools = [
        Tool("wikipedia", "Look up general knowledge on Wikipedia. Input: a topic or entity name.", wikipedia_search),
        Tool("calculator", "Evaluate a math expression, e.g. '(17*23)+sqrt(144)'. Input: the expression only.", calculator),
    ]
    if retriever is not None:
        tools.append(rag_tool(retriever))
    return tools
