"""A calculator the agents must use for arithmetic, instead of doing it in their heads.

Expressions are parsed with ``ast`` and only numbers, + - * / ** % and a few functions are
allowed, so this is safe to expose to a model.
"""

from __future__ import annotations

import ast
import operator
import re

from equity_agents.tools.base import Tool, ToolError

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max}


def safe_eval(expression: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 10:
                raise ToolError("Exponent too large")
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError(f"Unsupported expression element: {ast.dump(node)[:60]}")

    try:
        # Drop thousands separators inside numbers ("60,922") but keep argument commas.
        tree = ast.parse(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression), mode="eval")
        return float(ev(tree))
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression: {exc.msg}") from exc


def calculate(expression: str) -> str:
    return f"{safe_eval(expression):.6g}"


CALC_TOOL = Tool(
    "calculate",
    "Evaluate an arithmetic expression, e.g. '(130497 - 60922) / 60922 * 100'. Use it for every "
    "growth rate, margin or ratio; never compute numbers mentally.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    calculate,
)
