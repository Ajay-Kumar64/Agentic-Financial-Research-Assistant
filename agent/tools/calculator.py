import ast
import operator
import math
from typing import Any, Dict


class FinancialCalculatorTool:
    name = "financial_calculator"
    description = "Perform mathematical calculations: growth rates, ratios, percentages, CAGR."

    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def run(self, expression: str) -> Dict[str, Any]:
        try:
            result = self._eval(expression)
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "result": None,
                "error": str(e)
            }

    def _eval(self, node_or_expr):
        if isinstance(node_or_expr, str):
            tree = ast.parse(node_or_expr.strip(), mode='eval')
            return self._eval(tree.body)

        if isinstance(node_or_expr, ast.Num):
            return node_or_expr.n
        elif isinstance(node_or_expr, ast.Constant):
            return node_or_expr.value
        elif isinstance(node_or_expr, ast.BinOp):
            op_type = type(node_or_expr.op)
            if op_type not in self._SAFE_OPS:
                raise ValueError(f"Unsupported operation: {op_type}")
            return self._SAFE_OPS[op_type](
                self._eval(node_or_expr.left),
                self._eval(node_or_expr.right)
            )
        elif isinstance(node_or_expr, ast.UnaryOp):
            op_type = type(node_or_expr.op)
            if op_type not in self._SAFE_OPS:
                raise ValueError(f"Unsupported unary op: {op_type}")
            return self._SAFE_OPS[op_type](self._eval(node_or_expr.operand))
        elif isinstance(node_or_expr, ast.Call):
            return self._eval_call(node_or_expr)
        elif isinstance(node_or_expr, ast.Expression):
            return self._eval(node_or_expr.body)
        else:
            raise ValueError(f"Unsupported node: {type(node_or_expr)}")

    def _eval_call(self, node: ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        args = [self._eval(arg) for arg in node.args]

        if func_name == "growth_rate" and len(args) == 2:
            return ((args[1] - args[0]) / args[0]) * 100
        elif func_name == "cagr" and len(args) == 3:
            return ((args[1] / args[0]) ** (1 / args[2]) - 1) * 100
        elif func_name == "ratio" and len(args) == 2:
            return args[0] / args[1]
        elif func_name == "percentage" and len(args) == 2:
            return (args[0] / args[1]) * 100
        else:
            raise ValueError(f"Unknown function: {func_name} with args {args}")


# Global instance
calc_tool = FinancialCalculatorTool()