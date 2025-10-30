"""Calculator tool for mathematical operations."""
import re
from typing import Dict, Any


class CalculatorTool:
    """Tool for performing mathematical calculations."""

    def __init__(self):
        self.name = "calculator"
        self.description = "Performs mathematical calculations. Input should be a valid mathematical expression."

    def execute(self, expression: str) -> Dict[str, Any]:
        """Execute a mathematical calculation.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Dict containing the result or error
        """
        try:
            # Remove any potentially dangerous characters
            safe_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)

            # Evaluate the expression
            result = eval(safe_expression, {"__builtins__": {}}, {})

            return {
                "success": True,
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "error": str(e)
            }
