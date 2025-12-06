import re
from typing import Dict, Any

class CalculatorTool:

    def __init__(self):
        self.name = 'calculator'
        self.description = 'Performs mathematical calculations. Input should be a valid mathematical expression.'

    def execute(self, expression: str) -> Dict[str, Any]:
        try:
            safe_expression = re.sub('[^0-9+\\-*/().\\s]', '', expression)
            result = eval(safe_expression, {'__builtins__': {}}, {})
            return {'success': True, 'expression': expression, 'result': result}
        except Exception as e:
            return {'success': False, 'expression': expression, 'error': str(e)}