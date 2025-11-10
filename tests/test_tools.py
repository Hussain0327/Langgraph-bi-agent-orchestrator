import pytest
from unittest.mock import Mock, patch
from src.tools.calculator import CalculatorTool


class TestCalculatorTool:
    
    @pytest.fixture
    def calculator(self):
        return CalculatorTool()
    
    def test_calculate_roi(self, calculator):
        result = calculator.calculate_roi(investment=10000, returns=15000)
        assert result == 50.0
    
    def test_calculate_roi_zero_investment(self, calculator):
        with pytest.raises(ValueError):
            calculator.calculate_roi(investment=0, returns=5000)
    
    def test_calculate_roi_negative_investment(self, calculator):
        result = calculator.calculate_roi(investment=-10000, returns=15000)
        assert result < 0
