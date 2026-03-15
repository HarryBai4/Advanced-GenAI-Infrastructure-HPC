import pytest
from src.optimization.quantization import ModelOptimizer

def test_optimizer_init():
    # Structural test for optimizer class
    opt = ModelOptimizer("mock-model")
    assert opt.model_id == "mock-model"

def test_quantization_flow():
    # Ensure quantization method returns expected status
    opt = ModelOptimizer("mock-model")
    assert opt.apply_awq_quantization() == True
