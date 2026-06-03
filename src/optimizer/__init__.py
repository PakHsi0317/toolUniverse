"""Tool Optimizer: refines a ToolSpec by testing LLM invocation, diagnosing
failures, and rewriting only the field(s) at fault.

Components:
    TestPromptGenerator  — generates NL prompts that should trigger the tool
    InvocationTester     — runs the LLM tool-call test, returns FailureRecord
    FailureDiagnoser     — fills blamed_field/root_cause/suggested_rewrite
    SpecRewriter         — applies a SuggestedRewrite to produce a new ToolSpec
    OptimizerLoop        — orchestrates up to 3 rounds, tracks accuracy
"""

from .failure_diagnoser import DiagnosisResult, FailureDiagnoser
from .invocation_tester import (
    CallDimensions,
    InvocationResult,
    InvocationTester,
    diagnose_dimensions,
)
from .loop import IterationResult, OptimizationResult, OptimizerLoop
from .spec_rewriter import SpecRewriter
from .test_prompt_generator import PromptWithGroundTruth, TestPromptGenerator

__all__ = [
    "CallDimensions",
    "DiagnosisResult",
    "FailureDiagnoser",
    "InvocationResult",
    "InvocationTester",
    "IterationResult",
    "OptimizationResult",
    "OptimizerLoop",
    "PromptWithGroundTruth",
    "SpecRewriter",
    "TestPromptGenerator",
    "diagnose_dimensions",
]
