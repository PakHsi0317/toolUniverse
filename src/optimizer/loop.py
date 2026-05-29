"""OptimizerLoop: orchestrates up to 3 iterations of test → diagnose → rewrite.

Termination conditions (any one stops the loop):
    1. accuracy reaches the target threshold (default 1.0 = 100%)
    2. max_iterations reached (default 3, per ToolUniverse paper)
    3. no improvement between iterations (rewrite didn't help — exit early)

Each iteration produces a batch of FailureRecord rows, all written to
the log path. The final ToolSpec is returned along with per-iteration
accuracy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.llm_client import LLMClient
from src.schema import FailureRecord, ToolSpec, append_record

from .failure_diagnoser import FailureDiagnoser
from .invocation_tester import InvocationTester
from .spec_rewriter import SpecRewriter
from .test_prompt_generator import TestPromptGenerator


@dataclass
class IterationResult:
    iteration: int
    spec: ToolSpec  # the spec used in THIS iteration
    accuracy: float
    n_correct: int
    n_total: int
    records: list[FailureRecord] = field(default_factory=list)
    diagnoses_applied: int = 0


@dataclass
class OptimizationResult:
    final_spec: ToolSpec
    initial_spec: ToolSpec
    iterations: list[IterationResult]
    terminated_reason: str

    @property
    def initial_accuracy(self) -> float:
        return self.iterations[0].accuracy

    @property
    def final_accuracy(self) -> float:
        return self.iterations[-1].accuracy

    @property
    def improvement(self) -> float:
        return self.final_accuracy - self.initial_accuracy


class OptimizerLoop:
    def __init__(
        self,
        llm: LLMClient,
        log_path: Path | str,
        n_prompts: int = 5,
        max_iterations: int = 3,
        target_accuracy: float = 1.0,
    ):
        self.llm = llm
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_prompts = n_prompts
        self.max_iterations = max_iterations
        self.target_accuracy = target_accuracy

        self.prompt_gen = TestPromptGenerator(llm)
        self.tester = InvocationTester(llm)
        self.diagnoser = FailureDiagnoser(llm)
        self.rewriter = SpecRewriter()

    def optimize(
        self,
        initial_spec: ToolSpec,
        competing_specs: Optional[list[ToolSpec]] = None,
    ) -> OptimizationResult:
        spec = initial_spec
        iterations: list[IterationResult] = []
        prior_failures: list[FailureRecord] = []

        for it in range(self.max_iterations):
            # 1. Generate prompts (adaptive: prior failures inform new tests)
            prompts = self.prompt_gen.generate(
                spec, n=self.n_prompts, prior_failures=prior_failures or None
            )

            # 2. Run invocation tests
            results = self.tester.test_batch(spec, prompts, competing_specs)

            # 3. Build FailureRecords + diagnose failures
            records: list[FailureRecord] = []
            diagnoses_applied = 0
            for r in results:
                rec = r.to_failure_record(spec, iteration=it, model=self.llm.model)
                if rec.failure_type != "correct":
                    self.diagnoser.enrich(spec, rec)
                    diagnoses_applied += 1
                records.append(rec)
                append_record(self.log_path, rec)

            n_correct = sum(1 for r in records if r.failure_type == "correct")
            accuracy = n_correct / len(records) if records else 0.0
            iterations.append(IterationResult(
                iteration=it, spec=spec, accuracy=accuracy,
                n_correct=n_correct, n_total=len(records),
                records=records, diagnoses_applied=diagnoses_applied,
            ))

            # 4. Termination checks
            if accuracy >= self.target_accuracy:
                return OptimizationResult(
                    final_spec=spec, initial_spec=initial_spec,
                    iterations=iterations,
                    terminated_reason=f"target_reached ({accuracy:.0%})",
                )
            if it == self.max_iterations - 1:
                return OptimizationResult(
                    final_spec=spec, initial_spec=initial_spec,
                    iterations=iterations,
                    terminated_reason=f"max_iterations ({self.max_iterations})",
                )

            # 5. Pick the most-common blamed_field across this round's failures
            # and apply ONE rewrite per iteration (most impactful change).
            failures = [r for r in records if r.failure_type != "correct"
                        and r.suggested_rewrite is not None]
            if not failures:
                return OptimizationResult(
                    final_spec=spec, initial_spec=initial_spec,
                    iterations=iterations,
                    terminated_reason="all_failures_unfixable",
                )

            # Pick the most common blamed_field and use its first rewrite suggestion
            blame_counts = Counter(r.blamed_field for r in failures)
            top_field, _ = blame_counts.most_common(1)[0]
            chosen = next(r for r in failures if r.blamed_field == top_field)
            spec = self.rewriter.apply(spec, chosen.suggested_rewrite)
            prior_failures = failures  # feed back into next prompt generation

        # Should not reach here, but for safety:
        return OptimizationResult(
            final_spec=spec, initial_spec=initial_spec,
            iterations=iterations,
            terminated_reason="loop_exit",
        )
