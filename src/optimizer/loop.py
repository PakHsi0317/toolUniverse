"""OptimizerLoop: orchestrates up to 3 iterations of test → diagnose → rewrite.

Termination conditions (any one stops the loop):
    1. accuracy reaches the target threshold (default 1.0 = 100%)
    2. max_iterations reached (default 3, per ToolUniverse paper)
    3. no fixable failures remain
    4. (do-no-harm guard) the proposed rewrite would LOWER accuracy on the
       current prompts — the change is rejected and the loop stops with the
       last known-good spec.

The do-no-harm guard exists because rewriting a confusable tool's name or
description can DIVERGE: each edit perturbs behaviour and, under heavy
multi-tool competition, a good spec can be driven from 80% down to 20% across
3 unchecked iterations. The guard validates each candidate rewrite on the same
prompts BEFORE committing it, so optimization can never regress below baseline.

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
    def needs_redesign(self) -> bool:
        """True when the loop concluded the spec has a STRUCTURAL defect that
        field-level rewriting cannot fix (vs. a field-level bug or a clean spec).
        Signals the defect should be routed back to the Discoverer for redesign."""
        return self.terminated_reason.startswith("needs_redesign")

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
        do_no_harm: bool = True,
        detect_redesign: bool = True,
    ):
        self.llm = llm
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_prompts = n_prompts
        self.max_iterations = max_iterations
        self.target_accuracy = target_accuracy
        # When True, a rewrite is committed only if it does not lower accuracy
        # on the current round's prompts. Prevents divergence on confusable tools.
        self.do_no_harm = do_no_harm
        # When True, the loop distinguishes a STRUCTURAL defect (field-level
        # rewriting cannot help — the tool's identity is wrong) from an ordinary
        # guard rejection, and reports terminated_reason="needs_redesign".
        self.detect_redesign = detect_redesign

        self.prompt_gen = TestPromptGenerator(llm)
        self.tester = InvocationTester(llm)
        self.diagnoser = FailureDiagnoser(llm)
        self.rewriter = SpecRewriter()

    def optimize(
        self,
        initial_spec: ToolSpec,
        competing_specs: Optional[list[ToolSpec]] = None,
        eval_prompts: Optional[list[str]] = None,
    ) -> OptimizationResult:
        """Optimize a spec via test → diagnose → rewrite.

        eval_prompts: optional HELD-OUT prompts (a real, external distribution)
        used ONLY to judge structural defects (needs_redesign). The loop's own
        adaptively-generated prompts are written for the current spec and can
        flatter it — a structural defect (over-broad tool scope) hides under
        self-generated prompts but shows under held-out ones. When provided,
        needs_redesign is decided on this independent set, not on self-tests.
        """
        spec = initial_spec
        iterations: list[IterationResult] = []
        prior_failures: list[FailureRecord] = []

        def _held_out_acc(s: ToolSpec) -> Optional[float]:
            """Accuracy on the held-out set, or None if no eval_prompts given."""
            if not eval_prompts:
                return None
            res = self.tester.test_batch(s, eval_prompts, competing_specs)
            return sum(1 for r in res if r.failure_type == "correct") / len(res)

        # Held-out baseline of the ORIGINAL spec — the honest reference point for
        # "did any field-level fix actually help?" (self-test scores can rise on
        # prompts written for the spec while held-out accuracy never moves).
        initial_held_out = _held_out_acc(initial_spec)

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
                # Self-generated prompts say we're done. But a structural defect
                # (over-broad tool) can ace prompts written FOR it while still
                # failing real, held-out tasks. Cross-check on the held-out set:
                # if it passes there too, genuinely done; if it fails there, the
                # spec only looks fixed → structural defect → needs_redesign.
                held = _held_out_acc(spec)
                if (self.detect_redesign and held is not None
                        and held < self.target_accuracy):
                    return OptimizationResult(
                        final_spec=spec, initial_spec=initial_spec,
                        iterations=iterations,
                        terminated_reason=(
                            f"needs_redesign (self-test {accuracy:.0%} but held-out "
                            f"{held:.0%}; tool scope only fits its own prompts)"
                        ),
                    )
                return OptimizationResult(
                    final_spec=spec, initial_spec=initial_spec,
                    iterations=iterations,
                    terminated_reason=f"target_reached ({accuracy:.0%})",
                )
            if it == self.max_iterations - 1:
                # Exhausted all rounds without reaching target. Judge "still
                # broken" on the held-out set when available (self-test flatters
                # the spec). If field-level fixes never lifted the held-out score,
                # the defect resisted all field-level repair → structural.
                held = _held_out_acc(spec)
                if held is not None:
                    # Honest test: did any fix raise held-out above where we
                    # started, and are we still below target? If not → structural.
                    judge_acc = held
                    baseline = initial_held_out if initial_held_out is not None else held
                    improved = judge_acc > baseline
                else:
                    judge_acc = accuracy
                    improved = accuracy > iterations[0].accuracy
                stuck = judge_acc < self.target_accuracy and not improved
                if self.detect_redesign and stuck and diagnoses_applied > 0:
                    score_str = (f"held-out {held:.0%}" if held is not None
                                 else f"{accuracy:.0%}")
                    reason = (
                        f"needs_redesign (no field-level fix lifted {score_str} "
                        f"over {self.max_iterations} rounds; structural defect)"
                    )
                else:
                    reason = f"max_iterations ({self.max_iterations})"
                return OptimizationResult(
                    final_spec=spec, initial_spec=initial_spec,
                    iterations=iterations,
                    terminated_reason=reason,
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
            candidate = self.rewriter.apply(spec, chosen.suggested_rewrite)

            # Do-no-harm guard: re-test the candidate on THIS round's prompts.
            # Commit only if it does not regress; otherwise keep the current
            # (known-good) spec and stop — the fix direction is harmful.
            if self.do_no_harm:
                check = self.tester.test_batch(candidate, prompts, competing_specs)
                cand_acc = (
                    sum(1 for r in check if r.failure_type == "correct") / len(check)
                    if check else 0.0
                )
                if cand_acc < accuracy:
                    # The diagnoser's best single-field fix (top_field) was just
                    # rejected because it doesn't help. If the spec is still
                    # failing, the field-level fix the diagnoser identified CANNOT
                    # repair it → the defect is STRUCTURAL (the tool's
                    # identity/scope is wrong), not a field bug. Judge "still
                    # failing" on the HELD-OUT set when available (self-generated
                    # prompts flatter the spec and hide structural defects).
                    judge_acc = _held_out_acc(spec)
                    if judge_acc is None:
                        judge_acc = accuracy
                    if self.detect_redesign and judge_acc < self.target_accuracy:
                        reason = (
                            f"needs_redesign (field-level fix to '{top_field}' "
                            f"cannot lift held-out {judge_acc:.0%}; structural defect)"
                        )
                    else:
                        reason = (
                            f"no_harm_guard (rejected rewrite to '{top_field}': "
                            f"{cand_acc:.0%} < {accuracy:.0%})"
                        )
                    return OptimizationResult(
                        final_spec=spec, initial_spec=initial_spec,
                        iterations=iterations,
                        terminated_reason=reason,
                    )

            spec = candidate
            prior_failures = failures  # feed back into next prompt generation

        # Should not reach here, but for safety:
        return OptimizationResult(
            final_spec=spec, initial_spec=initial_spec,
            iterations=iterations,
            terminated_reason="loop_exit",
        )
