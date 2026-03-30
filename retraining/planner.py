"""
Training planner: dataset analysis → strategy selection → structured plan.

Three-stage pipeline
--------------------
1. **Analysis** (reasoning)   – inspect dataset structure, format, vocabulary,
                                and content patterns; emit a reasoning trace.
2. **Selection** (abstraction) – score every registered TrainingStrategy against
                                 the analysis and pick the best fit.
3. **Planning** (planning)    – produce a TrainingPlan with named phases, per-phase
                                reasoning, recommended parameters, and token estimates.

The plan is stored alongside the job record so every training decision is
auditable and reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TrainingPhase(BaseModel):
    """A single named phase in the training plan."""

    name: str
    description: str
    reasoning: str
    steps: List[str]


class TrainingPlan(BaseModel):
    """
    Fully-resolved training plan produced by the planner.

    Stored in the job record so the rationale behind every training run
    is permanently auditable.
    """

    strategy_name: str
    strategy_description: str
    rationale: str
    phases: List[TrainingPhase]
    estimated_tokens: int
    recommended_parameters: Dict[str, Any]
    created_at: str


@dataclass
class DatasetAnalysis:
    """
    Structured result of dataset inspection.

    Produced by TrainingPlanner.analyse() and passed to each strategy's
    suitability() and rationale() methods so every scoring decision can be
    traced back to concrete observations.
    """

    format: str                    # "text" | "jsonl"
    entry_count: int
    avg_entry_length: float
    total_chars: int
    detected_keys: Set[str]        # top-level keys from first JSONL entry
    vocabulary_diversity: float    # unique_words / total_words  (0–1)
    has_code: bool
    reasoning: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TrainingPlanner:
    """
    Orchestrates the analysis → selection → plan pipeline.

    Usage::

        planner = TrainingPlanner()
        plan, strategy, analysis = planner.plan(content, fmt, payload)
        modelfile = strategy.render_modelfile(
            base_model, content, instructions, plan.recommended_parameters
        )
    """

    def __init__(self) -> None:
        # Import here to avoid circular imports at module load time
        from retraining.strategies import (
            DomainAdaptationStrategy,
            FewShotStrategy,
            InstructionTuningStrategy,
            SystemPromptStrategy,
        )
        # Ordered from most-specific to least-specific so ties break
        # in favour of richer structure
        self._strategies = [
            InstructionTuningStrategy(),
            FewShotStrategy(),
            DomainAdaptationStrategy(),
            SystemPromptStrategy(),
        ]

    # ------------------------------------------------------------------
    # Stage 1 – Reasoning / analysis
    # ------------------------------------------------------------------

    def analyse(self, content: str, fmt: str) -> DatasetAnalysis:
        """
        Inspect the dataset and emit a structured analysis with a reasoning
        trace that explains every observation.
        """
        reasoning: List[str] = []
        lines = [line for line in content.splitlines() if line.strip()]
        detected_keys: Set[str] = set()
        entry_lengths: List[int] = []

        # --- format-specific parsing ---
        if fmt == "jsonl":
            entries = []
            for line in lines:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip malformed lines

            if entries:
                detected_keys = set(entries[0].keys())
                reasoning.append(
                    f"Parsed {len(entries)} valid JSONL entries. "
                    f"Top-level keys in first entry: {sorted(detected_keys)}."
                )
                entry_lengths = [len(json.dumps(e)) for e in entries]
            else:
                reasoning.append(
                    "Format declared as 'jsonl' but no valid JSON lines found. "
                    "Falling back to plain-text analysis."
                )
                fmt = "text"
                entry_lengths = [len(l) for l in lines]
        else:
            entry_lengths = [len(l) for l in lines]
            reasoning.append(
                f"Plain-text dataset with {len(lines)} non-empty lines."
            )

        entry_count = len(entry_lengths)
        avg_length = sum(entry_lengths) / max(entry_count, 1)
        total_chars = len(content)

        # --- vocabulary analysis (proxy for knowledge density) ---
        words = re.findall(r"\b[a-zA-Z]\w*\b", content.lower())
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)

        if vocab_diversity > 0.6:
            reasoning.append(
                f"High vocabulary diversity ({vocab_diversity:.2f}) — "
                "dataset likely covers a broad domain or multiple topics."
            )
        else:
            reasoning.append(
                f"Moderate-to-low vocabulary diversity ({vocab_diversity:.2f}) — "
                "dataset appears focused on a narrow topic or style."
            )

        # --- code detection ---
        code_patterns = r"\bdef \b|\bclass \b|\bimport \b|\bfunction\b|\bconst \b|\blet \b|\bvar \b"
        has_code = bool(re.search(code_patterns, content))
        if has_code:
            reasoning.append(
                "Code patterns detected. Strategies that preserve exact formatting "
                "will be preferred."
            )

        # --- entry-length observations ---
        reasoning.append(
            f"Average entry length: {avg_length:.0f} chars "
            f"({total_chars} total chars, ~{total_chars // 4} estimated tokens)."
        )
        if avg_length > 1000:
            reasoning.append(
                "Entries are long; a larger context window (num_ctx) will be recommended."
            )

        return DatasetAnalysis(
            format=fmt,
            entry_count=entry_count,
            avg_entry_length=avg_length,
            total_chars=total_chars,
            detected_keys=detected_keys,
            vocabulary_diversity=vocab_diversity,
            has_code=has_code,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Stage 2 – Abstraction / strategy selection
    # ------------------------------------------------------------------

    def select_strategy(self, analysis: DatasetAnalysis):
        """
        Score every registered strategy and return the best match.
        Scores and the winner are logged at INFO level for observability.
        """
        scores: List[Tuple[float, Any]] = [
            (strategy.suitability(analysis), strategy)
            for strategy in self._strategies
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        score_summary = {s.name: f"{sc:.2f}" for sc, s in scores}
        logger.info("Strategy suitability scores: %s", score_summary)

        best_score, best_strategy = scores[0]
        logger.info(
            "Selected strategy '%s' (score %.2f). Rationale: %s",
            best_strategy.name,
            best_score,
            best_strategy.rationale(analysis),
        )
        return best_strategy

    # ------------------------------------------------------------------
    # Stage 3 – Planning
    # ------------------------------------------------------------------

    def build_plan(
        self,
        analysis: DatasetAnalysis,
        strategy,
        payload: Dict[str, Any],
    ) -> TrainingPlan:
        """
        Construct a TrainingPlan with four named phases.  Each phase carries
        its own reasoning string so the final plan is fully self-documenting.
        """
        explicit_params: Dict[str, Any] = payload.get("parameters") or {}

        # --- infer recommended context window from dataset size ---
        estimated_tokens = analysis.total_chars // 4
        if analysis.entry_count < 50:
            recommended_ctx = 4096
            ctx_reasoning = (
                f"Small dataset ({analysis.entry_count} entries, ~{estimated_tokens} tokens). "
                "A 4 k context window is sufficient and keeps inference fast."
            )
        elif analysis.entry_count < 200:
            recommended_ctx = 8192
            ctx_reasoning = (
                f"Medium dataset ({analysis.entry_count} entries, ~{estimated_tokens} tokens). "
                "An 8 k context window balances coverage and performance."
            )
        else:
            recommended_ctx = 16384
            ctx_reasoning = (
                f"Large dataset ({analysis.entry_count} entries, ~{estimated_tokens} tokens). "
                "A 16 k context window is recommended to fit all training context."
            )

        recommended: Dict[str, Any] = {"num_ctx": recommended_ctx}
        recommended.update(explicit_params)  # caller-supplied params take precedence

        phases = [
            TrainingPhase(
                name="dataset_analysis",
                description="Inspect dataset structure, format, and linguistic characteristics.",
                reasoning="\n".join(analysis.reasoning),
                steps=[
                    f"Format: {analysis.format}",
                    f"Entry count: {analysis.entry_count}",
                    f"Average entry length: {analysis.avg_entry_length:.0f} chars",
                    f"Total chars: {analysis.total_chars} (~{estimated_tokens} tokens)",
                    f"Vocabulary diversity: {analysis.vocabulary_diversity:.2f}",
                    f"Code detected: {analysis.has_code}",
                    f"Detected JSONL keys: {sorted(analysis.detected_keys) or 'n/a'}",
                ],
            ),
            TrainingPhase(
                name="strategy_selection",
                description=(
                    f"Evaluate all registered strategies and select '{strategy.name}' "
                    "as the best fit for this dataset."
                ),
                reasoning=strategy.rationale(analysis),
                steps=[
                    f"Winning strategy: {strategy.name}",
                    f"Strategy description: {strategy.description}",
                    "All other strategies were scored and ranked; this one scored highest.",
                    "Modelfile will be generated using this strategy's template.",
                ],
            ),
            TrainingPhase(
                name="parameter_configuration",
                description="Determine Ollama PARAMETER directives for optimal training performance.",
                reasoning=ctx_reasoning + (
                    " Explicit parameters from the request override defaults." if explicit_params else ""
                ),
                steps=[f"PARAMETER {k} {v}" for k, v in recommended.items()],
            ),
            TrainingPhase(
                name="modelfile_generation",
                description="Render the Modelfile and submit to Ollama /api/create.",
                reasoning=(
                    "The Modelfile is written to the job directory before submission so "
                    "it can be inspected, audited, or replayed independently."
                ),
                steps=[
                    "Render Modelfile using selected strategy and resolved parameters.",
                    "Write rendered Modelfile to <job_dir>/Modelfile.",
                    "POST to Ollama /api/create (streaming or blocking per request flag).",
                    "Poll / stream response and write log entries to <job_dir>/logs.ndjson.",
                ],
            ),
        ]

        return TrainingPlan(
            strategy_name=strategy.name,
            strategy_description=strategy.description,
            rationale=strategy.rationale(analysis),
            phases=phases,
            estimated_tokens=estimated_tokens,
            recommended_parameters=recommended,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Convenience entry-point
    # ------------------------------------------------------------------

    def plan(
        self,
        content: str,
        fmt: str,
        payload: Dict[str, Any],
    ) -> Tuple[TrainingPlan, Any, DatasetAnalysis]:
        """
        Full pipeline: analyse → select → build.

        Returns ``(training_plan, strategy, analysis)`` so callers can use
        the strategy directly to render the Modelfile.
        """
        analysis = self.analyse(content, fmt)
        strategy = self.select_strategy(analysis)
        training_plan = self.build_plan(analysis, strategy, payload)
        return training_plan, strategy, analysis
