"""
Training strategies for Modelfile generation.

Each strategy encapsulates a distinct approach to teaching a base model new
behaviour via an Ollama Modelfile.  The planner scores every strategy against
the analysed dataset and picks the highest-scoring one.

Strategy hierarchy
------------------
TrainingStrategy (ABC)
├── SystemPromptStrategy      – free-form text injected as SYSTEM block
├── FewShotStrategy           – QA / input-output pairs as MESSAGE turns
├── InstructionTuningStrategy – instruction/response pairs (OpenAI-style)
└── DomainAdaptationStrategy  – structured topic/content entries assembled
                                into a rich SYSTEM block
"""

from __future__ import annotations

import json
import logging
import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from retraining.planner import DatasetAnalysis

logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """Abstract base class for all training strategies."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def suitability(self, analysis: DatasetAnalysis) -> float:
        """Return a score in [0, 1] indicating fit for this dataset analysis."""

    @abstractmethod
    def render_modelfile(
        self,
        base_model: str,
        dataset_content: str,
        instructions: str,
        parameters: Dict[str, Any],
    ) -> str:
        """Produce a complete Modelfile string."""

    @abstractmethod
    def rationale(self, analysis: DatasetAnalysis) -> str:
        """Explain in plain text why this strategy suits the dataset."""


class SystemPromptStrategy(TrainingStrategy):
    """
    Injects the entire dataset as a SYSTEM block.

    Best for: free-form prose, general domain knowledge, narrative text.
    The model receives the content as persistent background context on every
    inference call.
    """

    name = "system_prompt"
    description = (
        "Injects dataset content as a SYSTEM block; best for general "
        "knowledge augmentation with free-form text."
    )

    def suitability(self, analysis: DatasetAnalysis) -> float:
        if analysis.format == "text":
            return 0.85
        # JSONL with long entries and no structured keys → treat as prose
        if analysis.format == "jsonl" and not analysis.detected_keys & {
            "instruction", "response", "question", "answer",
            "input", "output", "prompt", "completion", "topic",
        }:
            return 0.60
        if analysis.avg_entry_length > 800:
            return 0.55
        return 0.20

    def render_modelfile(
        self,
        base_model: str,
        dataset_content: str,
        instructions: str,
        parameters: Dict[str, Any],
    ) -> str:
        sanitized = dataset_content.replace('"""', r'\"\"\"')
        header = textwrap.dedent(instructions).strip() if instructions else ""
        body = (header + "\n\n" + sanitized.strip()) if header else sanitized.strip()
        lines = [
            f"FROM {base_model}",
            "",
            'SYSTEM """',
            body,
            '"""',
        ]
        for k, v in parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        return "\n".join(lines)

    def rationale(self, analysis: DatasetAnalysis) -> str:
        return (
            f"Dataset is free-form text ({analysis.entry_count} entries, "
            f"avg {analysis.avg_entry_length:.0f} chars/entry). "
            "Injecting as a SYSTEM block provides the model with persistent "
            "background knowledge without restructuring the content."
        )


class FewShotStrategy(TrainingStrategy):
    """
    Formats QA / input-output pairs as alternating MESSAGE turns.

    Best for: question-answer datasets, input-output demonstrations where
    the goal is to teach response style and format via worked examples.
    """

    name = "few_shot"
    description = (
        "Formats QA or input/output pairs as MESSAGE turns; teaches "
        "response style through worked examples."
    )

    _Q_KEYS = {"question", "input", "query", "user"}
    _A_KEYS = {"answer", "output", "response", "assistant"}

    def suitability(self, analysis: DatasetAnalysis) -> float:
        keys = analysis.detected_keys
        if keys & self._Q_KEYS and keys & self._A_KEYS:
            return 0.92
        return 0.15

    def render_modelfile(
        self,
        base_model: str,
        dataset_content: str,
        instructions: str,
        parameters: Dict[str, Any],
    ) -> str:
        lines = [f"FROM {base_model}", ""]
        if instructions:
            lines += ['SYSTEM """', textwrap.dedent(instructions).strip(), '"""', ""]

        q_key = a_key = None
        entries = []
        try:
            for raw in dataset_content.splitlines():
                if raw.strip():
                    entries.append(json.loads(raw))
            if entries:
                sample_keys = set(entries[0].keys())
                q_key = next((k for k in self._Q_KEYS if k in sample_keys), None)
                a_key = next((k for k in self._A_KEYS if k in sample_keys), None)
        except (json.JSONDecodeError, StopIteration):
            pass

        if q_key and a_key:
            for entry in entries:
                q = str(entry.get(q_key, "")).replace('"', '\\"')
                a = str(entry.get(a_key, "")).replace('"', '\\"')
                lines += [f'MESSAGE user "{q}"', f'MESSAGE assistant "{a}"', ""]
        else:
            # Graceful fallback: inject as SYSTEM
            logger.warning("FewShotStrategy: could not resolve Q/A keys; falling back to SYSTEM injection")
            sanitized = dataset_content.replace('"""', r'\"\"\"')
            lines += ['SYSTEM """', sanitized.strip(), '"""']

        for k, v in parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        return "\n".join(lines)

    def rationale(self, analysis: DatasetAnalysis) -> str:
        matched = sorted(analysis.detected_keys & (self._Q_KEYS | self._A_KEYS))
        return (
            f"Detected QA/example structure in {analysis.entry_count} JSONL entries "
            f"(matched keys: {matched}). Formatting as MESSAGE turns teaches the model "
            "to replicate the demonstrated response patterns."
        )


class InstructionTuningStrategy(TrainingStrategy):
    """
    Uses instruction/response pairs to reinforce instruction-following.

    Best for: OpenAI-style fine-tuning datasets with explicit instruction
    and completion fields.  Produces tighter task adherence than FewShot.
    """

    name = "instruction_tuning"
    description = (
        "Formats instruction/response pairs as MESSAGE turns; strengthens "
        "instruction-following on explicit task directives."
    )

    _I_KEYS = {"instruction", "prompt", "task", "system"}
    _R_KEYS = {"response", "completion", "output", "assistant"}

    def suitability(self, analysis: DatasetAnalysis) -> float:
        keys = analysis.detected_keys
        if keys & self._I_KEYS and keys & self._R_KEYS:
            return 0.95
        return 0.10

    def render_modelfile(
        self,
        base_model: str,
        dataset_content: str,
        instructions: str,
        parameters: Dict[str, Any],
    ) -> str:
        lines = [f"FROM {base_model}", ""]
        if instructions:
            lines += ['SYSTEM """', textwrap.dedent(instructions).strip(), '"""', ""]

        i_key = r_key = None
        entries = []
        try:
            for raw in dataset_content.splitlines():
                if raw.strip():
                    entries.append(json.loads(raw))
            if entries:
                sample_keys = set(entries[0].keys())
                i_key = next((k for k in self._I_KEYS if k in sample_keys), None)
                r_key = next((k for k in self._R_KEYS if k in sample_keys), None)
        except (json.JSONDecodeError, StopIteration):
            pass

        if i_key and r_key:
            for entry in entries:
                inst = str(entry.get(i_key, "")).replace('"', '\\"')
                resp = str(entry.get(r_key, "")).replace('"', '\\"')
                lines += [f'MESSAGE user "{inst}"', f'MESSAGE assistant "{resp}"', ""]
        else:
            logger.warning("InstructionTuningStrategy: could not resolve instruction/response keys; falling back to SYSTEM injection")
            sanitized = dataset_content.replace('"""', r'\"\"\"')
            lines += ['SYSTEM """', sanitized.strip(), '"""']

        for k, v in parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        return "\n".join(lines)

    def rationale(self, analysis: DatasetAnalysis) -> str:
        matched = sorted(analysis.detected_keys & (self._I_KEYS | self._R_KEYS))
        return (
            f"Found instruction/response structure across {analysis.entry_count} examples "
            f"(matched keys: {matched}). Instruction-tuning format maximises the model's "
            "ability to follow explicit task directives."
        )


class DomainAdaptationStrategy(TrainingStrategy):
    """
    Assembles structured topic/content entries into a hierarchical SYSTEM prompt.

    Best for: knowledge-base datasets with distinct topics, reference material,
    or structured documents where preserving logical grouping matters.
    """

    name = "domain_adaptation"
    description = (
        "Assembles structured topic/content entries into a rich SYSTEM block; "
        "preserves logical knowledge grouping for domain experts."
    )

    _TOPIC_KEYS = {"topic", "category", "section", "title", "heading"}
    _CONTENT_KEYS = {"content", "text", "body", "description", "detail"}

    def suitability(self, analysis: DatasetAnalysis) -> float:
        keys = analysis.detected_keys
        if keys & self._TOPIC_KEYS and keys & self._CONTENT_KEYS:
            return 0.92
        # Many JSONL entries with no QA/instruction structure → probably domain docs
        if (
            analysis.format == "jsonl"
            and analysis.entry_count > 10
            and not keys & {"instruction", "question", "input", "prompt"}
        ):
            return 0.55
        return 0.10

    def render_modelfile(
        self,
        base_model: str,
        dataset_content: str,
        instructions: str,
        parameters: Dict[str, Any],
    ) -> str:
        blocks = []
        try:
            for raw in dataset_content.splitlines():
                if not raw.strip():
                    continue
                entry = json.loads(raw)
                topic = next((entry[k] for k in self._TOPIC_KEYS if k in entry), None)
                content = next((entry[k] for k in self._CONTENT_KEYS if k in entry), str(entry))
                blocks.append(f"## {topic}\n{content}" if topic else content)
        except (json.JSONDecodeError, StopIteration):
            blocks = [dataset_content]

        assembled = "\n\n".join(blocks).replace('"""', r'\"\"\"')
        header = (
            textwrap.dedent(instructions).strip()
            if instructions
            else "You are a domain expert. Use the structured knowledge below to answer accurately."
        )
        lines = [
            f"FROM {base_model}",
            "",
            'SYSTEM """',
            header,
            "",
            assembled,
            '"""',
        ]
        for k, v in parameters.items():
            lines.append(f"PARAMETER {k} {v}")
        return "\n".join(lines)

    def rationale(self, analysis: DatasetAnalysis) -> str:
        matched = sorted(analysis.detected_keys & (self._TOPIC_KEYS | self._CONTENT_KEYS))
        return (
            f"Dataset contains {analysis.entry_count} structured knowledge entries "
            f"(matched keys: {matched}). Assembling as a hierarchical SYSTEM block "
            "preserves logical grouping and maximises domain knowledge retention."
        )
