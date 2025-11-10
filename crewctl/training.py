"""Training helpers for converting base models into LRMs (LoRA adapters)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import CrewCtlConfig
from .model import ModelManager
from .utils import CrewCtlError, console, ensure_directory


@dataclass
class TrainingJob:
    base_model: str
    dataset: str
    output_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    register_with_ollama: bool = False


def _ensure_training_dependencies() -> None:
    try:
        import torch  # noqa
        import transformers  # noqa
        import datasets  # noqa
        import peft  # noqa
        import accelerate  # noqa
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise CrewCtlError(
            "Training dependencies are missing. Re-run the deploy script with --with-training-extras."
        ) from exc


def _load_dataset(dataset_spec: str):
    from datasets import load_dataset

    path = Path(dataset_spec)
    if path.exists():
        if path.is_dir():
            return load_dataset("json", data_files=str(path / "*.json"))
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(path))
        if suffix in {".txt"}:
            return load_dataset("text", data_files=str(path))
        raise CrewCtlError(f"Unsupported dataset file type: {suffix}")

    # treat as Hugging Face dataset identifier
    return load_dataset(dataset_spec)


def _prepare_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def _prepare_model(base_model: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    device_map = "auto"
    load_in_4bit = torch.cuda.is_available()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def _tokenize_dataset(dataset, tokenizer):
    def tokenize_function(example):
        text = example.get("text") or " ".join(str(value) for value in example.values())
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=min(tokenizer.model_max_length, 2048),
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist(),
        }

    tokenized = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    return tokenized


def train_lrm(job: TrainingJob, config: Optional[CrewCtlConfig] = None) -> Path:
    """Train a LoRA adapter from the given dataset and register it with Ollama."""
    _ensure_training_dependencies()

    config = config or CrewCtlConfig()
    output_dir = config.training_output_dir / job.output_name
    ensure_directory(output_dir)

    console.print(
        f"[bold blue]Starting LRM training[/] for base model [bold]{job.base_model}[/] "
        f"using dataset [bold]{job.dataset}[/]"
    )

    tokenizer = _prepare_tokenizer(job.base_model)
    raw_dataset = _load_dataset(job.dataset)
    train_split = raw_dataset["train"] if "train" in raw_dataset else raw_dataset
    tokenized_dataset = _tokenize_dataset(train_split, tokenizer)

    model = _prepare_model(job.base_model, job.lora_r, job.lora_alpha, job.lora_dropout)

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf"),
        per_device_train_batch_size=job.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=job.epochs,
        learning_rate=job.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    metadata = {
        "base_model": job.base_model,
        "dataset": job.dataset,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "learning_rate": job.learning_rate,
        "lora_r": job.lora_r,
        "lora_alpha": job.lora_alpha,
        "lora_dropout": job.lora_dropout,
    }
    (output_dir / "training-metadata.json").write_text(json.dumps(metadata, indent=2))

    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        (
            f"FROM {job.base_model}\n"
            f"ADAPTER adapter\n"
            f"PARAMETER temperature 0.7\n"
            f"PARAMETER num_ctx 4096\n"
        )
    )

    console.print(f"[bold green]Training complete[/]. Artifacts stored in {output_dir}")

    if job.register_with_ollama:
        model_name = f"{job.output_name}:{metadata['epochs']}ep"
        manager = ModelManager(config)
        console.print(f"[bold blue]Registering adapter as Ollama model {model_name}")
        manager._ollama("create", model_name, "-f", str(modelfile_path))
        manager.use_model(model_name)

    return output_dir
