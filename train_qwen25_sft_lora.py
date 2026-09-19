import argparse
import os
from dataclasses import dataclass

# AutoDL often cannot reach huggingface.co directly. Set this before importing
# datasets/transformers so huggingface_hub reads the mirror endpoint early.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_name: str
    output_dir: str
    max_samples: int
    eval_samples: int
    max_length: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    use_4bit: bool
    bf16: bool
    packing: bool
    assistant_only_loss: bool


def log(message: str):
    print(f"[SFT] {message}", flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="First SFT run: Qwen2.5-7B-Instruct + no_robots + LoRA/QLoRA"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        help="HF model id or local model path, for example /root/autodl-tmp/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--dataset_name", default="HuggingFaceH4/no_robots")
    parser.add_argument("--output_dir", default="./outputs/qwen25-7b-norobots-lora")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16.")
    parser.add_argument("--packing", action="store_true", help="Pack short samples into longer sequences.")
    parser.add_argument(
        "--full_loss",
        action="store_true",
        help="Train loss on the full conversation instead of assistant messages only.",
    )
    args = parser.parse_args()

    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        use_4bit=not args.no_4bit,
        bf16=not args.fp16,
        packing=args.packing,
        assistant_only_loss=not args.full_loss,
    )


def load_messages_dataset(cfg: TrainConfig):
    log(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    log(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split="train")
    log(f"Raw dataset size: {len(dataset)}")

    if "messages" not in dataset.column_names:
        raise ValueError(
            f"{cfg.dataset_name} does not contain a 'messages' column. "
            "Convert it to TRL conversational format first."
        )

    keep_n = min(cfg.max_samples + cfg.eval_samples, len(dataset))
    log(f"Shuffling and selecting {keep_n} samples")
    dataset = dataset.shuffle(seed=cfg.seed).select(range(keep_n))

    remove_columns = [name for name in dataset.column_names if name != "messages"]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    if cfg.eval_samples > 0 and len(dataset) > cfg.eval_samples:
        split = dataset.train_test_split(test_size=cfg.eval_samples, seed=cfg.seed)
        log(f"Train samples: {len(split['train'])}; eval samples: {len(split['test'])}")
        return split["train"], split["test"]

    log(f"Train samples: {len(dataset)}; eval disabled")
    return dataset, None


def load_model_and_tokenizer(cfg: TrainConfig):
    log(f"Loading tokenizer from: {cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.use_4bit:
        log("Using 4-bit QLoRA quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    log(f"Loading model from: {cfg.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    log("Model loaded")

    return model, tokenizer


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    log(f"Output dir: {cfg.output_dir}")

    train_dataset, eval_dataset = load_messages_dataset(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)

    log("Building LoRA config")
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        max_length=cfg.max_length,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50,
        save_total_limit=2,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if cfg.use_4bit else "adamw_torch",
        report_to="none",
        packing=cfg.packing,
        assistant_only_loss=cfg.assistant_only_loss,
        eos_token="<|im_end|>",
        seed=cfg.seed,
    )

    log("Building SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log("Starting training")
    trainer.train()
    log("Saving adapter and tokenizer")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Done. LoRA adapter saved to: {cfg.output_dir}")
    print("Quick test after training:")
    print(
        "python infer_lora.py "
        f"--base_model {cfg.model_name_or_path} "
        f"--adapter {cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
