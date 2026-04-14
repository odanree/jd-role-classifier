"""
LoRA fine-tuning demo for O*NET SOC role classification.

Uses BERT base + LoRA adapters (via PEFT) to train a multi-class classifier
over O*NET SOC codes using labeled job description data.

Usage:
    python scripts/fine_tune.py \
        --data-path data/labeled_jds.jsonl \
        --output-dir models/bert-lora-onet \
        --epochs 3 \
        --batch-size 16

Expected JSONL format (one JSON object per line):
    {"text": "Senior ML Engineer with PyTorch...", "soc_code": "15-2053.00"}
"""
import argparse
import json
from pathlib import Path


def load_labeled_data(data_path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            texts.append(obj["text"])
            labels.append(obj["soc_code"])
    return texts, labels


def build_label_map(labels: list[str]) -> dict[str, int]:
    unique = sorted(set(labels))
    return {code: i for i, code in enumerate(unique)}


def fine_tune(
    data_path: str,
    output_dir: str,
    base_model: str = "google-bert/bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.utils.data import Dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft torch")
        return

    texts, labels = load_labeled_data(data_path)
    label_map = build_label_map(labels)
    int_labels = [label_map[l] for l in labels]
    num_labels = len(label_map)

    print(f"Loaded {len(texts)} samples across {num_labels} SOC codes")
    print(f"Base model: {base_model}")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class JDDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    # Split 80/20
    split = int(len(texts) * 0.8)
    train_dataset = JDDataset(texts[:split], int_labels[:split])
    eval_dataset = JDDataset(texts[split:], int_labels[split:])

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label map for inference
    label_map_path = Path(output_dir) / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"Label map saved to {label_map_path}")
    print(f"Set MODEL_PATH={output_dir} and LOAD_CLASSIFIER=true in your .env to use this model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT+LoRA for O*NET SOC classification")
    parser.add_argument("--data-path", required=True, help="Path to labeled JSONL file")
    parser.add_argument("--output-dir", default="models/bert-lora-onet", help="Output directory for model")
    parser.add_argument("--base-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()
    fine_tune(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
