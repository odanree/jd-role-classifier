"""
BERT + LoRA/PEFT O*NET SOC role classifier.

In mock mode (MOCK_NLP=true or LOAD_CLASSIFIER=false): falls back to
keyword-overlap scoring via vocabulary.py.

In real mode: loads a fine-tuned BERT model with LoRA adapters from
MODEL_PATH for multi-label SOC code prediction.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.vocabulary import load_soc_codes, keyword_lookup


def _mock_classify(text: str, top_k: int = 5) -> list[dict]:
    """Keyword-overlap fallback for test/dev without GPU."""
    results = keyword_lookup(text, top_k=top_k)
    if not results:
        # Return first SOC code as default with low confidence
        codes = load_soc_codes()
        results = [{"soc_code": c["soc_code"], "title": c["title"], "confidence": 0.1} for c in codes[:top_k]]
    # Normalize scores to sum ~1 and assign ranks
    total = sum(r["confidence"] for r in results) or 1.0
    ranked = []
    for i, r in enumerate(results, 1):
        ranked.append({
            "rank": i,
            "soc_code": r["soc_code"],
            "soc_title": r["title"],
            "confidence": round(r["confidence"] / total, 4),
        })
    return ranked


@lru_cache(maxsize=1)
def _load_model():
    """Load fine-tuned BERT + LoRA model and tokenizer."""
    from transformers import AutoTokenizer
    from peft import PeftModel, PeftConfig
    import torch

    settings = get_settings()
    config = PeftConfig.from_pretrained(settings.model_path)
    from transformers import AutoModelForSequenceClassification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=len(load_soc_codes()),
    )
    model = PeftModel.from_pretrained(base_model, settings.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    return model, tokenizer


def classify_jd(text: str, top_k: int = 5) -> list[dict]:
    """
    Classify a job description and return top_k O*NET SOC predictions.

    Returns list of {rank, soc_code, soc_title, confidence}.
    """
    settings = get_settings()

    if not settings.load_classifier or settings.mock_nlp:
        return _mock_classify(text, top_k)

    import torch
    import torch.nn.functional as F

    model, tokenizer = _load_model()
    codes = load_soc_codes()

    inputs = tokenizer(
        text[:2048],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]

    top_indices = probs.topk(min(top_k, len(codes))).indices.tolist()
    top_probs = probs.topk(min(top_k, len(codes))).values.tolist()

    return [
        {
            "rank": rank + 1,
            "soc_code": codes[idx]["soc_code"],
            "soc_title": codes[idx]["title"],
            "confidence": round(prob, 4),
        }
        for rank, (idx, prob) in enumerate(zip(top_indices, top_probs))
    ]
