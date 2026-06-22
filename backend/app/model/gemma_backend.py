from __future__ import annotations

import time
from functools import lru_cache

from app.core.config import get_settings
from app.model.base import RankingBackend, RankingResult, Verdict

_LABEL_ORDER = (Verdict.MODEL_A, Verdict.MODEL_B, Verdict.TIE)


class GemmaLoRABackend(RankingBackend):
    name = "gemma-2-9b-lora-int4"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
        self._device = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
            import bitsandbytes  # noqa: F401
        except ImportError:
            return False

        import torch

        if not torch.cuda.is_available():
            return False
        if not self.settings.lora_adapter_path:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import (
            BitsAndBytesConfig,
            Gemma2ForSequenceClassification,
            GemmaTokenizerFast,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = GemmaTokenizerFast.from_pretrained(self.settings.base_checkpoint)
        tokenizer.add_eos_token = True
        tokenizer.padding_side = "right"

        base_model = Gemma2ForSequenceClassification.from_pretrained(
            self.settings.base_checkpoint,
            num_labels=self.settings.num_labels,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, self.settings.lora_adapter_path)
        model.eval()
        model.config.use_cache = False

        self._tokenizer = tokenizer
        self._model = model
        self._device = next(model.parameters()).device

    def _encode(self, prompt: str, response_a: str, response_b: str):
        import torch

        text = (
            f"<prompt>: {prompt}\n\n<response_a>: {response_a}\n\n<response_b>: {response_b}"
        )
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.settings.max_sequence_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoded.items()}

    def rank(self, prompt: str, response_a: str, response_b: str) -> RankingResult:
        import torch

        self._load()
        start = time.perf_counter()

        inputs = self._encode(prompt, response_a, response_b)
        with torch.no_grad(), torch.autocast(device_type=self._device.type, dtype=torch.float16):
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).squeeze(0).tolist()
        p_a, p_b, p_tie = probs

        verdict = _LABEL_ORDER[max(range(3), key=lambda i: probs[i])]
        latency_ms = (time.perf_counter() - start) * 1000

        return RankingResult(
            verdict=verdict,
            prob_a=round(p_a, 4),
            prob_b=round(p_b, 4),
            prob_tie=round(p_tie, 4),
            backend=self.name,
            latency_ms=round(latency_ms, 3),
        )


@lru_cache
def get_gemma_backend() -> GemmaLoRABackend:
    return GemmaLoRABackend()
