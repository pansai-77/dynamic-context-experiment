from __future__ import annotations
import time
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from .models import GenerationResult
from .prompts import SYSTEM_PROMPT

class QwenMLX:
    def __init__(self, model_name: str, max_new_tokens: int = 200, temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # temp=0 uses greedy decoding (argmax); required for deterministic Baseline runs.
        self.sampler = make_sampler(temp=temperature)
        self.model, self.tokenizer = load(model_name)

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def warm_up(self) -> None:
        generate(
            self.model,
            self.tokenizer,
            prompt=self._chat_prompt("请回复：好的。"),
            max_tokens=8,
            sampler=self.sampler,
            verbose=False,
        )

    def answer(self, user_prompt: str) -> GenerationResult:
        prompt = self._chat_prompt(user_prompt)
        input_tokens = self.count_tokens(prompt)
        started = time.perf_counter()
        answer = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=self.sampler,
            verbose=False,
        )
        llm_time_ms = (time.perf_counter() - started) * 1000
        output_tokens = self.count_tokens(answer)
        total_tokens = input_tokens + output_tokens
        tokens_per_second = output_tokens / max(llm_time_ms / 1000, 1e-9)
        return GenerationResult(
            answer=answer.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            llm_time_ms=llm_time_ms,
            tokens_per_second=tokens_per_second,
        )
