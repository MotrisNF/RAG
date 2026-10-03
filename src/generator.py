import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import Chunk


class Generator:
    MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    MAX_CONTEXT_TOKENS: int = 2500
    MAX_NEW_TOKENS: int = 256
    SYSTEM_PROMPT: str = (
        "You answer questions about the vLLM codebase. Answer ONLY from "
        "the context. Quote the exact names and values written in the "
        "context. If the context does not contain the answer, say: \"The "
        "context does not contain the answer.\""
    )
    REMINDER: str = (
        "Answer with the exact values from the context. If the question "
        "is not about vLLM, LLMs, model serving or programming, say: "
        "\"The context does not contain the answer.\""
    )

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float32)
        self._model.eval()

    def _build_context(self, chunks: list[Chunk]) -> str:
        context: str = ""
        total_context: int = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"[CHUNK {i}]\n"
            path = chunk.file_path + "\n"
            text = chunk.text + "\n"
            separator = "---\n"

            block = header + path + text + separator
            ids = self._tokenizer.encode(block, add_special_tokens=False)
            n_tokens = len(ids)

            if total_context + n_tokens > self.MAX_CONTEXT_TOKENS:
                break
            context += block
            total_context += n_tokens
        return context

    def _build_messages(self,
                        question: str,
                        context: str
                        ) -> list[dict[str, str]]:
        """Build the chat messages the model receives.

        The fixed rules go in the system message; the context and the
        question in the user message, with the question LAST: a small
        model attends most to what it reads right before answering. The
        most important rule is repeated next to the question.

        Args:
            question: The user's question.
            context: The retrieved chunks, from ``_build_context``.

        Returns:
            A system and a user message, in the format of
            ``apply_chat_template``.
        """
        user = (f"Context:\n{context}\n"
                f"Question: {question}\n\n"
                f"{self.REMINDER}")
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        context = self._build_context(chunks)
        messages = self._build_messages(question, context)

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self._tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            # transformers types `generate` for a protocol that the Auto
            # classes do not declare: a false positive of its type hints.
            output = self._model.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False)

        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        answers = self._tokenizer.batch_decode(new_tokens,
                                               skip_special_tokens=True)
        return answers[0].strip()
