import re


class Tokenizer:
    PATTERN = re.compile(r"[a-z0-9]+")

    def tokenize(self, text: str) -> list[str]:
        return self.PATTERN.findall(text.lower())
