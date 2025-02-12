import re

def word_count(text: str) -> int:
    text_cleaned = re.sub(r"[^\w\s]", "", text)
    words = text_cleaned.split()
    return len(words)
