# api_service/app/preprocess.py
import re

def clean_text(text: str) -> str:
    text = text.strip()
    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text
