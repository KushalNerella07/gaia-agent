# gaia_format.py

import re

def extract_final(text: str) -> str:
    """
    Return the content after 'FINAL ANSWER:' if present; otherwise return the
    first non-empty line. Case-insensitive; supports multiline.
    """
    if not isinstance(text, str):
        return ""
    m = re.search(r"final\s*answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: first non-empty line
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()

def canonical(text: str) -> str:
    """
    Collapse whitespace/newlines to single spaces.
    """
    if not isinstance(text, str):
        return ""
    # replace newlines/tabs with spaces, collapse runs of spaces
    t = re.sub(r"[\r\n\t]+", " ", text)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()
