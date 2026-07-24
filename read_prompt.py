
from pathlib import Path


def load_content():
    prompt_path = Path(__file__).resolve().parent / "SDWPF_ad.txt"
    with prompt_path.open(encoding="utf-8") as f:
        content = f.read()
    return content
