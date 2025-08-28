from pathlib import Path
import yaml, json, re

def read_settings():
    cfg = Path(__file__).parent / "config.yaml"
    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_windows_raw(path_str: str) -> str:
    # Helpful for Windows paths that include backslashes and spaces
    # e.g. r"C:\Users\sagni\Downloads\Price Sense\archive\flipkart_com-ecommerce_sample.csv"
    return path_str.replace("\\", "\\\\")

def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sanitize_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()
