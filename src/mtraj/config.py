from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("root", "processed"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = p if p.is_absolute() else ROOT / p
    return cfg
