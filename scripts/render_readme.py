"""README.template.md + results/metrics.json -> README.md (no hand-typed numbers).

Placeholders:
  {{dotted.path}}            value from metrics.json (list indices allowed: a.0)
  {{dotted.path:.3f}}        with a Python format spec
  {{dotted.path*100:.1f}}    scaled by 100 first (percentages)
  {{table:name}}             results/tables/<name>.md inlined
"""

import argparse
import json
import re
from pathlib import Path

from mtraj.config import ROOT

PLACEHOLDER = re.compile(r"\{\{\s*([^}:*\s]+)(\*100)?(?::([^}]+))?\s*\}\}")


def lookup(metrics: dict, path: str):
    node = metrics
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def render(template: str, metrics: dict, tables: Path) -> str:
    def sub(m: re.Match) -> str:
        key, scale, fmt = m.groups()
        if key.startswith("table") and fmt:
            return (tables / f"{fmt.strip()}.md").read_text().rstrip()
        value = lookup(metrics, key)
        if scale:
            value = value * 100
        return format(value, fmt) if fmt else str(value)

    return PLACEHOLDER.sub(sub, template)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    metrics = json.loads((results / "metrics.json").read_text())
    text = render((ROOT / "README.template.md").read_text(), metrics, results / "tables")
    (ROOT / "README.md").write_text(text)
    print(f"wrote {ROOT / 'README.md'}")


if __name__ == "__main__":
    main()
