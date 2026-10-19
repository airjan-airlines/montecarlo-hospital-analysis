import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "render_readme", Path(__file__).resolve().parents[1] / "scripts" / "render_readme.py")
render_readme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_readme)


def test_render_placeholders(tmp_path):
    (tmp_path / "t.md").write_text("| a |\n|---|\n| 1 |\n")
    metrics = {"m": 0.0832, "ci": [0.07, 0.09], "vrf": {"y1/cv": 3.38}, "n": 2069}
    out = render_readme.render(
        "{{m*100:.1f}}% [{{ci.0:.2f}}, {{ci.1:.2f}}] {{vrf.y1/cv:.1f}}x n={{n}}\n{{table:t}}",
        metrics, tmp_path)
    assert out == "8.3% [0.07, 0.09] 3.4x n=2069\n| a |\n|---|\n| 1 |"


def test_missing_key_fails_loudly(tmp_path):
    with pytest.raises(KeyError):
        render_readme.render("{{nope}}", {}, tmp_path)
