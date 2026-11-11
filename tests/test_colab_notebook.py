import json
from pathlib import Path


def test_colab_t4_notebook_is_valid_json_and_python():
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "t4_validation_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    code_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert code_cells

    for index, cell in enumerate(code_cells):
        source = cell.get("source")
        assert isinstance(source, list)
        compile("".join(source), f"{notebook_path.name}:cell-{index}", "exec")

    all_source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "ab9886635c7db234425aa7837bdb3f21434bb0b9" in all_source
    assert "compute_capability != (7, 5)" in all_source
    assert "scripts/profile_ncu.sh" in all_source
