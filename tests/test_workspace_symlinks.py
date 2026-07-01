"""Verify Workspace._copy_files preserves symlinks."""
from pathlib import Path

from heuresis.workspace import Workspace


def test_copy_files_preserves_symlinks(tmp_path: Path):
    """copytree must preserve relative symlinks (needed by discogen)."""
    source = tmp_path / "source"
    source.mkdir()
    discovered = source / "discovered"
    discovered.mkdir()
    (discovered / "loss.py").write_text("# loss")

    dataset = source / "MinAtar" / "Breakout"
    dataset.mkdir(parents=True)
    link = dataset / "loss.py"
    link.symlink_to("../../discovered/loss.py")

    ws = Workspace(files={"MinAtar": source / "MinAtar", "discovered": discovered})
    dest = tmp_path / "workspace"
    dest.mkdir()
    ws._copy_files(dest)

    copied_link = dest / "MinAtar" / "Breakout" / "loss.py"
    assert copied_link.is_symlink(), "symlink was dereferenced instead of preserved"
    assert str(copied_link.readlink()) == "../../discovered/loss.py"
