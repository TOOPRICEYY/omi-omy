import os

from omibridge.config import load_env_file


def test_env_file_inline_comments(tmp_path, monkeypatch):
    for k in ("A", "B", "C", "D"):
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / ".env"
    p.write_text('A=         # only a comment\nB=value  # comment\nC="quoted # kept"\nD=abc#def\n')
    load_env_file(p)
    assert os.environ["A"] == ""
    assert os.environ["B"] == "value"
    assert os.environ["C"] == "quoted # kept"
    assert os.environ["D"] == "abc#def"
