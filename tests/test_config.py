import os

from omibridge.config import load_env_file, parse_env_file


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


def test_parse_env_file_does_not_touch_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OMIBRIDGE_TOKEN", "from-the-shell")
    p = tmp_path / ".env"
    p.write_text('OMIBRIDGE_TOKEN="secret-token"  # webhook secret\nOTHER=x # note\n')
    assert parse_env_file(p) == {"OMIBRIDGE_TOKEN": "secret-token", "OTHER": "x"}
    assert os.environ["OMIBRIDGE_TOKEN"] == "from-the-shell"
    assert parse_env_file(tmp_path / "missing") == {}
