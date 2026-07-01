import os
from pathlib import Path

from heuresis.env import load_environment


def test_load_environment_reads_dotenv_without_overriding_existing_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "GEMINI_API_KEYS=alpha,beta\n"
        "OPENAI_API_KEY=from-dotenv\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)

    loaded = load_environment(dotenv, force=True)

    assert loaded == dotenv
    assert os.environ["OPENAI_API_KEY"] == "from-shell"
    assert os.environ["GEMINI_API_KEYS"] == "alpha,beta"
    assert os.environ["GEMINI_API_KEY"] == "alpha"
    assert os.environ["GOOGLE_GENERATIVE_AI_API_KEY"] == "alpha"
