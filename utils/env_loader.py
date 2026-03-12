from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_dotenv() -> None:
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)
