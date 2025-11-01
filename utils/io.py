# -*- coding: utf-8 -*-
import os


def resolve_csv_path(path: str) -> str:
    """Return a CSV path, trying the provided path first, then data/<basename>.

    Raises FileNotFoundError if neither exists.
    """
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        return p
    base = os.path.basename(p)
    candidate = os.path.join("data", base)
    if os.path.isfile(candidate):
        print(f"CSV not found at '{path}', using '{candidate}' from data/")
        return candidate
    # Provide a small hint of available CSV files in data/
    data_dir = "data"
    hint = ""
    if os.path.isdir(data_dir):
        try:
            files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
            if files:
                hint = f" Available in data/: {', '.join(files)}"
        except Exception:
            pass
    raise FileNotFoundError(f"CSV not found: '{path}' or '{candidate}'.{hint}")


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)

