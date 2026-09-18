#!/usr/bin/env python3
"""One place where credentials and settings enter the code.

Precedence (first wins):  variables already exported in the shell  >  genomics/.env  >  ~/.progenome.env  >  defaults.
Copy .env.example to .env and fill it in; .env is git-ignored. Every script that needs a credential or an endpoint
imports `settings` from here instead of reading os.environ itself.

    python config.py          # prints the effective settings with secrets masked, and where each came from
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

GENOMICS = Path(__file__).resolve().parent
ENV_FILES = (GENOMICS / ".env", Path.home() / ".progenome.env")
SECRET_KEYS = {"NVIDIA_API_KEY", "NEO4J_PASSWORD", "BREV_API_KEY"}


def _parse_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; 'export KEY=VALUE' accepted; blank lines and full-line comments ignored; quotes stripped."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1]:
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env_files() -> dict[str, str]:
    """Put file values into os.environ for keys that are not already set; return {key: source} for reporting."""
    source: dict[str, str] = {}
    for path in ENV_FILES:
        if not path.exists():
            continue
        for key, value in _parse_env_file(path).items():
            if key not in os.environ:
                os.environ[key] = value
                source[key] = str(path)
    return source


_SOURCES = load_env_files()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # LLM decoder (NVIDIA NIM or any OpenAI-compatible chat endpoint)
    nvidia_api_key: str = field(default_factory=lambda: _env("NVIDIA_API_KEY", ""))
    nim_model: str = field(default_factory=lambda: _env("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b"))
    nim_url: str = field(default_factory=lambda: _env("NIM_URL", "https://integrate.api.nvidia.com/v1/chat/completions"))
    # Neo4j browser
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "progenome"))
    # data source and defaults
    haploblocks_base: str = field(default_factory=lambda: _env("HAPLOBLOCKS_BASE", "https://data.haploblocks.org"))
    chrom: str = field(default_factory=lambda: _env("CHROM", "chr22"))
    # NVIDIA Brev (the CLI keeps its own login; these only pick the instance)
    brev_instance: str = field(default_factory=lambda: _env("BREV_INSTANCE", "progenome-gpu"))
    brev_type: str = field(default_factory=lambda: _env("BREV_TYPE", "g2-standard-4:nvidia-l4:1"))
    # paths (relative to this folder unless overridden)
    data_dir: Path = field(default_factory=lambda: Path(_env("PROGENOME_DATA_DIR", str(GENOMICS / "data"))))
    outputs_dir: Path = field(default_factory=lambda: Path(_env("PROGENOME_OUTPUTS_DIR", str(GENOMICS / "outputs"))))


settings = Settings()


def require(env_key: str) -> str:
    """Return a setting that must be present, or exit with instructions instead of a traceback."""
    value = os.environ.get(env_key, "")
    if not value or value.endswith("REPLACE_ME"):
        raise SystemExit(f"{env_key} is not set. Copy genomics/.env.example to genomics/.env (or ~/.progenome.env) and fill it in, "
                         f"or export {env_key}=... ; see README step 4.")
    return value


def describe() -> str:
    lines = []
    for f_ in fields(Settings):
        key = f_.name.upper() if not f_.name.endswith("_dir") else f"PROGENOME_{f_.name.upper()}"
        value = getattr(settings, f_.name)
        shown = ("***" + str(value)[-4:] if value else "(not set)") if key in SECRET_KEYS else str(value)
        origin = _SOURCES.get(key, "exported" if key in os.environ else "default")
        lines.append(f"{key:24s} {shown:60s} [{origin}]")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"env files read: {[str(p) for p in ENV_FILES if p.exists()] or 'none'}")
    print(describe())
    sys.exit(0)
