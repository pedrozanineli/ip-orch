import os
import subprocess
from typing import Set

from .helpers import DEFAULT_MODELS_BY_ENV


def _discover_conda_envs() -> Set[str]:
    try:
        out = subprocess.check_output(["conda", "env", "list"], text=True)
        envs = []
        for line in out.splitlines():
            if line.strip().startswith("#"):
                continue
            parts = [p for p in line.split() if p]
            if parts:
                envs.append(parts[0])
        return set(e for e in envs if e and e != "#")
    except Exception:
        return set()


def _guess_envs_dir() -> str:
    candidates = [
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs"),
        os.path.join(os.path.expanduser("~"), "anaconda3", "envs"),
        os.path.join(os.environ.get("CONDA_PREFIX", os.path.expanduser("~")), "envs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _discover_envs_from_dir(base_dir: str) -> set:
    envs = set()
    if not base_dir:
        return envs
    path = os.path.expanduser(base_dir)
    if not os.path.isdir(path):
        return envs
    def has_python(d: str) -> bool:
        return (
            os.path.isfile(os.path.join(d, "bin", "python")) or
            os.path.isfile(os.path.join(d, "Scripts", "python.exe"))
        )
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full) and has_python(full):
                envs.add(name)
    except Exception:
        pass
    return envs


def _current_conda_env() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or os.path.basename(os.environ.get("CONDA_PREFIX", ""))


def _normalize_token(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _match_known_token(env_name: str) -> str:
    nenv = _normalize_token(env_name)
    for token in DEFAULT_MODELS_BY_ENV.keys():
        if _normalize_token(token) in nenv:
            return token
    return ""


def _python_for_env(env_name: str, base_dir: str) -> str:
    if not base_dir:
        return ""
    base = os.path.expanduser(base_dir)
    roots = [os.path.join(base, env_name), os.path.join(base, f".{env_name}")]
    candidates = []
    for root in roots:
        candidates.extend([
            os.path.join(root, "bin", "python"),
            os.path.join(root, "Scripts", "python.exe"),
        ])
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return ""

