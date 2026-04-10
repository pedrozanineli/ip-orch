import json
import os
from typing import Dict, List, Tuple, Optional


CONFIG_DIR = os.path.expanduser("~/.ip-orch")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


DEFAULT_CONFIG = {
    "base_env": "base",
    "models_path": "",
    "envs_base_dir": "",
    # list of [env_name, model_name]
    "full_models": [],
    # alias -> status {"ok"|"broken"}
    "model_status": {},
}


def ensure_dir() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)


def load_config() -> Dict:
    """Load user config from CONFIG_PATH; if missing, return defaults.

    Ensures required keys exist and have the right types.
    """
    ensure_dir()
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Fallback to defaults if file is corrupted
        return DEFAULT_CONFIG.copy()

    # Normalize types and keys
    cfg = DEFAULT_CONFIG.copy()
    if isinstance(data, dict):
        cfg.update({k: v for k, v in data.items() if k in cfg})

    # Guarantee structure
    if not isinstance(cfg.get("full_models"), list):
        cfg["full_models"] = []
    if not isinstance(cfg.get("model_status"), dict):
        cfg["model_status"] = {}
    return cfg


def save_config(cfg: Dict) -> None:
    ensure_dir()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def add_model(env_name: str, model_name: str) -> None:
    """Append a new (env, model) pair to config, avoiding duplicates."""
    cfg = load_config()
    pair = [env_name, model_name]
    # Avoid duplicates
    if pair not in cfg["full_models"]:
        cfg["full_models"].append(pair)
        save_config(cfg)


def remove_model(env_name: str, model_name: Optional[str] = None) -> bool:
    """Remove pairs by env or a specific (env, model).

    Returns True if config changed.
    """
    cfg = load_config()
    before = list(cfg["full_models"])  # copy
    if model_name is None:
        cfg["full_models"] = [p for p in cfg["full_models"] if p[0] != env_name]
    else:
        cfg["full_models"] = [p for p in cfg["full_models"] if not (p[0] == env_name and p[1] == model_name)]
    changed = cfg["full_models"] != before
    if changed:
        save_config(cfg)
    return changed


def set_base_env(name: str) -> None:
    """Set the base orchestrator environment name."""
    cfg = load_config()
    cfg["base_env"] = name
    save_config(cfg)


def set_models_path(path: str) -> None:
    """Set the base path for local model weights."""
    cfg = load_config()
    cfg["models_path"] = path
    save_config(cfg)


def get_model_status_map() -> Dict[str, str]:
    """Return a copy of the alias->status map.

    Status values are strings: "ok" or "broken". Missing means unknown.
    """
    cfg = load_config()
    ms = cfg.get("model_status") or {}
    return dict(ms)


def set_model_status(alias: str, status: str) -> None:
    """Persist last-known status for a model alias.

    Accepted statuses: "ok", "broken". Silently ignores other values.
    """
    if status not in {"ok", "broken"}:
        return
    cfg = load_config()
    ms = cfg.get("model_status") or {}
    if ms.get(alias) == status:
        return
    ms[alias] = status
    cfg["model_status"] = ms
    save_config(cfg)
