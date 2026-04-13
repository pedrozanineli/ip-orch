from typing import List, Dict

def _get_aliases() -> Dict[str, str]:
    try:
        from ..core.model_factory import ModelFactory as _MF
        return getattr(_MF, "_ALIASES", {})
    except Exception:
        return {}


def _norm_token(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch in "-_")


def _canonical_alias(name: str) -> str:
    aliases = _get_aliases()
    norm = _norm_token(name)
    for key in aliases.keys():
        if _norm_token(key) == norm:
            return key
    return norm


def _group_pairs(pairs: List[List[str]]):
    grouped = {}
    for env, model in pairs:
        alias = _canonical_alias(model)
        grouped.setdefault(env, set()).add(alias)
    return {e: sorted(list(ms)) for e, ms in grouped.items()}


def _dedup_pairs(pairs: List[List[str]]):
    seen = set()
    out = []
    for env, model in pairs:
        clean_env = _clean_env(env)
        alias = _canonical_alias(model)
        key = (clean_env, alias)
        if key in seen:
            continue
        seen.add(key)
        out.append([clean_env, alias])
    return out


def _clean_env(name: str) -> str:
    return (name or "").strip().lstrip('.')


# Known calculator env names and default model suggestions
DEFAULT_MODELS_BY_ENV = {
    "fair-chem": "FAIRChem",
    "alphanet": "AlphaNet",
    "chgnet": "CHGNet",
    "deepmd": "DPA-3.1-3M-FT",
    "dptb": "DPTB",
    "eqnorm": "Eqnorm-MPtrj",
    "grace": "GRACE-2L-OAM",
    "mattersim": "MatterSim-v1",
    "hamgnn": "HAMGNN",
    "hienet": "HIENet",
    "m3gnet": "M3GNet",
    "mace": "MACE-MP-0",
    "matris": "MatRIS-10M-OAM",
    "nequip": "Nequip-OAM-L",
    "nequix": "Nequix-MP",
    "orb": "ORB-V3",
    "sevenn": "SevenNet-Omni",
    "tace": "TACE-v1-OAM-M",
    "upet": "PET-OAM-XL",
}

# Variants per package token
PACKAGE_VARIANTS = {
    "mace": ["mace-mp", "mace-mp-0", "mace-mpa-0"],
    "orb": ["orb-v3", "orb-v2", "orb-v2-mptrj"],
    "grace": ["grace-2l-oam", "grace-1l-oam", "grace-2l-mp-r6"],
    "nequip": ["nequip-oam-l", "nequip-mp-l", "allegro-mp-l", "allegro-oam-l"],
    "nequix": ["nequix-mp", "nequix-mp-pft"],
    "matris": ["matris-10m-oam", "matris-10m-mp"],
    "deepmd": ["dpa-3.1-3m-ft", "dpa-3.1-mptrj"],
    "sevenn": ["sevennet-omni", "sevennet-l3i5"],
}

