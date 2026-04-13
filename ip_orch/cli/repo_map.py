MODEL_FAMILY_REPOS = {
    "chgnet": "https://github.com/CederGroupHub/chgnet",
    "deepmd": "https://github.com/deepmodeling/deepmd-kit",
    "eqnorm": "https://github.com/yzchen08/eqnorm",
    "grace": "https://github.com/ICAMS/grace-tensorpotential",
    "hienet": "https://github.com/divelab/AIRS/tree/main/OpenMat/HIENet",
    "m3gnet": "https://github.com/materialyzeai/m3gnet",
    "mace": "https://github.com/ACEsuit/mace",
    "matris": "https://github.com/HPC-AI-Team/MatRIS",
    "mattersim": "https://github.com/microsoft/mattersim",
    "nequip": "https://github.com/mir-group/nequip",
    "nequix": "https://github.com/atomicarchitects/nequix",
    "orb": "https://github.com/orbital-materials/orb-models",
    "sevenn": "https://github.com/MDIL-SNU/SevenNet",
    "tace": "https://github.com/xvzemin/tace",
    "upet": "https://github.com/lab-cosmo/upet",
}


def repo_url_for_alias(alias: str) -> str:
    a = (alias or "").lower()
    if "chgnet" in a:
        return MODEL_FAMILY_REPOS["chgnet"]
    if a.startswith("dpa") or "deepmd" in a:
        return MODEL_FAMILY_REPOS["deepmd"]
    if "eqnorm" in a:
        return MODEL_FAMILY_REPOS["eqnorm"]
    if "grace" in a:
        return MODEL_FAMILY_REPOS["grace"]
    if "hienet" in a:
        return MODEL_FAMILY_REPOS["hienet"]
    if "m3gnet" in a:
        return MODEL_FAMILY_REPOS["m3gnet"]
    if a.startswith("mace"):
        return MODEL_FAMILY_REPOS["mace"]
    if "matris" in a:
        return MODEL_FAMILY_REPOS["matris"]
    if "mattersim" in a:
        return MODEL_FAMILY_REPOS["mattersim"]
    if "nequip" in a or "allegro" in a:
        return MODEL_FAMILY_REPOS["nequip"]
    if "nequix" in a:
        return MODEL_FAMILY_REPOS["nequix"]
    if a.startswith("orb"):
        return MODEL_FAMILY_REPOS["orb"]
    if "sevennet" in a or "sevenn" in a:
        return MODEL_FAMILY_REPOS["sevenn"]
    if "tace" in a:
        return MODEL_FAMILY_REPOS["tace"]
    if "upet" in a or a.startswith("pet"):
        return MODEL_FAMILY_REPOS["upet"]
    return "-"

