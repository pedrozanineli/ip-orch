import argparse
import os
import sys
import tempfile
import subprocess
from typing import List

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich import box

from ..config.config_store import (
    load_config,
    save_config,
    add_model,
    remove_model,
    set_base_env,
    set_models_path,
    get_model_status_map,
    set_model_status,
)

def _get_aliases():
    try:
        from ..core.model_factory import ModelFactory as _MF
        return getattr(_MF, "_ALIASES", {})
    except Exception:
        return {}


console = Console()

# Map family tokens to repository URLs
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

def _repo_url_for_alias(alias: str) -> str:
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

def _norm_token(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch in "-_")


def _canonical_alias(name: str) -> str:
    aliases = _get_aliases()
    norm = _norm_token(name)
    for key in aliases.keys():
        if _norm_token(key) == norm:
            return key
    return norm


def _group_pairs(pairs):
    grouped = {}
    for env, model in pairs:
        alias = _canonical_alias(model)
        grouped.setdefault(env, set()).add(alias)
    return {e: sorted(list(ms)) for e, ms in grouped.items()}


def _dedup_pairs(pairs):
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


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize or update persistent config (base env, models path).

    Prints a grouped summary of configured (env, model) pairs.
    """
    cfg = load_config()
    if args.base_env:
        cfg["base_env"] = args.base_env
    if args.models_path is not None:
        cfg["models_path"] = args.models_path
    save_config(cfg)
    console.print(Rule("[bold green]Config initialized/updated"))

    pairs = _dedup_pairs(cfg.get("full_models", []))
    grouped = _group_pairs(pairs)
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("#", style="dim")
    table.add_column("env", style="cyan")
    table.add_column("model", style="magenta")
    for idx, (env, models) in enumerate(sorted(grouped.items()), start=1):
        table.add_row(str(idx), env, ",".join(models))
    console.print(table)
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Add a new (env, model) pair to the config."""
    add_model(args.env, args.model)
    console.print(f"Added: env='{args.env}' model='{args.model}'")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    """Remove either all pairs for an env or a specific (env, model)."""
    changed = remove_model(args.env, args.model)
    if changed:
        console.print("Removed entry.")
        return 0
    console.print("No matching entry found.")
    return 1


def show_config(cfg=None) -> None:
    """Pretty-print current configuration grouped by env."""
    if cfg is None:
        cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    grouped = _group_pairs(pairs)
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("#", style="dim")
    table.add_column("env", style="cyan")
    table.add_column("model", style="magenta")
    for idx, (env, models) in enumerate(sorted(grouped.items()), start=1):
        table.add_row(str(idx), env, ",".join(models))
    console.print(table)


def cmd_show(args: argparse.Namespace) -> int:
    """Show configured model aliases and their status.

    One row per unique alias; status comes from persisted runs when present,
    otherwise defaults to ✓ for aliases supported by the registry and × for unknowns.
    """
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))

    aliases_cfg = []
    seen = set()
    for _env, model in pairs:
        alias = _canonical_alias(model)
        if alias not in seen:
            seen.add(alias)
            aliases_cfg.append(alias)

    status_map = get_model_status_map()
    known_aliases = set((_get_aliases() or {}).keys())

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("alias", style="cyan")
    table.add_column("status", style="green")
    for alias in sorted(aliases_cfg):
        st = status_map.get(alias)
        if st == "broken":
            mark = "×"
        elif st == "ok":
            mark = "✓"
        else:
            mark = "✓" if alias in known_aliases else "×"
        table.add_row(alias, mark)
    console.print(table)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List supported model aliases with optional filter, repo and configured flag."""
    aliases = _get_aliases()
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    configured_aliases = { _canonical_alias(m) for _e, m in pairs }

    rows = sorted((alias, target) for alias, target in aliases.items())

    if args.contains:
        sub = args.contains.lower()
        rows = [r for r in rows if sub in r[0].lower() or sub in (r[1] or "").lower() or sub in _repo_url_for_alias(r[0]).lower()]
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("alias", style="cyan")
    table.add_column("registry", style="magenta")
    table.add_column("repo", style="blue")
    table.add_column("configured", style="green")
    for alias, target in rows:
        url = _repo_url_for_alias(alias)
        configured_mark = "✓" if alias in configured_aliases else "×"
        table.add_row(alias, target, url, configured_mark)
    console.print(table)
    return 0


def cmd_packages(args: argparse.Namespace) -> int:
    """List package tokens and their known variants (autocomplete help)."""
    rows = sorted(PACKAGE_VARIANTS.items())
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("package", style="cyan")
    table.add_column("variants", style="magenta")
    for pkg, variants in rows:
        table.add_row(pkg, ", ".join(variants))
    console.print(table)
    return 0


def _generate_worker() -> str:
    content = f"""
import sys
import os
import importlib.util
import warnings
import inspect

warnings.filterwarnings("ignore")

def main():
    try:
        user_script_path = sys.argv[1]
        calculator_name_arg = sys.argv[2]
        model_name_arg = sys.argv[3]
        models_path_arg = sys.argv[4] if len(sys.argv) > 4 else ""
        repo_root_arg = sys.argv[5] if len(sys.argv) > 5 else ""

        # Ensure local package is importable inside target env
        if repo_root_arg and os.path.isdir(repo_root_arg):
            sys.path.insert(0, repo_root_arg)

        from ip_orch.core.model_factory import ModelFactory

        # Initialize calculator (defines 'calc') using ModelFactory
        calc = ModelFactory.create(model_name_arg, models_path=models_path_arg)

        if not os.path.exists(user_script_path):
            print(f"[Worker ERROR] User script not found: {{user_script_path}}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("user_logic", user_script_path)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)

        # Discover a target function automatically
        preferred = [
            "main",
            "mlip_entry",
            "your_function",
            "calculators_test",
            "run",
        ]
        logic_function = None
        for name in preferred:
            fn = getattr(user_module, name, None)
            if callable(fn):
                logic_function = fn
                break
        if logic_function is None:
            candidates = [
                (n, f) for n, f in inspect.getmembers(user_module, inspect.isfunction)
                if getattr(f, "__module__", None) == "user_logic"
            ]
            for _, fn in candidates:
                try:
                    if len(inspect.signature(fn).parameters) >= 2:
                        logic_function = fn
                        break
                except Exception:
                    continue
        if logic_function is None:
            names = ", ".join(n for n, _ in candidates) if 'candidates' in locals() else "(none)"
            print("[Worker ERROR] Could not find a function to run. Define 'mlip_entry(calculator_name, calc)'. Found:", names)
            sys.exit(1)

        logic_function(calculator_name_arg, calc)
    except Exception as e:
        print(f"[Worker ERROR] Failure running {{calculator_name_arg}}: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", prefix="ip-orch_worker_")
    f.write(content)
    f.close()
    return f.name


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the provided user script across configured models.

    Also records per-model success/failure to surface in the models list.
    """
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    models_path = cfg.get("models_path", "")
    envs_base_dir = cfg.get("envs_base_dir", "")
    if args.only:
        only = {_clean_env(x.strip()) for x in args.only.split(",") if x.strip()}
        pairs = [p for p in pairs if _clean_env(p[0]) in only]

    if not pairs:
        console.print("[red]No models configured. Use 'ip-orch add <env> <model>'.")
        return 1

    summary_lines = [f"{_clean_env(e)} → {m}" for e, m in pairs]
    summary = "\n".join(summary_lines) if summary_lines else "(no models)"
    console.print(Panel(f"IPORCH Orchestrator: starting execution\n{summary}", border_style="blue"))
    # compute repo root (parent of package dir) and pass to worker
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    for env_name, model_name in pairs:
        worker_path = _generate_worker()
        try:
            python_bin = _python_for_env(env_name, envs_base_dir)
            if python_bin:
                cmd = [
                    python_bin,
                    worker_path,
                    args.script,
                    env_name,
                    model_name,
                    models_path,
                    repo_root,
                ]
            else:
                cmd = [
                    "conda", "run", "-n", env_name,
                    "python", worker_path,
                    args.script,
                    env_name,
                    model_name,
                    models_path,
                    repo_root,
                ]
            header = f"{_clean_env(env_name).upper()} → {model_name}"
            console.print(Panel(f"{header}\n{' '.join(cmd)}", border_style="blue"))
            res = subprocess.run(cmd, text=True)
            alias_key = _canonical_alias(model_name)
            if res.returncode != 0:
                console.print(f"[red]Failed for {env_name} ({model_name})")

                try:
                    set_model_status(alias_key, "broken")
                except Exception:
                    pass
            else:
                console.print(f"[green]Success: {env_name}")
                try:
                    set_model_status(alias_key, "ok")
                except Exception:
                    pass
        finally:
            try:
                os.remove(worker_path)
            except Exception:
                pass
    return 0


def main(argv: List[str] = None) -> int:
    epilog = (
        "Commands (use one):\n"
        "  --add                ENV MODEL\n"
        "  --remove             ENV [MODEL]\n"
        "  --run                SCRIPT [--only env1,env2]\n"
        "  --available-models   [SUBSTR]\n"
        "  --configure\n"
    )
    parser = argparse.ArgumentParser(
        prog="ip-orch",
        description="IP-ORCH orchestrator CLI",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--add", nargs=2, metavar=("ENV", "MODEL"), help="Add (env, model) pair")
    grp.add_argument("--remove", nargs="+", metavar=("ENV", "MODEL"), help="Remove by env or a specific pair")
    grp.add_argument("--run", metavar="SCRIPT", help="Run your ASE script across models")
    grp.add_argument("--available-models", nargs="?", dest="available_models", const="", metavar="SUBSTR", help="List available model aliases (optional filter)")
    grp.add_argument("--configure", action="store_true", help="Interactive discovery and setup")

    parser.add_argument("--models-path", dest="models_path", default=None, help="Models path (reserved)")
    parser.add_argument("--only", dest="only", default=None, help="Comma-separated env filter (for --run)")

    args = parser.parse_args(argv)
    try:
        if args.add:
            env, model = args.add
            a = argparse.Namespace(env=env, model=model)
            return cmd_add(a)
        if args.remove:
            if len(args.remove) == 1:
                env, model = args.remove[0], None
            elif len(args.remove) >= 2:
                env, model = args.remove[0], args.remove[1]
            else:
                return 1
            a = argparse.Namespace(env=env, model=model)
            return cmd_remove(a)
        if args.run:
            a = argparse.Namespace(script=args.run, only=args.only)
            return cmd_run(a)
        # available models list
        if getattr(args, "available_models", None) is not None:
            a = argparse.Namespace(contains=(args.available_models or None))
            return cmd_models(a)
        if args.configure:
            return cmd_configure(argparse.Namespace())
        parser.print_help()
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user.[/yellow]")
        return 0


def _discover_conda_envs() -> List[str]:
    """Return a list of conda env names. Best-effort parsing."""
    try:
        proc = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            import json as _json
            data = _json.loads(proc.stdout)
            envs = []
            for p in data.get("envs", []):
                name = os.path.basename(p)
                if name == "envs":
                    name = os.path.basename(os.path.dirname(p))
                envs.append(name)
            return sorted(set(envs))
    except Exception:
        pass

    try:
        proc = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            envs = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                name = parts[0]
                if name and name != "*":
                    envs.append(name)
            return sorted(set(envs))
    except Exception:
        pass
    return []


def _guess_envs_dir() -> str:
    candidates = [
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
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
    """Return the token from DEFAULT_MODELS_BY_ENV that appears in env_name (fuzzy)."""
    nenv = _normalize_token(env_name)
    for token in DEFAULT_MODELS_BY_ENV.keys():
        if _normalize_token(token) in nenv:
            return token
    return ""


def _python_for_env(env_name: str, base_dir: str) -> str:
    """Return python interpreter for a venv/conda env under base_dir, if exists."""
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


def _interactive_edit_pairs(pairs: List[List[str]]) -> List[List[str]]:
    """Interactive add/edit/remove loop for (env, model) pairs."""
    while True:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("#", style="dim")
        table.add_column("env", style="cyan")
        table.add_column("model", style="magenta")
        for i, (e, m) in enumerate(pairs, start=1):
            table.add_row(str(i), e, m)
        console.print(table)

        choice = Prompt.ask("(a)dd / (r)emove / (m)odify / (d)one", choices=["a", "r", "m", "d"], default="d")
        if choice == "d":
            return pairs
        if choice == "a":
            env = _clean_env(Prompt.ask("Environment name"))
            model = Prompt.ask("Model identifier")
            if [env, model] not in pairs:
                pairs.append([env, model])
        elif choice == "m":
            if not pairs:
                console.print("No entries to edit.")
                continue
            idx = int(Prompt.ask("Index to edit", default="1")) - 1
            if 0 <= idx < len(pairs):
                env = _clean_env(Prompt.ask("Environment name", default=pairs[idx][0]))
                model = Prompt.ask("Model identifier", default=pairs[idx][1])
                pairs[idx] = [env, model]
        elif choice == "r":
            if not pairs:
                console.print("No entries to remove.")
                continue
            idx = int(Prompt.ask("Index to remove", default="1")) - 1
            if 0 <= idx < len(pairs):
                pairs.pop(idx)


def cmd_configure(args: argparse.Namespace) -> int:
    console.print(Panel("[bold]Interactive configuration[/bold]\nConfigure base env, scan directories and select models.", border_style="blue"))
    cfg = load_config()

    base_env = Prompt.ask("Base IP-Orch environment (e.g. mlip):", default=cfg.get("base_env", os.environ.get("CONDA_DEFAULT_ENV", "")))

    # Step 1: base directory (Conda envs or Python venvs)
    default_envs_dir = _guess_envs_dir()
    envs_base_dir = Prompt.ask("Base MLIPs environments directory (e.g. ~/miniconda3/envs)", default=cfg.get("envs_base_dir", default_envs_dir))

    discovered = _discover_envs_from_dir(envs_base_dir)
    # Step 2: add current env
    current_env = _current_conda_env()
    if current_env:
        discovered.add(current_env)
    # Step 3: augment with conda list fallback
    discovered |= set(_discover_conda_envs())
    discovered_list = sorted(discovered)
    if discovered_list:
        table_envs = Table(box=box.SIMPLE_HEAVY, header_style="bold")
        table_envs.add_column("env", style="cyan")
        table_envs.add_column("match", style="magenta")
        for e in discovered_list:
            token = _match_known_token(e)
            table_envs.add_row(e, token or "-")
        console.print(table_envs)
    else:
        console.print("No environments found under base directory.")

    proposed = []
    for env in discovered_list:
        match_token = _match_known_token(env)
        if match_token:
            variants = PACKAGE_VARIANTS.get(match_token)
            if variants:
                for model_alias in variants:
                    proposed.append([_clean_env(env), _canonical_alias(model_alias)])
            else:
                proposed.append([_clean_env(env), _canonical_alias(DEFAULT_MODELS_BY_ENV[match_token])])

    if proposed:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("env", style="cyan")
        table.add_column("suggested model", style="magenta")
        for e, m in proposed:
            table.add_row(e, m)
        console.print(table)

        if Confirm.ask("Add all suggested pairs?", default=True):
            pairs = proposed[:]
        else:
            pairs = []
    else:
        pairs = []

    # Merge with existing
    existing = cfg.get("full_models", [])
    for env, model in existing:
        alias = _canonical_alias(model)
        clean_env = _clean_env(env)
        if [clean_env, alias] not in pairs:
            pairs.append([clean_env, alias])

    # Interactive edit
    pairs = _interactive_edit_pairs(pairs)
    pairs = _dedup_pairs(pairs)

    cfg["base_env"] = base_env
    cfg["full_models"] = pairs
    cfg["envs_base_dir"] = envs_base_dir
    save_config(cfg)
    console.print("Configuration saved")
    # Final table: group models per env, comma-separated
    grouped = _group_pairs(cfg["full_models"]) if cfg.get("full_models") else {}
    final = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    final.add_column("#", style="dim")
    final.add_column("env", style="cyan")
    final.add_column("model", style="magenta")
    for idx, (env, models) in enumerate(sorted(grouped.items()), start=1):
        final.add_row(str(idx), env, ",".join(models))
    console.print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
