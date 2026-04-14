import os
import json
import tempfile
import subprocess
import argparse
import inspect
import re

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
    set_model_status,
)

from .helpers import (
    _get_aliases,
    _canonical_alias,
    _group_pairs,
    _dedup_pairs,
    _clean_env,
    DEFAULT_MODELS_BY_ENV,
    PACKAGE_VARIANTS,
)
from .repo_map import repo_url_for_alias
from .env_utils import (
    _discover_conda_envs,
    _guess_envs_dir,
    _discover_envs_from_dir,
    _current_conda_env,
    _match_known_token,
    _python_for_env,
)


console = Console()


def cmd_add(args: argparse.Namespace) -> int:
    add_model(args.env, args.model)
    console.print(f"Added: env='{args.env}' model='{args.model}'")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    changed = remove_model(args.env, args.model)
    if changed:
        console.print("Removed entry.")
        return 0
    console.print("No matching entry found.")
    return 1


def cmd_models(args: argparse.Namespace) -> int:
    aliases = _get_aliases()
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    configured_aliases = { _canonical_alias(m) for _e, m in pairs }

    rows = sorted((alias, target) for alias, target in aliases.items())

    if args.contains:
        sub = args.contains.lower()
        rows = [r for r in rows if sub in r[0].lower() or sub in (r[1] or "").lower() or sub in repo_url_for_alias(r[0]).lower()]
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("alias", style="cyan")
    table.add_column("registry", style="magenta")
    table.add_column("repo", style="blue")
    table.add_column("configured", style="green")
    for alias, target in rows:
        url = repo_url_for_alias(alias)
        configured_mark = "✓" if alias in configured_aliases else "×"
        table.add_row(alias, target, url, configured_mark)
    console.print(table)
    return 0


def _generate_worker() -> str:
    content = """
import sys
import os
import importlib.util
import warnings
import inspect
import json

warnings.filterwarnings("ignore")

def main():
    try:
        user_script_path = sys.argv[1]
        calculator_name_arg = sys.argv[2]
        model_name_arg = sys.argv[3]
        models_path_arg = sys.argv[4] if len(sys.argv) > 4 else ""
        repo_root_arg = sys.argv[5] if len(sys.argv) > 5 else ""

        linear_a_arg = sys.argv[6] if len(sys.argv) > 6 else ""
        linear_b_arg = sys.argv[7] if len(sys.argv) > 7 else ""
        linear_mode_arg = sys.argv[8] if len(sys.argv) > 8 else "total_energy"
        elements_arg = sys.argv[9] if len(sys.argv) > 9 else ""
        element_energies_json_arg = sys.argv[10] if len(sys.argv) > 10 else ""
        preflight_arg = sys.argv[11] if len(sys.argv) > 11 else ""

        def _to_float(s: str):
            s = (s or "").strip()
            return float(s) if s else None

        linear_a = _to_float(linear_a_arg)
        linear_b = _to_float(linear_b_arg)
        linear_mode = (linear_mode_arg or "total_energy").strip() or "total_energy"
        elements_raw = (elements_arg or "").strip()
        elements = [e.strip() for e in elements_raw.split(",") if e.strip()] if elements_raw else []
        element_energies_json_arg = (element_energies_json_arg or "").strip()
        preflight = (preflight_arg or "").strip() in ("1", "true", "True", "yes", "preflight")

        if repo_root_arg and os.path.isdir(repo_root_arg):
            sys.path.insert(0, repo_root_arg)

        from ip_orch.core.model_factory import ModelFactory
        from ip_orch.core.energy_correction import (
            wrap_linear_energy_correction,
            wrap_reference_energy_correction,
        )
        from ase import Atoms

        calc = ModelFactory.create(model_name_arg, models_path=models_path_arg)

        if (linear_a is None) ^ (linear_b is None):
            raise ValueError("Provide both a and b (or neither).")

        # Optional: compute element reference energies using the MLIP calculator itself.
        element_energies = None
        if element_energies_json_arg:
            try:
                element_energies = json.loads(element_energies_json_arg)
            except Exception as exc:
                raise ValueError(f"Failed to parse element energies JSON: {exc}")
        elif elements:
            element_energies = {}
            for sym in elements:
                # Single atom in a large non-periodic box.
                atom = Atoms(sym, positions=[(0.0, 0.0, 0.0)], cell=(20.0, 20.0, 20.0), pbc=False)
                atom.calc = calc
                element_energies[sym] = float(atom.get_potential_energy())

        if preflight:
            print("IPORCH_ELEMENT_ENERGIES=" + json.dumps(element_energies or {}))
            return

        calc = wrap_linear_energy_correction(calc, a=linear_a, b=linear_b, mode=linear_mode)
        calc = wrap_reference_energy_correction(calc, element_energies=element_energies)

        if not os.path.exists(user_script_path):
            print(f"[Worker ERROR] User script not found: {{user_script_path}}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("user_logic", user_script_path)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)

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
    cfg = load_config()
    pairs = _dedup_pairs(cfg.get("full_models", []))
    models_path = cfg.get("models_path", "")
    envs_base_dir = cfg.get("envs_base_dir", "")
    # Selection: either --envs or --models must be provided by caller
    selected_pairs = []
    if getattr(args, "envs", None):
        envs_sel = {_clean_env(x.strip()) for x in args.envs.split(",") if x.strip()}
        selected_pairs = [p for p in pairs if _clean_env(p[0]) in envs_sel]
    elif getattr(args, "models", None):
        models_sel = { _canonical_alias(x.strip()) for x in args.models.split(",") if x.strip() }
        selected_pairs = [p for p in pairs if _canonical_alias(p[1]) in models_sel]
    else:
        console.print("[red]For --run, provide either --envs or --models.")
        return 2

    if not selected_pairs:
        console.print("[red]No matching (env, model) pairs found for the selection.")
        console.print("Use 'ip-orch --configure' or 'ip-orch --add ENV MODEL' to set them.")
        return 1

    summary_lines = [f"{_clean_env(e)} → {m}" for e, m in selected_pairs]
    summary = "\n".join(summary_lines) if summary_lines else "(no models)"
    console.print(Panel(f"IP-Orch: starting execution\n{summary}", border_style="blue"))

    # Optional linear correction parameters (can come from JSON config and/or args)
    linear_a = getattr(args, "energy_linear_a", None)
    linear_b = getattr(args, "energy_linear_b", None)
    linear_mode = getattr(args, "energy_linear_mode", None) or "total_energy"
    elements = getattr(args, "correction_elements", None)
    no_energy_corr = bool(getattr(args, "no_energy_correction", False))
    linear_cfg_path = getattr(args, "energy_linear_config", None)
    if linear_cfg_path:
        try:
            with open(linear_cfg_path, "r", encoding="utf-8") as f:
                linear_cfg = json.load(f) or {}
            # Only fill missing values from config.
            if linear_a is None:
                linear_a = linear_cfg.get("a", None)
            if linear_b is None:
                linear_b = linear_cfg.get("b", None)
            if getattr(args, "energy_linear_mode", None) in (None, ""):
                linear_mode = linear_cfg.get("mode", linear_mode) or linear_mode
            if elements is None and not no_energy_corr:
                elements = linear_cfg.get("correction_elements", None)
        except Exception as exc:
            console.print(f"[red]Failed to read --energy-linear-config: {exc}")
            return 2

    if no_energy_corr:
        linear_a = None
        linear_b = None
        elements = None

    if (linear_a is None) ^ (linear_b is None):
        console.print("[red]Provide both --energy-linear-a and --energy-linear-b (or neither).")
        return 2
    if isinstance(elements, (list, tuple)):
        elements = ",".join(str(e).strip() for e in elements if str(e).strip())
    elements_enabled = bool(elements)
    linear_enabled = (linear_a is not None and linear_b is not None)
    if linear_enabled and linear_mode not in ("total_energy", "per_atom"):
        console.print("[red]--energy-linear-mode must be 'total_energy' or 'per_atom'.")
        return 2

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    for env_name, model_name in selected_pairs:
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

            element_energies = None
            if elements_enabled:
                # Preflight to compute per-element reference energies inside the MLIP env,
                # so we can show the correction term in the panel.
                preflight_cmd = list(cmd)
                preflight_cmd.extend(
                    [
                        "" if linear_a is None else str(linear_a),
                        "" if linear_b is None else str(linear_b),
                        str(linear_mode),
                        str(elements),
                        "",  # element_energies_json (computed in preflight)
                        "preflight",
                    ]
                )
                pre = subprocess.run(preflight_cmd, text=True, capture_output=True)
                if pre.returncode != 0:
                    console.print(f"[red]Failed to preflight element energies for {env_name} ({model_name}).")
                    if pre.stdout:
                        console.print(pre.stdout)
                    if pre.stderr:
                        console.print(pre.stderr)
                    return 2
                for line in (pre.stdout or "").splitlines():
                    if line.startswith("IPORCH_ELEMENT_ENERGIES="):
                        try:
                            element_energies = json.loads(line.split("=", 1)[1])
                        except Exception:
                            element_energies = None
                        break

            if linear_enabled or elements_enabled:
                cmd.extend(
                    [
                        "" if linear_a is None else str(linear_a),
                        "" if linear_b is None else str(linear_b),
                        str(linear_mode),
                        "" if not elements else str(elements),
                        "" if not element_energies else json.dumps(element_energies),
                        "",  # preflight flag (empty = normal run)
                    ]
                )
            header = f"{_clean_env(env_name).upper()} → {model_name}"
            # Display command with an explicit break after the worker path.
            cmd_display = " ".join(cmd)
            try:
                worker_idx = cmd.index(worker_path)
                if worker_idx + 1 < len(cmd):
                    cmd_display = " ".join(cmd[: worker_idx + 1]) + "\n" + " ".join(cmd[worker_idx + 1 :])
            except ValueError:
                pass

            extra = ""
            if element_energies:
                parts = []
                for k, v in sorted(element_energies.items()):
                    try:
                        parts.append(f"{k}={float(v):.2f} eV")
                    except Exception:
                        parts.append(f"{k}={v} eV")
                # Ensure a clean line break separating from the command.
                extra = "\nreference energy correction: " + " | ".join(parts)

            console.print(Panel(f"{header}\n{cmd_display}{extra}", border_style="blue"))
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


def _interactive_edit_pairs(pairs: List[List[str]]):
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
            raw = Prompt.ask("Index(es) to remove (e.g. 1 3,5)", default="1")
            tokens = [t for t in re.split(r"[\s,]+", (raw or "").strip()) if t]
            try:
                idxs = sorted({int(t) - 1 for t in tokens}, reverse=True)
            except ValueError:
                console.print("[red]Invalid indices.[/red]")
                continue
            removed = 0
            for idx in idxs:
                if 0 <= idx < len(pairs):
                    pairs.pop(idx)
                    removed += 1
            if removed == 0:
                console.print("No valid indices removed.")
            else:
                console.print(f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")


def cmd_configure(args: argparse.Namespace) -> int:
    console.print(Panel("[bold]Interactive configuration[/bold]\nConfigure base env, scan directories and select models.", border_style="blue"))
    cfg = load_config()

    base_env = Prompt.ask("Base IP-Orch environment (e.g. mlip):", default=cfg.get("base_env", os.environ.get("CONDA_DEFAULT_ENV", "")))

    default_envs_dir = _guess_envs_dir()
    envs_base_dir = Prompt.ask("Base MLIPs environments directory (e.g. ~/miniconda3/envs)", default=cfg.get("envs_base_dir", default_envs_dir))

    discovered = _discover_envs_from_dir(envs_base_dir)
    current_env = _current_conda_env()
    if current_env:
        discovered.add(current_env)
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

    existing = cfg.get("full_models", [])
    for env, model in existing:
        alias = _canonical_alias(model)
        clean_env = _clean_env(env)
        if [clean_env, alias] not in pairs:
            pairs.append([clean_env, alias])

    pairs = _interactive_edit_pairs(pairs)
    pairs = _dedup_pairs(pairs)

    cfg["base_env"] = base_env
    cfg["full_models"] = pairs
    cfg["envs_base_dir"] = envs_base_dir
    save_config(cfg)
    console.print("Configuration saved")

    grouped = _group_pairs(cfg["full_models"]) if cfg.get("full_models") else {}
    final = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    final.add_column("#", style="dim")
    final.add_column("env", style="cyan")
    final.add_column("model", style="magenta")
    for idx, (env, models) in enumerate(sorted(grouped.items()), start=1):
        final.add_row(str(idx), env, ",".join(models))
    console.print(final)
    return 0
