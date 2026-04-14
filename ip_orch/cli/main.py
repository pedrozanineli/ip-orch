import argparse
from typing import List

from rich.console import Console

from .commands import cmd_add, cmd_remove, cmd_models, cmd_run, cmd_configure


console = Console()


def main(argv: List[str] = None) -> int:
    epilog = (
        "run selection (required with --run):\n"
        "  --envs ENV1,ENV2    Run all configured models for the selected environments.\n"
        "  --models A1,A2      Run only the specified model aliases across environments where they are configured.\n"
        "\n"
        "examples:\n"
        "  ip-orch --add mace mace_mp\n"
        "  ip-orch --remove mace mace_mp\n"
        "  ip-orch --run script.py --envs mace,orb\n"
        "  ip-orch --run script.py --models mace-mp,orb-v3\n"
        "  ip-orch --available-models mace\n"
    )
    parser = argparse.ArgumentParser(
        prog="ip-orch",
        description=(
            "IP-ORCH: orchestrate ASE runs across multiple MLIP environments"
        ),
        usage=(
            "ip-orch [-h]\n"
            "               (--add ENV MODEL | --remove ENV [MODEL ...] | --run SCRIPT |\n"
            "                --available-models [SUBSTR] | --configure)\n"
            "               [--models-path PATH]\n"
            "               [--envs ENV1,ENV2 | --models ALIAS1,ALIAS2]"
        ),
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--add", nargs=2, metavar=("ENV", "MODEL"), help="Register a (environment, model) pair")
    grp.add_argument(
        "--remove",
        nargs="+",
        metavar=("ENV", "MODEL"),
        help="Remove all models from ENV or specific ENV-MODEL pairs",
    )
    grp.add_argument(
        "--run",
        metavar="SCRIPT",
        help="Run an ASE script across selected models (requires one of --envs or --models)",
    )
    grp.add_argument(
        "--available-models",
        nargs="?",
        dest="available_models",
        const="",
        metavar="SUBSTR",
        help="List available model aliases (optionally filter by name)",
    )
    grp.add_argument("--configure", action="store_true", help="Interactive environment discovery and setup")

    parser.add_argument(
        "--models-path",
        dest="models_path",
        default=None,
        metavar="PATH",
        help="Custom models directory (advanced)",
    )
    parser.add_argument(
        "--envs",
        dest="envs",
        default=None,
        metavar="ENV1,ENV2",
        help="Select environments to run (uses all their configured models)",
    )
    parser.add_argument(
        "--models",
        dest="models",
        default=None,
        metavar="ALIAS1,ALIAS2",
        help="Select specific model aliases to run (across configured environments)",
    )

    args = parser.parse_args(argv)
    try:
        if args.add:
            env, model = args.add
            a = argparse.Namespace(env=env, model=model)
            return cmd_add(a)
        if args.remove:
            # --remove ENV [MODEL ...]
            if len(args.remove) == 1:
                env, model = args.remove[0], None
                a = argparse.Namespace(env=env, model=model)
                return cmd_remove(a)
            elif len(args.remove) >= 2:
                env, models = args.remove[0], args.remove[1:]
                rc = 1
                for model in models:
                    a = argparse.Namespace(env=env, model=model)
                    this_rc = cmd_remove(a)
                    # Consider overall success if any removal succeeded
                    if this_rc == 0:
                        rc = 0
                return rc
            else:
                return 1
        if args.run:
            # Enforce selection flags for --run
            if not args.envs and not args.models:
                console.print("[red]For --run, provide either --envs or --models.")
                return 2
            a = argparse.Namespace(script=args.run, envs=args.envs, models=args.models)
            return cmd_run(a)
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


if __name__ == "__main__":
    raise SystemExit(main())
