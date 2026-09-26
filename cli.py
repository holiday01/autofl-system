"""
AutoFL CLI

Usage:
  python -m autofl convert   <script.py> [--output fl_client.py]
  python -m autofl preflight <fl_client.py> --config global_config.yaml [--client-id site-1]
  python -m autofl run       <fl_client.py> --config global_config.yaml --num-clients N
  python -m autofl hardware  (detect and print local hardware profile)
"""
import argparse
import sys
from pathlib import Path


def cmd_hardware(_args) -> int:
    from autofl.hardware.detector import detect, print_profile
    print_profile(detect())
    return 0


def cmd_convert(args) -> int:
    from autofl.converter.ast_converter import convert
    out = convert(args.script, args.output)
    print(f"\nGenerated: {out}")
    if args.check:
        print("\nRunning syntax check on generated file...")
        import ast as _ast
        try:
            _ast.parse(out.read_text())
            print("  Syntax OK")
        except SyntaxError as e:
            print(f"  Syntax ERROR: {e}")
            return 1
    return 0


def cmd_preflight(args) -> int:
    from autofl.config.config_manager import ConfigManager
    from autofl.preflight.validator import run_preflight

    mgr = ConfigManager(args.config)
    local_cfg = args.local_config
    cfg = mgr.build_client_config(
        local_config_path=local_cfg if local_cfg else None,
        auto_hardware=True,
    )

    if args.show_config:
        mgr.print_summary(cfg)

    result = run_preflight(
        client_id=args.client_id,
        fl_client_module_path=args.fl_module,
        config=cfg,
        required_packages=args.require or ["torch"],
        report_path=args.report,
    )
    return 0 if result.success else 1


def cmd_run(args) -> int:
    """
    Minimal local simulation: spawn N clients, run FL rounds.
    Each client shares the same fl_module but gets its own config instance
    (simulating heterogeneous hardware by using auto_hardware=True per client).
    """
    from autofl.config.config_manager import ConfigManager
    from autofl.preflight.validator import run_preflight
    from autofl.fl_runtime.client import FLClient
    from autofl.fl_runtime.server import FLServer
    import torch

    mgr = ConfigManager(args.config)
    fl_module = Path(args.fl_module)
    n = args.num_clients

    print("=" * 50)
    print("AutoFL — Preflight phase")
    print("=" * 50)

    clients = []
    for i in range(n):
        cid = f"site-{i+1}"
        cfg = mgr.build_client_config(auto_hardware=True)
        cfg["client_id"] = cid

        result = run_preflight(cid, fl_module, cfg)
        if not result.success:
            print(f"  WARNING: {cid} failed preflight, will be excluded.")
            continue

        client = FLClient(cid, fl_module, cfg)
        clients.append(client)

    if not clients:
        print("No clients passed preflight.")
        return 1

    # Build initial global weights from first client's model
    init_weights = clients[0].get_weights()
    server = FLServer(
        global_weights=init_weights,
        config=mgr.global_cfg,
        results_dir=args.output_dir,
    )
    for c in clients:
        server.register_preflight(c.client_id, {"success": True})

    server.run(clients, num_rounds=args.num_rounds)
    model_path = server.save_model()
    print(f"\nModel saved: {model_path}")
    return 0


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="autofl",
        description="Auto-convert PyTorch scripts to FL and run federated training",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # hardware
    p_hw = sub.add_parser("hardware", help="Detect local hardware and suggest params")

    # convert
    p_conv = sub.add_parser("convert", help="Convert a training script to FL client module")
    p_conv.add_argument("script", help="Path to PyTorch training script")
    p_conv.add_argument("--output", "-o", default=None, help="Output path (default: <script>_fl_client.py)")
    p_conv.add_argument("--check", action="store_true", help="Syntax-check the generated file")

    # preflight
    p_pre = sub.add_parser("preflight", help="Run pre-flight validation on a client machine")
    p_pre.add_argument("fl_module", help="Path to generated fl_client module")
    p_pre.add_argument("--config", required=True, help="Path to global_config.yaml")
    p_pre.add_argument("--local-config", default=None, help="Path to local_config.yaml (optional)")
    p_pre.add_argument("--client-id", default="client-0")
    p_pre.add_argument("--require", nargs="*", help="Extra packages to check (e.g. timm)")
    p_pre.add_argument("--report", default=None, help="Save JSON report to this path")
    p_pre.add_argument("--show-config", action="store_true", help="Print effective config")

    # run
    p_run = sub.add_parser("run", help="Run FL simulation locally (multi-client)")
    p_run.add_argument("fl_module", help="Path to generated fl_client module")
    p_run.add_argument("--config", required=True, help="Path to global_config.yaml")
    p_run.add_argument("--num-clients", type=int, default=2)
    p_run.add_argument("--num-rounds", type=int, default=None, help="Override num_rounds")
    p_run.add_argument("--output-dir", default="fl_results")

    args = parser.parse_args(argv)
    dispatch = {
        "hardware": cmd_hardware,
        "convert": cmd_convert,
        "preflight": cmd_preflight,
        "run": cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
