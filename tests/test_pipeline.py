"""
End-to-end pipeline test:
  1. Convert sample_training_script.py → fl_client module
  2. Run preflight on the generated module
  3. Run 2-round FL simulation with 2 local clients
"""
import sys
import tempfile
from pathlib import Path

import pytest

# Allow running as: python tests/test_pipeline.py
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))


# -----------------------------------------------------------------------
# 1. Converter
# -----------------------------------------------------------------------

def test_convert_produces_valid_python():
    from autofl.converter.ast_converter import convert

    src = ROOT / "examples" / "sample_training_script.py"
    with tempfile.TemporaryDirectory() as td:
        out = convert(src, Path(td) / "fl_client.py")
        assert out.exists(), "Output file not created"
        import ast
        ast.parse(out.read_text())   # raises SyntaxError if broken


def test_analysis_detects_components():
    from autofl.converter.ast_converter import ScriptAnalyzer

    src = (ROOT / "examples" / "sample_training_script.py").read_text()
    result = ScriptAnalyzer(src).analyse()
    assert result.model_class == "SimpleClassifier"
    assert result.dataset_class == "SyntheticDataset"
    assert result.optimizer_class == "AdamW"
    assert "CrossEntropyLoss" in result.loss_calls


# -----------------------------------------------------------------------
# 2. Hardware detector
# -----------------------------------------------------------------------

def test_hardware_detect_runs():
    from autofl.hardware.detector import detect
    hw = detect()
    assert hw.cpu_count >= 1
    assert hw.suggested_batch_size >= 1


# -----------------------------------------------------------------------
# 3. Config manager
# -----------------------------------------------------------------------

def test_config_merge_respects_locked():
    from autofl.config.config_manager import ConfigManager, ConfigError
    import tempfile, yaml

    global_cfg = {
        "num_rounds": 5,
        "local_epochs": 1,
        "aggregation_algorithm": "FedAvg",
        "learning_rate": 0.001,
        "allowed_overrides": ["batch_size", "use_amp"],
        "batch_size": 16,
    }
    with tempfile.TemporaryDirectory() as td:
        gpath = Path(td) / "global.yaml"
        gpath.write_text(yaml.dump(global_cfg))
        mgr = ConfigManager(gpath)

        # valid local override
        lpath = Path(td) / "local.yaml"
        lpath.write_text(yaml.dump({"batch_size": 32, "use_amp": True}))
        cfg = mgr.build_client_config(lpath, auto_hardware=False)
        assert cfg["local"]["batch_size"] == 32

        # invalid: try to override locked param
        bad = Path(td) / "bad.yaml"
        bad.write_text(yaml.dump({"learning_rate": 1.0}))
        with pytest.raises(ConfigError):
            mgr.build_client_config(bad, auto_hardware=False)


# -----------------------------------------------------------------------
# 4. Preflight
# -----------------------------------------------------------------------

def test_preflight_passes_on_sample():
    from autofl.converter.ast_converter import convert
    from autofl.preflight.validator import run_preflight

    src = ROOT / "examples" / "sample_training_script.py"
    with tempfile.TemporaryDirectory() as td:
        fl_mod = convert(src, Path(td) / "fl_client.py")

        config = {
            "num_rounds": 2,
            "local_epochs": 1,
            "aggregation_algorithm": "FedAvg",
            "learning_rate": 1e-3,
            "seed": 0,
            "data_path": ".",
            "local": {"batch_size": 4, "use_amp": False},
        }
        result = run_preflight("test-client", fl_mod, config)
        assert result.success, f"Preflight failed at [{result.error_stage}]: {result.error_message}"


# -----------------------------------------------------------------------
# 5. Full FL simulation
# -----------------------------------------------------------------------

def test_fl_run_two_clients():
    from autofl.converter.ast_converter import convert
    from autofl.preflight.validator import run_preflight
    from autofl.fl_runtime.client import FLClient
    from autofl.fl_runtime.server import FLServer

    src = ROOT / "examples" / "sample_training_script.py"
    with tempfile.TemporaryDirectory() as td:
        fl_mod = convert(src, Path(td) / "fl_client.py")

        base_cfg = {
            "num_rounds": 2,
            "local_epochs": 1,
            "aggregation_algorithm": "FedAvg",
            "learning_rate": 1e-3,
            "seed": 0,
            "data_path": ".",
            "local": {"batch_size": 4, "use_amp": False},
        }

        clients = []
        for i in range(2):
            cid = f"site-{i+1}"
            cfg = {**base_cfg, "client_id": cid}
            r = run_preflight(cid, fl_mod, cfg)
            assert r.success, f"{cid} preflight failed: {r.error_message}"
            clients.append(FLClient(cid, fl_mod, cfg))

        init_w = clients[0].get_weights()
        server = FLServer(init_w, base_cfg, results_dir=Path(td) / "results")
        for c in clients:
            server.register_preflight(c.client_id, {"success": True})

        server.run(clients, num_rounds=2)

        results_json = Path(td) / "results" / "fl_results.json"
        assert results_json.exists()
        import json
        data = json.loads(results_json.read_text())
        assert len(data) == 2   # 2 rounds


if __name__ == "__main__":
    print("Running tests directly...")
    test_convert_produces_valid_python();      print("PASS: convert syntax")
    test_analysis_detects_components();        print("PASS: analysis")
    test_hardware_detect_runs();               print("PASS: hardware")
    test_config_merge_respects_locked();       print("PASS: config")
    test_preflight_passes_on_sample();         print("PASS: preflight")
    test_fl_run_two_clients();                 print("PASS: fl simulation")
    print("\nAll tests passed.")
