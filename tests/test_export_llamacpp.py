from pathlib import Path

from axm_zombie.manifest import load_manifest
from axm_zombie.planner import build_plan
from axm_zombie.export.llamacpp import export_llamacpp

REPO = Path(__file__).resolve().parent.parent

def test_llamacpp_scaffolds(tmp_path):
    plan = build_plan(load_manifest(str(REPO / "examples" / "cluster_4x3090_singlebox.json")))
    export_llamacpp(plan, tmp_path)

    host = (tmp_path / "run_llamacpp_host.sh").read_text()
    # All four GPUs appear as RPC endpoints in pipeline order with a VRAM-proportional split.
    assert "--rpc 127.0.0.1:50052,127.0.0.1:50053,127.0.0.1:50054,127.0.0.1:50055" in host
    assert "--tensor-split 24,24,24,24" in host

    workers = (tmp_path / "run_rpc_workers.sh").read_text()
    assert "node-a)" in workers
    for gpu, port in [(0, 50052), (1, 50053), (2, 50054), (3, 50055)]:
        assert f"CUDA_VISIBLE_DEVICES={gpu} rpc-server --host 0.0.0.0 --port {port} &" in workers

    notes = (tmp_path / "llamacpp_notes.md").read_text()
    assert "node-a:gpu:0 -> 127.0.0.1:50052" in notes
