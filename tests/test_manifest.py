from axm_zombie.manifest import load_manifest

def test_load_manifest(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("""cluster: {name: c, nodes: [{id: n, host: h, gpus: [{index: 0, name: g, vram_gb: 24, mem_bw_gbps: 900}]}]}
model: {name: m}
policy: {target_tps: 10}
""")
    m = load_manifest(str(p))
    assert m.cluster_name == "c"
    assert len(m.nodes) == 1
    assert m.model.name == "m"
