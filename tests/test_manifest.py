import json

import pytest

from axm_zombie.manifest import load_manifest

def _minimal(**overrides):
    data = {
        "cluster": {
            "name": "c",
            "nodes": [
                {
                    "id": "n",
                    "host": "h",
                    "gpus": [{"index": 0, "name": "g", "vram_gb": 24, "mem_bw_gbps": 900}],
                }
            ],
        },
        "model": {"name": "m", "dtype": "fp16"},
        "policy": {"target_tps": 10},
    }
    data.update(overrides)
    return data

def _write(tmp_path, data, name="m.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)

def test_load_manifest_json(tmp_path):
    m = load_manifest(_write(tmp_path, _minimal()))
    assert m["schema_version"] == 1
    assert m["cluster"]["name"] == "c"
    assert len(m["cluster"]["nodes"]) == 1
    assert m["model"]["name"] == "m"
    assert m["model"]["kv_cache"] == "fp16"  # defaults to dtype
    assert m["policy"]["target_tps"] == 10
    assert m["policy"]["reserve_gb_per_gpu"] == 4.0  # defaulted

def test_load_manifest_yaml_equivalent(tmp_path):
    yaml = pytest.importorskip("yaml")
    data = _minimal()
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    from_yaml = load_manifest(str(p))
    from_json = load_manifest(_write(tmp_path, _minimal()))
    assert from_yaml == from_json

def test_missing_dtype_rejected(tmp_path):
    data = _minimal()
    del data["model"]["dtype"]
    with pytest.raises(ValueError, match="model.dtype"):
        load_manifest(_write(tmp_path, data))

def test_unknown_schema_version_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(_write(tmp_path, _minimal(schema_version=2)))

def test_strict_requires_mem_bw(tmp_path):
    data = _minimal()
    del data["cluster"]["nodes"][0]["gpus"][0]["mem_bw_gbps"]
    load_manifest(_write(tmp_path, data))  # lax mode accepts
    with pytest.raises(ValueError, match="strict"):
        load_manifest(_write(tmp_path, data), strict=True)
