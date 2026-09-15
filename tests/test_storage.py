from pathlib import Path

from futures_intelligence.storage import ImmutableRawStore, ManifestStore
from futures_intelligence.types import DataRequest


def test_immutable_store_and_manifest_replay(tmp_path):
    request = DataRequest("fixture", "es", ("ESM8",), "json", "a", "b")
    raw = ImmutableRawStore(tmp_path)
    first = raw.write(request, b'{"ok":true}', suffix="json")
    second = raw.write(request, b'{"ok":true}', suffix="json")
    assert first.content_hash == second.content_hash
    assert not first.reused
    assert second.reused
    assert Path(first.raw_path).read_bytes() == b'{"ok":true}'

    manifest = ManifestStore(tmp_path)
    manifest.begin(request, cost_estimate_usd=0.01)
    manifest.complete(first)
    replay = manifest.successful(request.request_hash)
    assert replay and replay["content_hash"] == first.content_hash


def test_experiment_is_always_shadow(tmp_path):
    store = ManifestStore(tmp_path)
    store.record_experiment("x", {"promoted": True, "metric": 99})
    with store.connect() as con:
        value = con.execute(
            "SELECT promoted, manifest_json FROM experiment_manifest WHERE experiment_id='x'"
        ).fetchone()
    assert value[0] is False
    assert '"promoted": false' in value[1]

