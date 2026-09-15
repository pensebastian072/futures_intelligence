from futures_intelligence.providers.lse import LseProvider


def test_es_dated_symbol_pattern_does_not_confuse_estx(monkeypatch, tmp_path):
    class FakeSettings:
        data_root = tmp_path
        raw = {"lse": {"legacy_secret_file": str(tmp_path / "missing.json"),
                       "base_url": "https://example.invalid"}}

    provider = LseProvider(FakeSettings())
    provider.key = "not-real"
    values = iter([
        ({"datasets": ["futures"]}, type("R", (), {"reused": True})()),
        ([{"symbol": "ES.F"}, {"symbol": "ESTX50.F"}],
         type("R", (), {"reused": True})()),
    ])
    monkeypatch.setattr(provider, "get_json", lambda *args, **kwargs: next(values))
    assert provider.audit()["dated_contracts_present"] is False
