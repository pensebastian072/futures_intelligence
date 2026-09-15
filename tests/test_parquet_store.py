import pandas as pd

from futures_intelligence.storage import ParquetStore


def test_normalized_partitions_are_content_addressed(tmp_path):
    store = ParquetStore(tmp_path)
    frame = pd.DataFrame({"x": [1, 2], "date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    first = store.write(frame, table_name="features_daily", partition_key="year=2020",
                        feature_version="v1")
    second = store.write(frame, table_name="features_daily", partition_key="year=2020",
                         feature_version="v1")
    assert first == second
    assert first.exists()
    with store.manifest.connect() as con:
        count = con.execute("SELECT count(*) FROM partition_manifest").fetchone()[0]
    assert count == 1
