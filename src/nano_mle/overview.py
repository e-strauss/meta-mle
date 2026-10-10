"""Static data overview: a harness-computed summary of every task source, without
model calls or plan code. It gives every role the schema and a sample of the data
when a profile runs without (or before) explorations.

Tables are summarised from a sample of their first rows (and the row count where
the format records it); directories by their file count and extensions.
"""

from pathlib import Path

SAMPLE_ROWS = 10_000
HEAD_ROWS = 5
MAX_COLUMNS = 200


def table_summary(frame, rows=None):
    columns = []
    for name in list(frame.columns)[:MAX_COLUMNS]:
        series = frame[name]
        column = {"name": str(name), "dtype": str(series.dtype),
                  "missing_fraction": round(float(series.isna().mean()), 4),
                  "unique": int(series.nunique(dropna=True))}
        if series.dtype.kind in "iuf" and series.notna().any():
            column.update(min=float(series.min()), max=float(series.max()), mean=float(series.mean()))
        columns.append(column)
    return {"rows": rows, "sampled_rows": len(frame), "columns_total": frame.shape[1],
            "columns": columns, "head": frame.head(HEAD_ROWS).astype(str).to_dict(orient="records")}


def summarize(location):
    import pandas as pd
    import pyarrow.dataset as ds

    lowered = location.lower().rstrip("/")
    if lowered.endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz", ".txt")):
        sep = "\t" if ".tsv" in lowered else ","
        return {"format": "csv", **table_summary(pd.read_csv(location, sep=sep, nrows=SAMPLE_ROWS))}
    if lowered.endswith(".parquet") or "://" in location or Path(location).is_dir():
        try:
            dataset = ds.dataset(location, format="parquet")
            sample = dataset.head(SAMPLE_ROWS).to_pandas()
            return {"format": "parquet", **table_summary(sample, dataset.count_rows())}
        except Exception:
            if "://" in location or not Path(location).is_dir():
                raise
    path = Path(location)
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        extensions = {}
        for p in files:
            extensions[p.suffix.lower() or "(none)"] = extensions.get(p.suffix.lower() or "(none)", 0) + 1
        return {"format": "directory", "files": len(files), "extensions": extensions,
                "examples": [str(p.relative_to(path)) for p in files[:HEAD_ROWS]]}
    return {"format": path.suffix.lower() or "file", "bytes": path.stat().st_size}


def data_overview(sources):
    """{source name: summary}; a source that cannot be read reports its error."""
    overview = {}
    for name, source in sources.items():
        try:
            overview[name] = summarize(source["path"])
        except Exception as error:
            overview[name] = {"error": " ".join(f"{type(error).__name__}: {error}".split())[:500]}
    return overview
