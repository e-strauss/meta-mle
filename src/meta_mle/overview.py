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
    """Column statistics and the first rows as strings; missing values are None, so the
    summary is valid JSON."""
    import pandas as pd

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
            "columns": columns, "head": [{str(k): None if pd.isna(v) else str(v) for k, v in row.items()}
                                         for row in frame.head(HEAD_ROWS).to_dict(orient="records")]}


def listing(location):
    """Files under a directory (local or remote), relative to it; None for a file."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(location)
    if not fs.isdir(root):
        return None
    root = root.rstrip("/")
    return sorted(path[len(root) + 1:] for path in fs.find(root))


def summarize(location):
    import pandas as pd
    import pyarrow.dataset as ds

    lowered = location.lower().rstrip("/")
    if lowered.endswith((".csv", ".tsv", ".csv.gz", ".tsv.gz", ".txt")):
        sep = "\t" if ".tsv" in lowered else ","
        return {"format": "csv", **table_summary(pd.read_csv(location, sep=sep, nrows=SAMPLE_ROWS))}
    files = listing(location)
    if lowered.endswith(".parquet") or (files and all(f.endswith(".parquet") for f in files)):
        dataset = ds.dataset(location, format="parquet")
        return {"format": "parquet", **table_summary(dataset.head(SAMPLE_ROWS).to_pandas(), dataset.count_rows())}
    if files is not None:
        extensions = {}
        for f in files:
            suffix = Path(f).suffix.lower() or "(none)"
            extensions[suffix] = extensions.get(suffix, 0) + 1
        return {"format": "directory", "files": len(files), "extensions": extensions, "examples": files[:HEAD_ROWS]}
    import fsspec

    fs, path = fsspec.core.url_to_fs(location)
    return {"format": Path(path).suffix.lower() or "file", "bytes": fs.size(path)}


def data_overview(sources):
    """{source name: summary}; a source that cannot be read reports its error."""
    overview = {}
    for name, source in sources.items():
        try:
            overview[name] = summarize(source["path"])
        except Exception as error:
            overview[name] = {"error": " ".join(f"{type(error).__name__}: {error}".split())[:500]}
    return overview
