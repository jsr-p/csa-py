from csa import estimate, agg_te

import polars as pl

res = estimate(
    data=(
        pl.read_csv(
            "data/fastdid/sim5.csv",
            ignore_errors=True,
            infer_schema_length=50_000,
            columns=["G", "time", "y", "unit"],
            schema_overrides={
                "G": pl.String,
                "time": pl.Int64,
                "y": pl.Float64,
                "unit": pl.String,
            },
        )
        .rename(
            {"G": "g", "unit": "id", "time": "t"},
        )
        .select("g", "t", "y", "id")
        .with_columns(pl.col("g").str.replace("Inf", "0").cast(pl.Int64))
    ),
    group="g",
    time="t",
    outcome="y",
    unit="id",
    balanced=True,
    control="notyet",
    method="dr",
    verbose=True,
)
agg_res = agg_te(res, method="dynamic", boot=False)
