"""
Benchmark script for CSA estimation vs fastdid R package.
"""

from csa import estimate, bench


i = 4
data = bench.read_fastdid(f"data/fastdid/sim{i}.csv")
res = estimate(
    data=data,
    group="g",
    time="t",
    outcome="y",
    unit="id",
    balanced=True,
    control="notyet",
    method="dr",
    verbose=True,
)
res.estimates.write_csv("data/testing/fastdidpy.csv")
