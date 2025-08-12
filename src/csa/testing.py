import json
import numpy as np
from pathlib import Path
import polars as pl

from csa.utils import pull, sorted_unqs
from csa import (
    AggTeResult,
    CsaParams,
    dynamic_csa,
    calendar_csa,
    group_csa,
    simple_csa,
)

FP_DATA = Path(__file__).parents[2].joinpath("data", "testing")


def load_json(fp: Path | str):
    with open(fp, "r") as f:
        return json.load(f)


def assert_allclose_csa(csao: AggTeResult, csa_res: pl.DataFrame, csa_r: dict):
    assert np.allclose(e1 := pull(csao.estimates, "att"), e2 := pull(csa_res, "att")), (
        f"ATT differ: {e1} != {e2}"
    )
    assert np.allclose(e1 := csao.overall_att, e2 := csa_r["oatt"]), (
        f"OATT differ: {e1} != {e2}"
    )
    assert np.allclose(e1 := pull(csao.estimates, "se"), e2 := pull(csa_res, "se")), (
        f"SE differ: {e1} != {e2}"
    )
    assert np.allclose(e1 := csao.overall_se, e2 := csa_r["ose"]), (
        f"OSE differ: {e1} != {e2}"
    )


def test_dynamic_csa(csap: CsaParams, agg_data: dict):
    csao = dynamic_csa(csap)
    csa_r = agg_data["dynamic"]
    csa_res = pl.DataFrame(csa_r["res"])
    assert_allclose_csa(csao, csa_res, csa_r)
    print("Dynamic ok")


def test_calendar_csa(csap: CsaParams, agg_data: dict):
    csao = calendar_csa(csap)
    csa_r = agg_data["calendar"]
    csa_res = pl.DataFrame(csa_r["res"])
    assert_allclose_csa(csao, csa_res, csa_r)
    print("Calendar ok")


def test_group_csa(csap: CsaParams, agg_data: dict):
    csao = group_csa(csap)
    csa_r = agg_data["group"]
    csa_res = pl.DataFrame(csa_r["res"])
    assert_allclose_csa(csao, csa_res, csa_r)
    print("Group ok")


def test_simple_csa(csap: CsaParams, agg_data: dict):
    csao_b = simple_csa(csap)
    csa_r = agg_data["simple"]
    assert np.allclose(csao_b.overall_att, csa_r["oatt"]), (
        f"OATT differ: {csao_b.overall_att} != {csa_r['oatt']}"
    )
    assert np.allclose(csao_b.overall_se, csa_r["ose"]), (
        f"OSE differ: {csao_b.overall_se} != {csa_r['ose']}"
    )
    print("Simple ok")


def helper_test_csa(csap: CsaParams, agg_data: dict):
    test_dynamic_csa(csap, agg_data)
    test_calendar_csa(csap, agg_data)
    test_group_csa(csap, agg_data)
    test_simple_csa(csap, agg_data)


def assert_close(arr1: np.ndarray, arr2: np.ndarray | int):
    assert np.allclose(arr1, arr2), f"failed {arr1}; {arr2}"


def assert_cprobs(out: pl.DataFrame, t_col: str = "t"):
    K = pull(out, "K")
    ks = sorted_unqs(K)
    pge = pull(out, "pge")
    for k in ks:
        assert_close(pge[K == k].sum(), 1)

    t = pull(out, t_col)
    t = t
    ts = sorted_unqs(t[K >= 0])
    pgt = pull(out, "pgt")
    for tv in ts:
        assert_close(pgt[t == tv].sum(), 1)
