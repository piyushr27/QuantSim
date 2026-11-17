"""Parity tests: the Java engine must reproduce the NumPy reference exactly.

Requires the built jar (`mvn package`). Panels are random, written in the
binary format, simulated by both engines across all modes / cost settings.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from bench_vs_reference import synthetic  # noqa: E402
from reference import Costs, load_panel, read_matrix, simulate  # noqa: E402

JAR = ROOT / "target" / "quantsim-1.0.0.jar"
JAVA = str(Path.home() / ".local/opt/jdk/bin/java")
COLS = ("gross", "net", "turnover", "linear_cost", "impact_cost", "borrow_cost", "long", "short")

pytestmark = pytest.mark.skipif(not JAR.exists(), reason="run `mvn package` first")


@pytest.fixture(scope="module")
def panel_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("panel")
    synthetic(d, 400, 120, seed=3)
    return d


CASES = [
    ("alpha", "zero", Costs(), "sector", 0.0, None),
    ("alpha", "realistic", Costs.realistic(), "sector", 0.0, None),
    ("alpha", "realistic", Costs.realistic(), "market", -0.3, None),
    ("alpha", "flat:7.5", Costs.flat(7.5), "none", 0.0, None),
    ("alpha", "realistic", Costs.realistic(), "sector", 0.0, 0.01),   # per-name cap
    ("weights", "zero", Costs(), "none", 0.0, None),
    ("weights", "realistic", Costs.realistic(), "none", -1.0, None),
]


@pytest.mark.parametrize("mode,cflag,costs,neut,delist,maxw", CASES)
def test_java_matches_reference(panel_dir, tmp_path, mode, cflag, costs, neut, delist, maxw):
    p = load_panel(panel_dir)
    T, N = p.shape
    sig_file = panel_dir / "signal.f32"
    if mode == "weights":
        s = read_matrix(sig_file, T, N)
        w = np.nan_to_num(s) / np.maximum(np.abs(np.nan_to_num(s)).sum(axis=1, keepdims=True), 1e-12)
        sig_file = tmp_path / "w.f64"
        w.astype("<f8").tofile(sig_file)
    out = tmp_path / "java.csv"
    cmd = [JAVA, "-jar", str(JAR), "sim", "--panel", str(panel_dir), "--signal", str(sig_file), "--mode", mode,
           "--costs", cflag, "--neut", neut, "--delist", str(delist), "--out", str(out)]
    if maxw:
        cmd += ["--maxw", str(maxw)]
    subprocess.run(cmd, check=True, capture_output=True)
    j = pd.read_csv(out)
    ref = simulate(p, read_matrix(sig_file, T, N), mode=mode, neut=neut, delist_return=delist,
                   costs=costs, max_weight=maxw or 0.0)
    assert len(j) == T
    for c in COLS:
        # PnL/cost columns are printed with 12 significant digits, exposures with 9 decimals.
        tol = 1e-9 if c in ("long", "short") else 1e-10
        np.testing.assert_allclose(j[c].to_numpy(), ref[c], rtol=1e-10, atol=tol, err_msg=c)
    assert np.abs(ref["gross"]).sum() > 0  # the case actually trades
