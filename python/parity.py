"""Java vs NumPy-reference parity and speed on the real panel.

For each (signal, config) case: run the Java CLI (`sim --out csv`) and the
reference simulator, compare daily gross/net/turnover/costs, and time both.
Writes results/parity.csv.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from reference import Costs, load_panel, read_matrix, simulate  # noqa: E402

JAR = ROOT / "target" / "quantsim-1.0.0.jar"
JAVA = str(Path.home() / ".local/opt/jdk/bin/java")
COLS = ("gross", "net", "turnover", "linear_cost", "impact_cost", "borrow_cost")

CASES = [
    # name, signal file, mode, costs flag, Costs, neut, delist
    ("mom_252_21 zero", "signals/mom_252_21.f32", "alpha", "zero", Costs(), "sector", 0.0),
    ("mom_252_21 realistic", "signals/mom_252_21.f32", "alpha", "realistic", Costs.realistic(), "sector", 0.0),
    ("rev_5d realistic", "signals/rev_5d.f32", "alpha", "realistic", Costs.realistic(), "sector", 0.0),
    ("sue_eps flat5 market", "signals/sue_eps.f32", "alpha", "flat:5", Costs.flat(5), "market", 0.0),
    ("size realistic delist", "signals/size.f32", "alpha", "realistic", Costs.realistic(), "sector", -0.3),
    ("AlphaForge weights zero", "alphaforge_weights.f32", "weights", "zero", Costs(), "none", 0.0),
    ("AlphaForge weights flat5", "alphaforge_weights.f32", "weights", "flat:5", Costs.flat(5), "none", 0.0),
    ("AlphaForge weights realistic", "alphaforge_weights.f32", "weights", "realistic", Costs.realistic(), "none", 0.0),
]


def run_java(sig: Path, mode: str, costs: str, neut: str, delist: float, out: Path) -> float:
    cmd = [JAVA, "-Xmx2g", "-jar", str(JAR), "sim", "--panel", str(ROOT / "data/panel"), "--signal", str(sig),
           "--mode", mode, "--costs", costs, "--neut", neut, "--delist", str(delist), "--out", str(out)]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    line = res.stdout.splitlines()[0]          # "... simulate X ms"
    return float(line.split("simulate")[1].split("ms")[0])


def main() -> None:
    p = load_panel(ROOT / "data/panel")
    T, N = p.shape
    rows = []
    tmp = ROOT / "results" / "_java_tmp.csv"
    for name, sig, mode, cflag, costs, neut, delist in CASES:
        s = read_matrix(ROOT / "data" / sig, T, N)
        java_ms = run_java(ROOT / "data" / sig, mode, cflag, neut, delist, tmp)
        t0 = time.perf_counter()
        ref = simulate(p, s, mode=mode, neut=neut, delist_return=delist, costs=costs)
        py_ms = (time.perf_counter() - t0) * 1e3
        j = pd.read_csv(tmp)
        diff = max(np.max(np.abs(j[c].to_numpy() - ref[c])) for c in COLS)
        sr = lambda x: np.sqrt(252) * x.mean() / x.std()
        rows.append(dict(case=name, days=T, names=N, max_abs_diff=diff, java_ms=java_ms, numpy_ms=py_ms,
                         speedup=py_ms / java_ms, net_sharpe_java=sr(j["net"].to_numpy()),
                         net_sharpe_ref=sr(ref["net"])))
        print(name, f"diff={diff:.2e} java={java_ms:.0f}ms numpy={py_ms:.0f}ms")
    tmp.unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "parity.csv", index=False, float_format="%.6g")
    print(f"max diff over all cases: {df.max_abs_diff.max():.2e}; median speedup {df.speedup.median():.0f}x")


if __name__ == "__main__":
    main()
