"""Steady-state speed: Java engine vs the NumPy reference on identical inputs.

Cases: the real panel (4,233 days x 503 names) and a synthetic 12-year x
3,000-name panel written in QuantSim's binary format. Both engines run the
same sector-neutral alpha with realistic costs; Java timing is the median of
warm runs after JIT warm-up, NumPy the median of 3 runs. Results are checked
for parity and appended to results/bench_vs_reference.csv.
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

JAVA = str(Path.home() / ".local/opt/jdk/bin/java")
JAR = ROOT / "target" / "quantsim-1.0.0.jar"


def synthetic(dirpath: Path, T: int, N: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    d = dirpath
    d.mkdir(parents=True, exist_ok=True)
    ret = rng.normal(0.0002, 0.015, (T, N))
    ret[0] = np.nan
    close = 50 * np.exp(np.nancumsum(np.nan_to_num(ret), axis=0))
    first = np.where(rng.random(N) < 0.2, rng.integers(0, T // 2, N), 0)
    last = np.where(rng.random(N) < 0.1, first + rng.integers(1, T, N), T).clip(max=T)
    t = np.arange(T)[:, None]
    alive = (t >= first) & (t < last)
    close[~alive] = np.nan
    ret[~alive] = np.nan
    adv = np.where(alive, rng.lognormal(17, 1, N), np.nan)
    vol = np.where(alive, 0.015, np.nan)
    valid = alive & (t >= first + 20)
    (d / "shape.txt").write_text(f"{T} {N}\n")
    dates = pd.bdate_range("2010-01-04", periods=T).strftime("%Y-%m-%d")
    (d / "dates.txt").write_text("\n".join(dates) + "\n")
    (d / "tickers.txt").write_text("\n".join(f"S{i}" for i in range(N)) + "\n")
    (d / "sectors.txt").write_text("\n".join(f"sec{i % 11}" for i in range(N)) + "\n")
    for n, a in (("close", close), ("ret", ret), ("adv", adv), ("vol", vol)):
        a.astype("<f8").tofile(d / f"{n}.f64")
    valid.astype(np.uint8).tofile(d / "valid.u8")
    sig = rng.normal(size=(T, N))
    sig[~alive] = np.nan
    sig.astype("<f4").tofile(d / "signal.f32")
    return d / "signal.f32"


def bench(panel_dir: Path, sig_file: Path, label: str) -> dict:
    p = load_panel(panel_dir)
    T, N = p.shape
    s = read_matrix(sig_file, T, N)
    out = subprocess.run([JAVA, "-Xmx4g", "-jar", str(JAR), "sim", "--panel", str(panel_dir), "--signal", str(sig_file),
                          "--costs", "realistic", "--reps", "7", "--out", str(ROOT / "results/_b.csv")],
                         check=True, capture_output=True, text=True).stdout
    java_ms = float(out.split("warm median")[1].split("ms")[0])
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ref = simulate(p, s, costs=Costs.realistic())
        times.append((time.perf_counter() - t0) * 1e3)
    j = pd.read_csv(ROOT / "results/_b.csv")
    (ROOT / "results/_b.csv").unlink()
    diff = float(np.max(np.abs(j["net"].to_numpy() - ref["net"])))
    r = dict(case=label, days=T, names=N, java_warm_ms=java_ms, numpy_ms=float(np.median(times)),
             speedup=float(np.median(times)) / java_ms, max_abs_diff_net=diff)
    print(r)
    return r


def main() -> None:
    rows = [bench(ROOT / "data/panel", ROOT / "data/signals/mom_252_21.f32", "real S&P 500 panel")]
    tmp = ROOT / "data" / "synthetic_3000"
    sig = synthetic(tmp, 3024, 3000)
    rows.append(bench(tmp, sig, "synthetic 12y x 3000"))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "bench_vs_reference.csv", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
