"""Export an AlphaForge dataset to QuantSim's binary columnar format.

Reads ../AlphaForge/data/{prices,meta}.parquet and writes

    data/panel/            shape.txt dates.txt tickers.txt sectors.txt
                           close.f64 ret.f64 adv.f64 vol.f64 valid.u8
    data/signals/<alpha>.f32          AlphaForge neutralized alphas (if cached)
    data/alphaforge_weights.f32       AlphaForge optimizer weights (if present)

The universe rule mirrors AlphaForge: priced, >= 252 days of history,
close > $5, and already added to the S&P 500.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphaforge", default=str(ROOT.parent / "AlphaForge"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    af, out = Path(a.alphaforge), Path(a.out)
    (out / "panel").mkdir(parents=True, exist_ok=True)

    px = pd.read_parquet(af / "data" / "prices.parquet")
    meta = pd.read_parquet(af / "data" / "meta.parquet").set_index("ticker")
    wide = {c: px.pivot(index="date", columns="ticker", values=c).sort_index()
            for c in ("close", "adj_close", "volume")}
    close, adj, volu = wide["close"], wide["adj_close"], wide["volume"]
    dates, tick = close.index, close.columns
    ret = adj.pct_change(fill_method=None)
    adv = (close * volu).rolling(20, min_periods=16).mean()
    vol = ret.rolling(20, min_periods=16).std()
    history = close.notna().cumsum()
    added = pd.DataFrame(dates.values[:, None] >= meta.loc[tick, "date_added"].values[None, :], dates, tick)
    valid = close.notna() & (history >= 252) & (close > 5.0) & added

    d = out / "panel"
    (d / "shape.txt").write_text(f"{len(dates)} {len(tick)}\n")
    (d / "dates.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n")
    (d / "tickers.txt").write_text("\n".join(tick) + "\n")
    (d / "sectors.txt").write_text("\n".join(meta.loc[tick, "sector"].fillna("Unknown")) + "\n")
    for name, df in (("close", close), ("ret", ret), ("adv", adv), ("vol", vol)):
        df.to_numpy(dtype="<f8").tofile(d / f"{name}.f64")
    valid.to_numpy(dtype=np.uint8).tofile(d / "valid.u8")
    print(f"panel {len(dates)} x {len(tick)} -> {d}")

    cache = af / "cache" / "alphas_neutral"
    if cache.exists():
        (out / "signals").mkdir(exist_ok=True)
        for f in sorted(cache.glob("*.parquet")):
            s = pd.read_parquet(f).reindex(index=dates, columns=tick)
            s.to_numpy(dtype="<f4").tofile(out / "signals" / f"{f.stem}.f32")
        print(f"{len(list(cache.glob('*.parquet')))} alpha signals -> {out / 'signals'}")
    wfile = af / "results" / "portfolio_weights.parquet"
    if wfile.exists():
        w = pd.read_parquet(wfile).reindex(index=dates, columns=tick).fillna(0.0)
        w.to_numpy(dtype="<f4").tofile(out / "alphaforge_weights.f32")
        print(f"AlphaForge weights ({(w.abs().sum(axis=1) > 0).sum()} active days) -> {out / 'alphaforge_weights.f32'}")


if __name__ == "__main__":
    main()
