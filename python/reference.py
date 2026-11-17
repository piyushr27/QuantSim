"""Reference implementation of the QuantSim simulator in NumPy.

Deliberately written for clarity rather than speed: one vectorized pass over
the cross-section per day. It is the specification the Java engine is tested
against (see ``parity.py`` / ``test_parity.py``); any behavioural change in
Java must be mirrored here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Costs:
    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    impact_coef: float = 0.0
    borrow_bps_annual: float = 0.0
    max_participation: float = 0.0

    @staticmethod
    def realistic() -> "Costs":
        return Costs(0.5, 2.0, 0.1, 50.0, 0.05)

    @staticmethod
    def flat(bps: float) -> "Costs":
        return Costs(commission_bps=bps)


@dataclass
class Panel:
    dates: np.ndarray
    tickers: list[str]
    sector: np.ndarray   # int ids
    close: np.ndarray    # (T, N)
    ret: np.ndarray
    adv: np.ndarray
    vol: np.ndarray
    valid: np.ndarray    # bool

    @property
    def shape(self):
        return self.close.shape


def load_panel(d: Path) -> Panel:
    d = Path(d)
    T, N = map(int, (d / "shape.txt").read_text().split())
    rd = lambda n: np.fromfile(d / n, dtype="<f8").reshape(T, N)
    sectors = (d / "sectors.txt").read_text().splitlines()
    ids = {s: i for i, s in enumerate(dict.fromkeys(sectors))}
    return Panel(np.array((d / "dates.txt").read_text().splitlines(), dtype="datetime64[D]"),
                 (d / "tickers.txt").read_text().splitlines(),
                 np.array([ids[s] for s in sectors]),
                 rd("close.f64"), rd("ret.f64"), rd("adv.f64"), rd("vol.f64"),
                 np.fromfile(d / "valid.u8", dtype=np.uint8).reshape(T, N).astype(bool))


def read_matrix(path: Path, T: int, N: int) -> np.ndarray:
    dt = "<f4" if str(path).endswith(".f32") else "<f8"
    return np.fromfile(path, dtype=dt).reshape(T, N).astype(np.float64)


def _targets_alpha(sig: np.ndarray, valid: np.ndarray, sector: np.ndarray, neut: str,
                   book: float, max_weight: float) -> np.ndarray:
    ok = valid & np.isfinite(sig)
    x = np.where(ok, sig, 0.0)
    if neut == "market" and ok.any():
        x = np.where(ok, x - x[ok].mean(), 0.0)
    elif neut == "sector":
        nsec = sector.max() + 1
        s = np.bincount(sector[ok], weights=x[ok], minlength=nsec).astype(float)
        c = np.bincount(sector[ok], minlength=nsec)
        mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
        x = np.where(ok, x - mean[sector], 0.0)
    g = np.abs(x).sum()
    tgt = x * (book / g) if g > 0 else np.zeros_like(x)
    if max_weight > 0:
        tgt = np.clip(tgt, -max_weight * book, max_weight * book)
    return tgt


def simulate(p: Panel, signal: np.ndarray, mode: str = "alpha", delay: int = 1, book: float = 20e6,
             neut: str = "sector", max_weight: float = 0.0, delist_return: float = 0.0,
             costs: Costs = Costs(), start: int = 0, end: int | None = None) -> dict[str, np.ndarray]:
    if delay < 1:
        raise ValueError("delay must be >= 1")
    T, N = p.shape
    end = T if end is None else end
    h = np.zeros(N)
    lin = (costs.commission_bps + costs.half_spread_bps) * 1e-4
    borrow_daily = costs.borrow_bps_annual * 1e-4 / 252
    out = {k: np.zeros(end - start) for k in
           ("gross", "net", "linear_cost", "impact_cost", "borrow_cost", "turnover", "long", "short")}
    for t in range(start, end):
        k = t - start
        price_ok = np.isfinite(p.close[t])
        # 1. mark to market / delist
        held = h != 0
        gone = held & ~price_ok
        r = p.ret[t]
        live = held & price_ok & np.isfinite(r)
        pnl = (h[gone] * delist_return).sum() + (h[live] * r[live]).sum()
        h[gone] = 0.0
        h[live] *= 1 + r[live]
        # 2. targets
        s = t - delay
        if s < 0:
            tgt = np.zeros(N)
        elif mode == "weights":
            w = signal[s]
            tgt = np.where(p.valid[t] & np.isfinite(w), w * book, 0.0)
        else:
            tgt = _targets_alpha(signal[s], p.valid[t], p.sector, neut, book, max_weight)
        # 3. trade
        dx = np.where(price_ok, tgt - h, 0.0)
        adv = p.adv[t]
        if costs.max_participation > 0:
            cap = np.where(adv > 0, costs.max_participation * adv, 0.0)
            dx = np.clip(dx, -cap, cap)
        q = np.abs(dx)
        h += dx
        h[np.abs(h) < 1e-9] = 0.0
        imp = 0.0
        if costs.impact_coef:
            m = (q > 0) & (adv > 0) & np.isfinite(p.vol[t])
            imp = (q[m] * costs.impact_coef * p.vol[t][m] * np.sqrt(q[m] / adv[m])).sum()
        commission = q.sum() * lin
        sht = -h[h < 0].sum()
        borrow = sht * borrow_daily
        out["gross"][k] = pnl / book
        out["linear_cost"][k] = commission / book
        out["impact_cost"][k] = imp / book
        out["borrow_cost"][k] = borrow / book
        out["net"][k] = (pnl - commission - imp - borrow) / book
        out["turnover"][k] = q.sum() / book
        out["long"][k] = h[h > 0].sum() / book
        out["short"][k] = sht / book
    return out
