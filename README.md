# QuantSim — High-Throughput Daily Equity Backtesting Engine (Java)

A daily, cost-aware portfolio simulator written in plain Java 21 with no runtime dependencies. It works in two modes:
- **alpha mode:** takes a raw cross-sectional signal and converts it to sector- or market-neutral, unit-gross weights.
- **weights mode:** takes target weights directly, for example from the optimizer in [AlphaForge](https://github.com/piyushr27/AlphaForge).

It reproduces a NumPy reference implementation to within 1e-10 and runs 4–18× faster.

## Results

| | |
|---|---|
| 12 years × 3,000 stocks (synthetic), realistic costs, single thread | **~150–170 ms** per backtest (≈ 60 M stock-days/s) |
| Real S&P 500 panel, 4,233 days × 503 stocks | **~17–22 ms** per backtest |
| Speedup vs. NumPy reference (identical inputs and outputs) | **4.1–4.3×** at 3,000 names, **15–18×** at 503 names |
| Parallel sweep: 49 AlphaForge alphas × 3 cost models | **147 backtests in 3.0 s** on 12 threads, including disk I/O |
| Java vs. reference daily PnL difference | ≤ 5e-12 on the real panel (8 cases), < 1e-10 in 7 randomized parity tests |

Benchmarks: `results/bench.csv` and `results/bench_vs_reference.csv`. Java times are the median of 7 warm runs after JIT warm-up. Timings on this WSL2 laptop vary by about ±15% between runs.

### Re-simulating AlphaForge's portfolio with realistic costs (OOS 2019-01 → 2026-10)
The AlphaForge optimizer weights were replayed through QuantSim as an independent check of AlphaForge's own PnL. QuantSim's zero-cost gross Sharpe is 0.557 against AlphaForge's 0.562. With a flat 5 bps cost, QuantSim's net Sharpe is 0.42 against AlphaForge's 0.43.

| Book size | Net Sharpe | Gross Sharpe | Impact cost / yr | Spread + commission / yr | Borrow / yr |
|---|---|---|---|---|---|
| $10M | 0.43 | 0.56 | 0.04% | 0.34% | 0.25% |
| $100M | 0.42 | 0.56 | 0.13% | 0.34% | 0.25% |
| $500M | 0.39 | 0.56 | 0.29% | 0.34% | 0.25% |
| $2B | 0.36 | 0.57 | 0.50% | 0.34% | 0.25% |
| $10B | 0.25 | 0.49 | 0.65% | 0.32% | 0.25% |

At $10B the 5% ADV participation cap starts leaving trades unfilled, which also degrades gross Sharpe. Source: `results/alphaforge_capacity.csv`.

### Alpha library under costs (`results/sweep.csv`)
Of the 49 AlphaForge alphas, 33 have a positive OOS Sharpe before costs, but only 15 remain positive under the realistic model. Fast reversal alphas are hit hardest: `rev_1d` goes from 0.11 before costs to −1.97 after.

## Model

Each day *t*, in order:
1. **Mark to market.** Dollar holdings earn the total return `ret[t]`, adjusted for splits and dividends. A held name whose price disappears is treated as delisted and closed at a configurable `delistReturn`.
2. **Targets** come from signal row `t − delay`. Delay must be at least 1, so a close-*t* signal trades at close *t+1*. Names outside the universe on the trade date get a target of 0.
3. **Trade** toward the target. Each day's trade is capped at `maxParticipation × ADV20`, and unfilled size carries to the next day. Costs on traded dollars *q*:
   `commission + half-spread = q·(c + s)·1e-4`, and impact `= q · k · σ · sqrt(q / ADV)` (the square-root law).
4. **Borrow fee** on overnight short dollars, `bps/252` per day.

"Realistic" defaults are 0.5 bps commission, 2 bps half-spread, impact k = 0.1, 50 bps/yr borrow and 5% of ADV. The parameters are defined in `CostModel.java`.

## Design
- **Columnar, date-major storage** (`Panel`): each field is a single `double[T*N]`, so a day's cross-section is contiguous. Every pass over a day is a sequential scan with no per-object overhead. Trading and exposure accounting are fused into one pass.
- **Binary format** (`PanelIO`): little-endian float64/float32 matrices, memory-mapped and bulk-copied. The 4,233 × 503 panel loads in about 0.3 s.
- **Parallel sweeps** (`Sweep`): the panel is immutable and shared across threads, each `Simulator` keeps private scratch buffers, and no locks are needed. A test checks that parallel results are bit-identical to sequential ones.
- **Reference implementation** (`python/reference.py`): a deliberately simple NumPy version that serves as the executable specification. `python/test_parity.py` runs both engines on randomized panels across every mode and cost setting.

## Tests
```
mvn package                          # 11 JUnit tests
pytest python/test_parity.py         # 7 Java-vs-NumPy parity cases
```
The JUnit tests cover:
- delay-1 timing: the trade-day return is not earned, and delay 0 is rejected;
- drift plus rebalance cost, square-root impact, the participation cap filling over several days, delisting, and short-only borrow;
- sector neutrality;
- **future signal rows cannot change past PnL**;
- a positive-IC signal is profitable;
- the binary format round-trips, and parallel runs equal sequential ones.

## Usage
```
mvn package                                           # JDK 21+, Maven 3.9+
python python/export_panel.py                         # needs ../AlphaForge/data (see AlphaForge README)
java -jar target/quantsim-1.0.0.jar sim   --panel data/panel --signal data/signals/mom_252_21.f32 --costs realistic --out results/mom.csv
java -jar target/quantsim-1.0.0.jar sim   --panel data/panel --signal data/alphaforge_weights.f32 --mode weights --book 1e8
java -jar target/quantsim-1.0.0.jar sweep --panel data/panel --signals data/signals --threads 12
java -jar target/quantsim-1.0.0.jar bench --T 3024 --N 3000
python python/parity.py && python python/bench_vs_reference.py
```

## Limitations
- The simulator works on daily bars and fills at the close. It has no intraday or limit-order-book model, and borrow rates are flat rather than stock-specific.
- The real panel inherits AlphaForge's survivorship bias (current S&P 500 constituents only). Delisting handling is therefore exercised in tests and synthetic benchmarks but rarely triggers on real data.
