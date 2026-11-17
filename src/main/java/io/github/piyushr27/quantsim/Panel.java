package io.github.piyushr27.quantsim;

import java.time.LocalDate;

/**
 * Immutable columnar market-data panel.
 *
 * <p>Every field is one contiguous {@code double[]} of length {@code T*N} in
 * <b>date-major</b> order: the cross-section for day {@code t} is
 * {@code [t*N, (t+1)*N)}. The simulator walks forward one day at a time and
 * touches whole cross-sections, so this layout keeps every inner loop a
 * sequential, cache-friendly scan.
 *
 * <ul>
 *   <li>{@code close} – close price (NaN when not trading / delisted)</li>
 *   <li>{@code ret}   – total return close(t-1) → close(t), split and dividend adjusted</li>
 *   <li>{@code adv}   – trailing 20-day average dollar volume, known at close t</li>
 *   <li>{@code vol}   – trailing 20-day daily return volatility, known at close t</li>
 *   <li>{@code valid} – tradable-universe membership at close t</li>
 * </ul>
 */
public final class Panel {
    public final int T;
    public final int N;
    public final LocalDate[] dates;
    public final String[] tickers;
    public final int[] sector;          // sector id per instrument, 0..nSectors-1
    public final int nSectors;
    public final double[] close;
    public final double[] ret;
    public final double[] adv;
    public final double[] vol;
    public final boolean[] valid;

    public Panel(LocalDate[] dates, String[] tickers, int[] sector,
                 double[] close, double[] ret, double[] adv, double[] vol, boolean[] valid) {
        this.T = dates.length;
        this.N = tickers.length;
        long cells = (long) T * N;
        for (Object a : new Object[]{close, ret, adv, vol}) {
            if (((double[]) a).length != cells) throw new IllegalArgumentException("field length != T*N");
        }
        if (valid.length != cells) throw new IllegalArgumentException("valid length != T*N");
        if (sector.length != N) throw new IllegalArgumentException("sector length != N");
        this.dates = dates;
        this.tickers = tickers;
        this.sector = sector;
        int max = -1;
        for (int s : sector) max = Math.max(max, s);
        this.nSectors = max + 1;
        this.close = close;
        this.ret = ret;
        this.adv = adv;
        this.vol = vol;
        this.valid = valid;
    }

    public int indexOf(LocalDate d) {
        for (int t = 0; t < T; t++) if (!dates[t].isBefore(d)) return t;
        return T;
    }
}
