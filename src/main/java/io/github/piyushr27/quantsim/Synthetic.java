package io.github.piyushr27.quantsim;

import java.time.LocalDate;
import java.util.SplittableRandom;

/** Deterministic synthetic panels for tests and benchmarks. */
public final class Synthetic {
    private Synthetic() {}

    /**
     * Random-walk prices for N names over T business days. Each name lists and
     * delists at random times (about 10% of names delist mid-sample) so
     * universe churn and delisting are exercised.
     */
    public static Panel panel(int T, int N, int nSectors, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        LocalDate[] dates = new LocalDate[T];
        LocalDate d = LocalDate.of(2010, 1, 4);
        for (int t = 0; t < T; t++) {
            while (d.getDayOfWeek().getValue() > 5) d = d.plusDays(1);
            dates[t] = d;
            d = d.plusDays(1);
        }
        String[] tickers = new String[N];
        int[] sector = new int[N];
        int[] first = new int[N], last = new int[N];
        for (int i = 0; i < N; i++) {
            tickers[i] = "S" + i;
            sector[i] = i % nSectors;
            first[i] = rng.nextDouble() < 0.2 ? rng.nextInt(T / 2) : 0;
            last[i] = rng.nextDouble() < 0.1 ? first[i] + rng.nextInt(Math.max(1, T - first[i])) : T;
        }
        long cells = (long) T * N;
        double[] close = new double[(int) cells], ret = new double[(int) cells];
        double[] adv = new double[(int) cells], vol = new double[(int) cells];
        boolean[] valid = new boolean[(int) cells];
        double[] px = new double[N];
        for (int i = 0; i < N; i++) px[i] = 20 + 80 * rng.nextDouble();
        for (int t = 0; t < T; t++) {
            int b = t * N;
            for (int i = 0; i < N; i++) {
                int k = b + i;
                if (t < first[i] || t >= last[i]) {
                    close[k] = Double.NaN; ret[k] = Double.NaN; adv[k] = Double.NaN; vol[k] = Double.NaN;
                    continue;
                }
                double r = t == first[i] ? Double.NaN : 0.0002 + 0.015 * gauss(rng);
                if (!Double.isNaN(r)) px[i] *= 1 + r;
                close[k] = px[i];
                ret[k] = r;
                adv[k] = 5e6 * (1 + 20 * ((i * 7919) % 100) / 100.0);
                vol[k] = 0.015;
                valid[k] = t >= first[i] + 20;
            }
        }
        return new Panel(dates, tickers, sector, close, ret, adv, vol, valid);
    }

    /** Signal = noisy forecast of the next-next-day return (so a delay-1 strategy has edge). */
    public static double[] signal(Panel p, double ic, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        double[] s = new double[p.T * p.N];
        for (int t = 0; t < p.T; t++) {
            for (int i = 0; i < p.N; i++) {
                int k = t * p.N + i;
                double fut = t + 2 < p.T ? p.ret[(t + 2) * p.N + i] : Double.NaN;
                s[k] = Double.isFinite(fut) ? ic * fut / 0.015 + Math.sqrt(1 - ic * ic) * gauss(rng) : gauss(rng);
                if (Double.isNaN(p.close[k])) s[k] = Double.NaN;
            }
        }
        return s;
    }

    static double gauss(SplittableRandom r) {
        double u = r.nextDouble(), v = r.nextDouble();
        return Math.sqrt(-2 * Math.log(1 - u)) * Math.cos(2 * Math.PI * v);
    }
}
