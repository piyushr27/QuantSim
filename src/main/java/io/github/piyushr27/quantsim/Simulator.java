package io.github.piyushr27.quantsim;

/**
 * Daily event-loop simulator over a columnar {@link Panel}.
 *
 * <p>For each day t (in order):
 * <ol>
 *   <li><b>Mark to market.</b> Each held dollar position earns ret[t]. A held name
 *       whose close is missing at t is treated as delisted: it earns
 *       {@code delistReturn} and is closed at no cost.</li>
 *   <li><b>Targets.</b> From signal row {@code t - delay} (information known at that
 *       close). Names not in the universe at t get a target of 0.</li>
 *   <li><b>Trade.</b> Move toward the target, capped at {@code maxParticipation × ADV}.
 *       Pay commission + half-spread + square-root impact on the traded dollars.</li>
 *   <li><b>Borrow.</b> Charge the daily borrow fee on short dollars held overnight.</li>
 * </ol>
 * PnL on day t = Σ h<sub>t-1</sub>·ret<sub>t</sub> − costs<sub>t</sub>, reported as a
 * fraction of {@code bookSize}. Instances hold mutable scratch state and are not
 * thread-safe; create one per thread (the {@link Panel} itself is shared).
 */
public final class Simulator {
    private final Panel p;
    private final SimConfig cfg;
    private final double[] h;        // dollar holdings
    private final double[] target;   // scratch: target dollars
    private final double[] secSum;
    private final int[] secCnt;

    public Simulator(Panel panel, SimConfig cfg) {
        this.p = panel;
        this.cfg = cfg;
        this.h = new double[panel.N];
        this.target = new double[panel.N];
        this.secSum = new double[Math.max(1, panel.nSectors)];
        this.secCnt = new int[Math.max(1, panel.nSectors)];
    }

    /** Simulate dates [start, end). {@code signal} is a date-major T*N matrix aligned to the panel. */
    public SimResult run(double[] signal, int start, int end) {
        final int N = p.N;
        if (signal.length != (long) p.T * N) throw new IllegalArgumentException("signal must be T*N");
        if (start < 0 || end > p.T || start >= end) throw new IllegalArgumentException("bad date range");
        java.util.Arrays.fill(h, 0.0);
        final CostModel c = cfg.costs();
        final double book = cfg.bookSize();
        final double linRate = c.linearRate();
        final double borrowDaily = c.borrowBpsAnnual() * 1e-4 / 252.0;
        final double partCap = c.maxParticipation();
        SimResult r = new SimResult(p, start, end);

        for (int t = start; t < end; t++) {
            final int base = t * N;
            final int k = t - start;

            // 1. mark to market (+ delisting)
            double pnl = 0.0;
            int delisted = 0;
            for (int i = 0; i < N; i++) {
                double hi = h[i];
                if (hi == 0.0) continue;
                if (Double.isNaN(p.close[base + i])) {
                    pnl += hi * cfg.delistReturn();
                    h[i] = 0.0;
                    delisted++;
                    continue;
                }
                double ri = p.ret[base + i];
                if (Double.isFinite(ri)) {
                    pnl += hi * ri;
                    h[i] = hi * (1.0 + ri);
                }
            }

            // 2. targets from signal row t-delay
            int s = t - cfg.delay();
            if (s >= 0) {
                buildTargets(signal, s * N, base, book);
            } else {
                java.util.Arrays.fill(target, 0.0);
            }

            // 3. trade with participation cap and costs; 4. exposure stats (fused pass)
            double traded = 0.0, commission = 0.0, impact = 0.0;
            double lng = 0.0, sht = 0.0;
            int nl = 0, ns = 0;
            for (int i = 0; i < N; i++) {
                double hi = h[i];
                double dx = target[i] - hi;
                if (dx != 0.0 && !Double.isNaN(p.close[base + i])) {
                    double adv = p.adv[base + i];
                    boolean fill = true;
                    if (partCap > 0) {
                        double cap = (adv > 0) ? partCap * adv : 0.0;
                        if (dx > cap) dx = cap;
                        else if (dx < -cap) dx = -cap;
                        fill = dx != 0.0;
                    }
                    if (fill) {
                        double q = Math.abs(dx);
                        hi += dx;
                        if (Math.abs(hi) < 1e-9) hi = 0.0;
                        h[i] = hi;
                        traded += q;
                        commission += q * linRate;
                        impact += c.impactCost(q, adv, p.vol[base + i]);
                    }
                }
                if (hi > 0) { lng += hi; nl++; }
                else if (hi < 0) { sht -= hi; ns++; }
            }
            double borrow = sht * borrowDaily;

            r.gross[k] = pnl / book;
            r.linearCost[k] = commission / book;
            r.impactCost[k] = impact / book;
            r.borrowCost[k] = borrow / book;
            r.net[k] = (pnl - commission - impact - borrow) / book;
            r.turnover[k] = traded / book;
            r.longExposure[k] = lng / book;
            r.shortExposure[k] = sht / book;
            r.nLong[k] = nl;
            r.nShort[k] = ns;
            r.delisted[k] = delisted;
        }
        System.arraycopy(h, 0, r.finalHoldings, 0, N);
        return r;
    }

    public SimResult run(double[] signal) {
        return run(signal, 0, p.T);
    }

    private void buildTargets(double[] sig, int sBase, int tBase, double book) {
        final int N = p.N;
        if (cfg.mode() == SimConfig.Mode.WEIGHTS) {
            for (int i = 0; i < N; i++) {
                double w = sig[sBase + i];
                target[i] = (p.valid[tBase + i] && Double.isFinite(w)) ? w * book : 0.0;
            }
            return;
        }
        // ALPHA mode: usable names are those with a finite signal (known at close s)
        // that are still in the universe on the trade date t.
        SimConfig.Neutralization neut = cfg.neutralization();
        java.util.Arrays.fill(secSum, 0.0);
        java.util.Arrays.fill(secCnt, 0);
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            double x = sig[sBase + i];
            if (p.valid[tBase + i] && Double.isFinite(x)) {
                target[i] = x;
                sum += x; cnt++;
                int g = p.sector[i];
                secSum[g] += x; secCnt[g]++;
            } else {
                target[i] = Double.NaN;
            }
        }
        double gross = 0.0;
        double mean = cnt > 0 ? sum / cnt : 0.0;
        for (int i = 0; i < N; i++) {
            double x = target[i];
            if (Double.isNaN(x)) { target[i] = 0.0; continue; }
            switch (neut) {
                case MARKET -> x -= mean;
                case SECTOR -> x -= secSum[p.sector[i]] / secCnt[p.sector[i]];
                case NONE -> { }
            }
            target[i] = x;
            gross += Math.abs(x);
        }
        double scale = gross > 0 ? book / gross : 0.0;
        double cap = cfg.maxWeight() > 0 ? cfg.maxWeight() * book : Double.POSITIVE_INFINITY;
        for (int i = 0; i < N; i++) {
            double v = target[i] * scale;
            target[i] = v > cap ? cap : (v < -cap ? -cap : v);
        }
    }
}
