package io.github.piyushr27.quantsim;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SimulatorTest {
    static final double EPS = 1e-12;

    /** Tiny hand-built panel: T days × 2 names, constant ADV/vol, all valid. */
    static Panel tiny(double[][] rets) {
        int T = rets.length, N = rets[0].length;
        LocalDate[] d = new LocalDate[T];
        for (int t = 0; t < T; t++) d[t] = LocalDate.of(2020, 1, 1).plusDays(t);
        double[] close = new double[T * N], ret = new double[T * N], adv = new double[T * N], vol = new double[T * N];
        boolean[] valid = new boolean[T * N];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < N; i++) {
                int k = t * N + i;
                ret[k] = rets[t][i];
                close[k] = Double.isNaN(rets[t][i]) && t > 0 ? Double.NaN : 100;
                adv[k] = 1e9;
                vol[k] = 0.02;
                valid[k] = !Double.isNaN(close[k]);
            }
        String[] tk = new String[N];
        int[] sec = new int[N];
        for (int i = 0; i < N; i++) tk[i] = "X" + i;
        return new Panel(d, tk, sec, close, ret, adv, vol, valid);
    }

    static double[] flat(double[][] m) {
        return Arrays.stream(m).flatMapToDouble(Arrays::stream).toArray();
    }

    @Test
    void longShortPnlUsesDelayOne() {
        // Signal on day 0 says long X0, short X1. Traded at close 1, earns returns of day 2.
        double[][] r = {{0, 0}, {0.50, -0.50}, {0.10, -0.05}, {0, 0}};
        Panel p = tiny(r);
        double[][] s = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
        SimResult res = new Simulator(p, SimConfig.alpha(CostModel.ZERO).withNeutralization(SimConfig.Neutralization.MARKET))
                .run(flat(s));
        assertEquals(0.0, res.gross[0], EPS);
        assertEquals(0.0, res.gross[1], EPS, "the day-1 return (known when trading at close 1) must not be earned");
        assertEquals(0.5 * 0.10 + (-0.5) * (-0.05), res.gross[2], EPS);
        assertEquals(1.0, res.turnover[1], EPS);
    }

    @Test
    void delayZeroIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> SimConfig.alpha(CostModel.ZERO).withDelay(0));
    }

    @Test
    void holdingsDriftAndRebalanceCostIsCharged() {
        double[][] r = {{0, 0}, {0, 0}, {0.10, 0.0}, {0, 0}};
        Panel p = tiny(r);
        double[] w = flat(new double[][]{{0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}});
        SimResult res = new Simulator(p, SimConfig.weights(CostModel.flat(10))).run(w);
        // Day 1 trade: |0.5| + |0.5| = 1.0 of book at 10 bps.
        assertEquals(1.0, res.turnover[1], EPS);
        assertEquals(-10e-4, res.net[1], EPS);
        // Day 2: long leg grew 10% -> 0.55; trade back 0.05 of book.
        assertEquals(0.05, res.gross[2], EPS);
        assertEquals(0.05, res.turnover[2], EPS);
        assertEquals(0.05 - 0.05 * 10e-4, res.net[2], EPS);
    }

    @Test
    void squareRootImpactAndParticipationCap() {
        double[][] r = {{0}, {0}, {0}, {0}};
        Panel p = tiny(r);
        Arrays.fill(p.adv, 1e6);
        CostModel c = new CostModel(0, 0, 0.5, 0, 0.10);  // cap 10% of ADV = $100k per day
        SimConfig cfg = new SimConfig(SimConfig.Mode.WEIGHTS, 1, 1e6, SimConfig.Neutralization.NONE, 0, 0, c);
        SimResult res = new Simulator(p, cfg).run(new double[]{0.25, 0.25, 0.25, 0.25});
        // Target $250k; fills $100k, $100k, $50k over days 1..3.
        assertEquals(0.10, res.turnover[1], EPS);
        assertEquals(0.10, res.turnover[2], EPS);
        assertEquals(0.05, res.turnover[3], EPS);
        double q = 1e5, expected = q * 0.5 * 0.02 * Math.sqrt(q / 1e6);
        assertEquals(expected / 1e6, res.impactCost[1], EPS);
    }

    @Test
    void delistedPositionIsClosedWithDelistReturn() {
        double[][] r = {{0, 0}, {0, 0}, {0, 0}, {Double.NaN, 0}, {Double.NaN, 0}};
        Panel p = tiny(r);
        double[] w = flat(new double[][]{{0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}});
        SimResult res = new Simulator(p, SimConfig.weights(CostModel.ZERO).withDelistReturn(-0.30)).run(w);
        assertEquals(0.5 * -0.30, res.gross[3], EPS);
        assertEquals(1, res.delisted[3]);
        assertEquals(0, res.nLong[3]);
        assertEquals(0.0, res.gross[4], EPS);
    }

    @Test
    void borrowChargedOnShortsOnly() {
        double[][] r = {{0, 0}, {0, 0}, {0, 0}};
        Panel p = tiny(r);
        CostModel c = new CostModel(0, 0, 0, 252.0, 0);  // 252 bps/yr -> 1 bp/day
        SimResult res = new Simulator(p, SimConfig.weights(c)).run(flat(new double[][]{{0.5, -0.5}, {0.5, -0.5}, {0.5, -0.5}}));
        assertEquals(0.5 * 1e-4, res.borrowCost[1], EPS);
    }

    @Test
    void sectorNeutralTargetsSumToZeroPerSector() {
        Panel p = Synthetic.panel(60, 50, 5, 1);
        double[] sig = Synthetic.signal(p, 0.05, 2);
        SimResult res = new Simulator(p, SimConfig.alpha(CostModel.ZERO)).run(sig, 0, 40);
        double[] h = res.finalHoldings;
        double[] bySector = new double[5];
        double gross = 0;
        for (int i = 0; i < p.N; i++) { bySector[p.sector[i]] += h[i]; gross += Math.abs(h[i]); }
        // holdings drifted one day after the last rebalance, so allow drift-sized error
        for (double v : bySector) assertEquals(0.0, v / gross, 0.02);
    }

    @Test
    void ignoresFutureSignalRows() {
        Panel p = Synthetic.panel(200, 40, 4, 3);
        double[] sig = Synthetic.signal(p, 0.1, 4);
        SimConfig cfg = SimConfig.alpha(CostModel.realistic());
        SimResult a = new Simulator(p, cfg).run(sig, 0, 150);
        double[] tampered = sig.clone();
        for (int k = 149 * p.N; k < tampered.length; k++) tampered[k] = 1e9 * (k % 3 - 1);  // rows ≥ 149
        SimResult b = new Simulator(p, cfg).run(tampered, 0, 150);
        assertArrayEquals(a.net, b.net, 0.0);
    }

    @Test
    void positiveIcSignalMakesMoney() {
        Panel p = Synthetic.panel(800, 200, 5, 5);
        SimResult r = new Simulator(p, SimConfig.alpha(CostModel.ZERO)).run(Synthetic.signal(p, 0.1, 6));
        assertTrue(r.stats().sharpe() > 3, r.stats().toString());
    }

    @Test
    void panelRoundTripsThroughDisk(@TempDir Path dir) throws Exception {
        Panel p = Synthetic.panel(30, 7, 3, 9);
        PanelIO.save(p, dir);
        Panel q = PanelIO.load(dir);
        assertArrayEquals(p.close, q.close, 0.0);
        assertArrayEquals(p.valid, q.valid);
        assertEquals(List.of(p.dates), List.of(q.dates));
        Path sig = dir.resolve("s.f64");
        double[] s = Synthetic.signal(p, 0.1, 1);
        PanelIO.writeMatrix(sig, s);
        assertArrayEquals(s, PanelIO.readMatrix(sig, p.T, p.N), 0.0);
        assertTrue(Files.size(sig) == 30L * 7 * 8);
    }

    @Test
    void parallelSweepMatchesSequential() throws Exception {
        Panel p = Synthetic.panel(300, 80, 4, 11);
        double[] sig = Synthetic.signal(p, 0.05, 12);
        SimConfig cfg = SimConfig.alpha(CostModel.realistic());
        double[] seq = new Simulator(p, cfg).run(sig).net;
        List<Sweep.Job> jobs = java.util.stream.IntStream.range(0, 16)
                .mapToObj(i -> new Sweep.Job("j" + i, () -> sig, cfg)).toList();
        for (Sweep.Outcome o : Sweep.run(p, jobs, 8, 0, p.T)) assertArrayEquals(seq, o.result().net, 0.0);
    }
}
