package io.github.piyushr27.quantsim;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;

/** Daily simulation output (all money values as a fraction of book size). */
public final class SimResult {
    public final Panel panel;
    public final int start, end, days;
    public final double[] gross, net, linearCost, impactCost, borrowCost, turnover, longExposure, shortExposure;
    public final int[] nLong, nShort, delisted;
    public final double[] finalHoldings;

    SimResult(Panel p, int start, int end) {
        this.panel = p;
        this.start = start;
        this.end = end;
        this.days = end - start;
        gross = new double[days]; net = new double[days];
        linearCost = new double[days]; impactCost = new double[days]; borrowCost = new double[days];
        turnover = new double[days]; longExposure = new double[days]; shortExposure = new double[days];
        nLong = new int[days]; nShort = new int[days]; delisted = new int[days];
        finalHoldings = new double[p.N];
    }

    /** Statistics over days [from, to) relative to the panel's date index. */
    public Stats stats(int from, int to) {
        int a = Math.max(from, start) - start, b = Math.min(to, end) - start;
        return Stats.of(net, a, b, gross, turnover, linearCost, impactCost, borrowCost);
    }

    public Stats stats() {
        return stats(start, end);
    }

    public void writeCsv(Path file) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(file))) {
            w.println("date,gross,net,linear_cost,impact_cost,borrow_cost,turnover,long,short,n_long,n_short,delisted");
            for (int k = 0; k < days; k++) {
                w.printf("%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.9f,%.9f,%d,%d,%d%n",
                        panel.dates[start + k], gross[k], net[k], linearCost[k], impactCost[k], borrowCost[k],
                        turnover[k], longExposure[k], shortExposure[k], nLong[k], nShort[k], delisted[k]);
            }
        }
    }

    public record Stats(int days, double annReturn, double annVol, double sharpe, double maxDrawdown,
                        double grossSharpe, double avgTurnover, double annLinearCost, double annImpactCost,
                        double annBorrowCost) {

        static Stats of(double[] net, int a, int b, double[] gross, double[] to,
                        double[] lin, double[] imp, double[] bor) {
            int n = b - a;
            if (n < 2) return new Stats(n, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                    Double.NaN, Double.NaN, Double.NaN, Double.NaN);
            double[] m = meanStd(net, a, b), g = meanStd(gross, a, b);
            double eq = 0, peak = 0, dd = 0, t = 0, l = 0, im = 0, bo = 0;
            for (int k = a; k < b; k++) {
                eq += net[k];
                peak = Math.max(peak, eq);
                dd = Math.min(dd, eq - peak);
                t += to[k]; l += lin[k]; im += imp[k]; bo += bor[k];
            }
            double s252 = Math.sqrt(252);
            return new Stats(n, m[0] * 252, m[1] * s252, m[1] > 0 ? s252 * m[0] / m[1] : Double.NaN, dd,
                    g[1] > 0 ? s252 * g[0] / g[1] : Double.NaN, t / n, l / n * 252, im / n * 252, bo / n * 252);
        }

        private static double[] meanStd(double[] x, int a, int b) {
            double s = 0;
            for (int k = a; k < b; k++) s += x[k];
            double mu = s / (b - a), v = 0;
            for (int k = a; k < b; k++) v += (x[k] - mu) * (x[k] - mu);
            return new double[]{mu, Math.sqrt(v / (b - a - 1))};
        }

        @Override
        public String toString() {
            return String.format("days=%d sharpe=%.3f grossSharpe=%.3f annRet=%.4f annVol=%.4f maxDD=%.4f "
                            + "turnover=%.4f costs/yr(lin=%.4f impact=%.4f borrow=%.4f)",
                    days, sharpe, grossSharpe, annReturn, annVol, maxDrawdown, avgTurnover,
                    annLinearCost, annImpactCost, annBorrowCost);
        }
    }
}
