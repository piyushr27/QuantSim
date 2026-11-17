package io.github.piyushr27.quantsim;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Command line entry point.
 *
 * <pre>
 * sim   --panel DIR --signal FILE [--mode alpha|weights] [--costs zero|realistic|flat:BPS] [--book USD]
 *       [--delay 1] [--neut sector|market|none] [--delist R] [--from DATE] [--to DATE] [--out CSV]
 * sweep --panel DIR --signals DIR [--threads K] [--split DATE] [--out CSV]
 * bench [--T 3024] [--N 3000] [--reps 5] [--threads K] [--out CSV]
 * </pre>
 */
public final class Main {
    public static void main(String[] argv) throws Exception {
        if (argv.length == 0) {
            System.err.println("usage: quantsim sim|sweep|bench [options]");
            System.exit(2);
        }
        Map<String, String> a = parse(argv);
        switch (argv[0]) {
            case "sim" -> sim(a);
            case "sweep" -> sweep(a);
            case "bench" -> Bench.run(a);
            default -> throw new IllegalArgumentException("unknown command " + argv[0]);
        }
    }

    static Map<String, String> parse(String[] argv) {
        Map<String, String> m = new HashMap<>();
        for (int i = 1; i < argv.length; i++) {
            if (!argv[i].startsWith("--")) throw new IllegalArgumentException("bad arg " + argv[i]);
            m.put(argv[i].substring(2), i + 1 < argv.length && !argv[i + 1].startsWith("--") ? argv[++i] : "true");
        }
        return m;
    }

    static CostModel costs(String s) {
        if (s == null || s.equals("realistic")) return CostModel.realistic();
        if (s.equals("zero")) return CostModel.ZERO;
        if (s.startsWith("flat:")) return CostModel.flat(Double.parseDouble(s.substring(5)));
        throw new IllegalArgumentException("costs: zero|realistic|flat:BPS");
    }

    static SimConfig config(Map<String, String> a) {
        CostModel c = costs(a.get("costs"));
        SimConfig cfg = "weights".equals(a.get("mode")) ? SimConfig.weights(c) : SimConfig.alpha(c);
        if (a.containsKey("delay")) cfg = cfg.withDelay(Integer.parseInt(a.get("delay")));
        if (a.containsKey("neut")) cfg = cfg.withNeutralization(SimConfig.Neutralization.valueOf(a.get("neut").toUpperCase()));
        if (a.containsKey("delist")) cfg = cfg.withDelistReturn(Double.parseDouble(a.get("delist")));
        if (a.containsKey("maxw")) cfg = cfg.withMaxWeight(Double.parseDouble(a.get("maxw")));
        if (a.containsKey("book")) cfg = new SimConfig(cfg.mode(), cfg.delay(), Double.parseDouble(a.get("book")),
                cfg.neutralization(), cfg.maxWeight(), cfg.delistReturn(), cfg.costs());
        return cfg;
    }

    static void sim(Map<String, String> a) throws IOException {
        long t0 = System.nanoTime();
        Panel p = PanelIO.load(Path.of(a.get("panel")));
        double[] sig = PanelIO.readMatrix(Path.of(a.get("signal")), p.T, p.N);
        long t1 = System.nanoTime();
        int from = a.containsKey("from") ? p.indexOf(LocalDate.parse(a.get("from"))) : 0;
        int to = a.containsKey("to") ? p.indexOf(LocalDate.parse(a.get("to")).plusDays(1)) : p.T;
        SimConfig cfg = config(a);
        SimResult r = new Simulator(p, cfg).run(sig, from, to);
        long t2 = System.nanoTime();
        System.out.printf("panel %d days x %d names; load %.0f ms, simulate %.1f ms%n",
                p.T, p.N, (t1 - t0) / 1e6, (t2 - t1) / 1e6);
        int reps = Integer.parseInt(a.getOrDefault("reps", "0"));
        if (reps > 0) {  // steady-state timing after JIT warm-up
            for (int i = 0; i < 3; i++) new Simulator(p, cfg).run(sig, from, to);
            double[] ms = new double[reps];
            for (int i = 0; i < reps; i++) {
                long s0 = System.nanoTime();
                new Simulator(p, cfg).run(sig, from, to);
                ms[i] = (System.nanoTime() - s0) / 1e6;
            }
            java.util.Arrays.sort(ms);
            System.out.printf("warm median %.2f ms over %d reps%n", ms[reps / 2], reps);
        }
        System.out.println(cfg);
        System.out.println(r.stats());
        if (a.containsKey("out")) r.writeCsv(Path.of(a.get("out")));
    }

    /** Every signal file × {zero, flat 5 bps, realistic} costs, in parallel. */
    static void sweep(Map<String, String> a) throws Exception {
        Panel p = PanelIO.load(Path.of(a.get("panel")));
        int threads = Integer.parseInt(a.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        int split = p.indexOf(LocalDate.parse(a.getOrDefault("split", "2019-01-01")));
        List<Path> files;
        try (Stream<Path> s = Files.list(Path.of(a.get("signals")))) {
            files = s.filter(f -> f.toString().endsWith(".f32") || f.toString().endsWith(".f64")).sorted().toList();
        }
        Map<String, CostModel> models = new java.util.LinkedHashMap<>();
        models.put("zero", CostModel.ZERO);
        models.put("flat5", CostModel.flat(5));
        models.put("realistic", CostModel.realistic());
        List<Sweep.Job> jobs = new ArrayList<>();
        for (Path f : files) {
            String name = f.getFileName().toString().replaceAll("\\.f(32|64)$", "");
            for (var e : models.entrySet()) {
                SimConfig cfg = SimConfig.alpha(e.getValue());
                jobs.add(new Sweep.Job(name + "|" + e.getKey(), () -> {
                    try {
                        return PanelIO.readMatrix(f, p.T, p.N);
                    } catch (IOException ex) {
                        throw new java.io.UncheckedIOException(ex);
                    }
                }, cfg));
            }
        }
        long t0 = System.nanoTime();
        List<Sweep.Outcome> out = Sweep.run(p, jobs, threads, 0, p.T);
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.printf("%d simulations (%d signals x %d cost models) on %d threads in %.2f s (%.1f sims/s, incl. I/O)%n",
                out.size(), files.size(), models.size(), threads, secs, out.size() / secs);
        Path o = Path.of(a.getOrDefault("out", "results/sweep.csv"));
        Files.createDirectories(o.toAbsolutePath().getParent());
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(o))) {
            w.println("signal,costs,period,sharpe,gross_sharpe,ann_return,ann_vol,max_dd,turnover,lin_cost,impact_cost,borrow_cost");
            for (Sweep.Outcome x : out) {
                String[] nm = x.name().split("\\|");
                for (String per : new String[]{"is", "oos"}) {
                    SimResult.Stats s = per.equals("is") ? x.result().stats(0, split) : x.result().stats(split, p.T);
                    w.printf("%s,%s,%s,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f%n", nm[0], nm[1], per,
                            s.sharpe(), s.grossSharpe(), s.annReturn(), s.annVol(), s.maxDrawdown(),
                            s.avgTurnover(), s.annLinearCost(), s.annImpactCost(), s.annBorrowCost());
                }
            }
        }
        System.out.println("wrote " + o);
    }
}
