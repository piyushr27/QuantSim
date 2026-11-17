package io.github.piyushr27.quantsim;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Throughput benchmark on a synthetic panel (default 12 years × 3,000 names,
 * realistic costs, sector-neutral alpha mode). Reports the median of
 * {@code reps} single-thread runs after JIT warm-up, then the throughput of a
 * parallel sweep of independent simulations.
 */
final class Bench {
    private Bench() {}

    static void run(Map<String, String> a) throws Exception {
        int T = Integer.parseInt(a.getOrDefault("T", "3024"));
        int N = Integer.parseInt(a.getOrDefault("N", "3000"));
        int reps = Integer.parseInt(a.getOrDefault("reps", "7"));
        int threads = Integer.parseInt(a.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        Panel p = Synthetic.panel(T, N, 11, 42);
        double[] sig = Synthetic.signal(p, 0.02, 7);
        SimConfig cfg = SimConfig.alpha(CostModel.realistic());

        for (int i = 0; i < 3; i++) new Simulator(p, cfg).run(sig);   // JIT warm-up
        double[] ms = new double[reps];
        SimResult r = null;
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            r = new Simulator(p, cfg).run(sig);
            ms[i] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(ms);
        double med = ms[reps / 2];
        double cellsPerSec = (double) T * N / (med / 1e3);

        int jobs = threads * 4;
        List<Sweep.Job> js = new ArrayList<>();
        for (int j = 0; j < jobs; j++) js.add(new Sweep.Job("j" + j, () -> sig, cfg));
        Sweep.run(p, js, threads, 0, T);  // warm-up
        long t0 = System.nanoTime();
        Sweep.run(p, js, threads, 0, T);
        double sweepSec = (System.nanoTime() - t0) / 1e9;

        String line = String.format("T=%d N=%d single-thread median %.1f ms (min %.1f, max %.1f) = %.1f M stock-days/s; "
                        + "parallel sweep: %d sims on %d threads in %.2f s = %.1f sims/s%n",
                T, N, med, ms[0], ms[reps - 1], cellsPerSec / 1e6, jobs, threads, sweepSec, jobs / sweepSec);
        System.out.print(line);
        System.out.println("sanity: " + r.stats());
        if (a.containsKey("out")) {
            Path o = Path.of(a.get("out"));
            Files.createDirectories(o.toAbsolutePath().getParent());
            boolean exists = Files.exists(o);
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(o,
                    java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND))) {
                if (!exists) w.println("T,N,median_ms,min_ms,max_ms,mcells_per_s,sweep_sims,threads,sweep_s,sims_per_s");
                w.printf("%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.3f,%.2f%n", T, N, med, ms[0], ms[reps - 1],
                        cellsPerSec / 1e6, jobs, threads, sweepSec, jobs / sweepSec);
            }
        }
    }
}
