package io.github.piyushr27.quantsim;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Supplier;

/**
 * Runs many independent simulations in parallel. The {@link Panel} is shared
 * read-only; each task builds its own {@link Simulator} (private scratch state),
 * so no locking is needed.
 */
public final class Sweep {
    private Sweep() {}

    public record Job(String name, Supplier<double[]> signal, SimConfig config) {}

    public record Outcome(String name, SimConfig config, SimResult result) {}

    public static List<Outcome> run(Panel panel, List<Job> jobs, int threads, int start, int end)
            throws InterruptedException, ExecutionException {
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Outcome>> fs = new ArrayList<>(jobs.size());
            for (Job j : jobs) {
                fs.add(ex.submit(() -> new Outcome(j.name(), j.config(),
                        new Simulator(panel, j.config()).run(j.signal().get(), start, end))));
            }
            List<Outcome> out = new ArrayList<>(jobs.size());
            for (Future<Outcome> f : fs) out.add(f.get());
            return out;
        } finally {
            ex.shutdown();
        }
    }
}
