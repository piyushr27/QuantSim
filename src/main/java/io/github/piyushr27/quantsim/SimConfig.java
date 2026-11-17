package io.github.piyushr27.quantsim;

/**
 * Simulation settings.
 *
 * @param mode         ALPHA: signal row is a raw alpha, converted to weights by
 *                     demeaning (per {@code neutralization}) and scaling to unit gross.
 *                     WEIGHTS: signal row already holds target weights (fraction of book).
 * @param delay        days between signal date and trade date. With delay 1 a signal
 *                     computed at close t is traded at close t+1 and first earns the
 *                     close(t+1)→close(t+2) return. Delay 0 is look-ahead and rejected.
 * @param bookSize     gross dollar book used to convert weights to dollars.
 * @param maxWeight    per-name |weight| cap in ALPHA mode (≤0 disables).
 * @param delistReturn return applied to a position whose price disappears while held.
 */
public record SimConfig(Mode mode, int delay, double bookSize, Neutralization neutralization,
                        double maxWeight, double delistReturn, CostModel costs) {

    public enum Mode { ALPHA, WEIGHTS }

    public enum Neutralization { NONE, MARKET, SECTOR }

    public SimConfig {
        if (delay < 1) throw new IllegalArgumentException("delay must be >= 1 (delay 0 uses same-close data to trade)");
        if (!(bookSize > 0)) throw new IllegalArgumentException("bookSize must be > 0");
    }

    public static SimConfig alpha(CostModel c) {
        return new SimConfig(Mode.ALPHA, 1, 20e6, Neutralization.SECTOR, 0.0, 0.0, c);
    }

    public static SimConfig weights(CostModel c) {
        return new SimConfig(Mode.WEIGHTS, 1, 20e6, Neutralization.NONE, 0.0, 0.0, c);
    }

    public SimConfig withCosts(CostModel c) {
        return new SimConfig(mode, delay, bookSize, neutralization, maxWeight, delistReturn, c);
    }

    public SimConfig withDelay(int d) {
        return new SimConfig(mode, d, bookSize, neutralization, maxWeight, delistReturn, costs);
    }

    public SimConfig withNeutralization(Neutralization n) {
        return new SimConfig(mode, delay, bookSize, n, maxWeight, delistReturn, costs);
    }

    public SimConfig withDelistReturn(double r) {
        return new SimConfig(mode, delay, bookSize, neutralization, maxWeight, r, costs);
    }

    public SimConfig withMaxWeight(double w) {
        return new SimConfig(mode, delay, bookSize, neutralization, w, delistReturn, costs);
    }
}
