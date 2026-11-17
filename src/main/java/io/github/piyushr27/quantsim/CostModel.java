package io.github.piyushr27.quantsim;

/**
 * Transaction-cost model applied to every trade of {@code |q|} dollars in a
 * name with trailing average dollar volume {@code adv} and daily volatility
 * {@code sigma}:
 *
 * <pre>
 *   commission = |q| * commissionBps * 1e-4
 *   spread     = |q| * halfSpreadBps * 1e-4            (pay half the bid-ask)
 *   impact     = |q| * impactCoef * sigma * sqrt(|q| / adv)   (square-root law)
 *   borrow     = |short $| * borrowBpsAnnual * 1e-4 / 252   (charged daily)
 * </pre>
 *
 * {@code maxParticipation} caps each day's trade at that fraction of ADV
 * ({@code <= 0} disables the cap); unfilled size carries to the next day.
 */
public record CostModel(double commissionBps, double halfSpreadBps, double impactCoef,
                        double borrowBpsAnnual, double maxParticipation) {

    public static final CostModel ZERO = new CostModel(0, 0, 0, 0, 0);

    /** Defaults for liquid US large caps. */
    public static CostModel realistic() {
        return new CostModel(0.5, 2.0, 0.1, 50.0, 0.05);
    }

    public static CostModel flat(double bpsOneWay) {
        return new CostModel(bpsOneWay, 0, 0, 0, 0);
    }

    double linearRate() {
        return (commissionBps + halfSpreadBps) * 1e-4;
    }

    double impactCost(double absQ, double adv, double sigma) {
        if (impactCoef == 0 || !(adv > 0) || !Double.isFinite(sigma)) return 0.0;
        return absQ * impactCoef * sigma * Math.sqrt(absQ / adv);
    }
}
