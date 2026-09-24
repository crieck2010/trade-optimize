"""End-to-end example: estimate, optimize, analyze, rebalance."""

from trade_optimize import estimates, optimize, analytics, trades_to_target
from trade_optimize.adapters import to_agent_allocation

# synthetic correlated returns: 4 assets, 252 periods
import random
rng = random.Random(7)
rets = [[rng.gauss(0.0005 * (i + 1), 0.01 + 0.004 * i) for i in range(4)]
        for _ in range(252)]
symbols = ["AAA", "BBB", "CCC", "DDD"]

mu = estimates.shrink_mean(rets, delta=0.3)
sigma = estimates.shrink_covariance(rets, delta=0.2)

for method in ("min_variance", "max_sharpe", "risk_parity", "equal_weight"):
    w = optimize(sigma, mu, method=method, max_weight=0.5)
    st = analytics.portfolio_stats(w, mu, sigma)
    print(f"{method:>12}: " +
          " ".join(f"{s} {x:.0%}" for s, x in zip(symbols, w)) +
          f"  | vol {st['volatility_ann']:.1%}  sharpe {st['sharpe_ann']:.2f}")

# agent-ready allocation for the trade-agents PM
alloc = to_agent_allocation(symbols, rets, method="max_sharpe")
print("\nPM allocation:", {k: round(v, 3) for k, v in alloc["weights"].items()})

# rebalance $100k from a starting position into the max-Sharpe weights
w = optimize(sigma, mu, method="max_sharpe", max_weight=0.5)
orders = trades_to_target(
    {"AAA": 100.0}, dict(zip(symbols, w)),
    {"AAA": 50.0, "BBB": 75.0, "CCC": 30.0, "DDD": 120.0},
    capital=100_000.0)
print(f"\n{len(orders)} orders to reach target:")
for o in orders:
    print(f"  {o['side']:>4} {o['quantity']:8.2f} {o['symbol']}")
