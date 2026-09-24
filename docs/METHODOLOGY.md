# Methodology — trade-optimize v0.1.0

The math this engine implements, and where it can mislead you.

## 1. The Markowitz problem

Given expected returns μ and covariance Σ, every optimizer here solves a
version of:

    min  ½·w′Σw − γ·μ′w     s.t.  Σw = 1,  0 ≤ w ≤ cap

- **γ = 0** → the global **minimum-variance** portfolio.
- **γ > 0** → a risk-aversion tradeoff; sweeping γ traces the **efficient
  frontier**. The **maximum-Sharpe** (tangency) portfolio is the frontier
  point with the highest (μ′w − rf)/σₚ.

The solver is projected gradient descent: step size 1/L where L is the
largest eigenvalue of Σ (power iteration), projected each step onto the
capped simplex. The projection is exact — bisection on the Lagrange
multiplier λ with xᵢ(λ) = clip(yᵢ − λ, 0, cap), solving Σx = 1.

The γ grid is log-spaced over four decades up to an auto-detected γ_max,
where doubling γ stops changing the solution (the frontier's concentrated
top end).

## 2. Estimation: the part that actually matters

Optimization is the easy half; estimation error dominates real portfolio
outcomes. v0.1.0 ships deliberately defensive defaults:

- **Covariance**: sample covariance shrunk toward its diagonal
  (Σ = (1−δ)S + δ·diag(S), default δ = 0.2). This kills the noisiest
  off-diagonal correlations while keeping each asset's variance — a simple,
  honest stand-in for Ledoit-Wolf.
- **EWMA covariance** (RiskMetrics λ = 0.94) for short-horizon risk that
  should react to recent volatility.
- **Means**: raw sample means, or shrunk toward the grand mean
  (James-Stein flavour, δ = 0.3) so the noisiest estimates don't dominate.

Cholesky decomposition doubles as a positive-definiteness check: a
non-PD covariance raises instead of silently producing nonsense.

## 3. Risk parity

Equal risk contribution: wᵢ(Σw)ᵢ = σₚ²/n for all i. Solved by damped
fixed-point iteration (wᵢ ← wᵢ·σₚ²/(n(Σw)ᵢ), renormalized to the capped
simplex each step) — simpler than Spinu's cyclic coordinate descent but
convergent on well-conditioned covariances.

## 4. Analytics

- **Risk contributions** RCᵢ = wᵢ(Σw)ᵢ: sum to portfolio variance; the
  shares sum to 1. The basis for concentration limits.
- **Diversification ratio**: weighted-average asset vol / portfolio vol
  (≥ 1; higher = more diversification benefit).
- **Herfindahl**: Σwᵢ², from 1/N (fully spread) to 1 (single asset).

## 5. Rebalancing

`trades_to_target` converts target weights to share quantities against
current holdings and prices, emitting plain {symbol, side, quantity}
dicts — the shape trade-paper's approval queue expects. Names that fell
out of the target are liquidated; dust below `min_qty` is skipped.

## 6. Known limitations

- **Estimation error is the dominant risk.** Shrinkage helps; it does not
  fix garbage inputs. Means are far noisier than covariances — treat
  max-Sharpe weights built on short samples with skepticism.
- **No transaction costs in the optimizer.** Costs enter at rebalance time
  (thresholds, min quantities), not in the objective.
- **Single-period, myopic.** No multi-period planning, no regime
  conditioning, no turnover penalty in the objective — those are future
  extensions.
- **Long-only only.** No 130/30, no market-neutral formulations yet.
- **The frontier grid is finite** (default 25–60 γ points); the reported
  max-Sharpe point is the best grid point, not an analytic optimum.
- **Risk parity is a fixed-point iteration**, not the full Spinu CCD;
  pathological covariances can stall it (it returns its best effort
  deterministically rather than raising).
