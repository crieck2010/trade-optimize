# Changelog — trade-optimize

## v0.1.0 — 2026-09-24

Initial release: Markowitz portfolio optimization for the trade-suite.

**Numerics**
- Pure-Python linear algebra: Cholesky (with PD check), solve, Gauss-Jordan
  inverse, power-iteration max eigenvalue. Stdlib-only.

**Estimators**
- Sample mean, grand-mean shrinkage, sample covariance, diagonal-shrinkage
  covariance (default δ = 0.2), EWMA covariance (RiskMetrics λ = 0.94).

**Optimizers** (long-only, optional per-asset weight cap)
- Global minimum variance (γ = 0).
- Maximum Sharpe: max-Sharpe point on the swept frontier.
- Efficient frontier via log-spaced risk-aversion (γ) sweep, auto-detected top end.
- Risk parity via damped fixed-point iteration toward equal risk contribution.
- 1/N equal-weight baseline.
- Single projected-gradient QP core with exact capped-simplex projection.

**Analytics**
- Portfolio stats (per-period + annualized return/vol/Sharpe), risk
  contributions, diversification ratio, Herfindahl concentration.

**Rebalancing**
- `trades_to_target`: target weights → buy/sell order dicts vs current
  holdings; liquidates dropped names; skips dust.

**Interop**
- `to_agent_allocation` → trade-agents PM allocator; `to_backtest_weights`
  → trade-backtest schedule block; `to_paper_orders` → trade-paper order
  intents; `portfolio_vol_for_risk` + risk contributions → trade-risk.
- `returns_from_dict_bars`: duck-typed bar ingestion, no sibling imports.

**CLI & packaging**
- `trade-optimize optimize|frontier|allocate|license|update-check`;
  synthetic demo universe or wide-CSV input.
- Stdlib-only, MIT licensed, 31 tests.
