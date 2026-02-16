"""Configuration validation and description utilities."""

from __future__ import annotations

from marginbt.engine import BacktestEngine


def describe_engine(engine: BacktestEngine) -> str:
    """Return a human-readable summary of engine settings."""
    cfg = engine.config
    lines = [
        "+--- Engine Configuration ---",
        f"| init_cash:           {cfg.init_cash}",
        f"| fees (per side):     {cfg.fees}",
        f"| slippage:            {cfg.slippage}",
        f"| freq:                {cfg.freq}",
        f"| leverage:            {cfg.leverage}x",
        f"| maint. margin rate:  {cfg.maintenance_margin_rate}",
        f"| daily loss limit:    {cfg.daily_loss_limit_pct*100:.2f}%",
        f"| kill switch DD:      {cfg.kill_switch_drawdown_pct*100:.2f}%",
        f"| gap SL policy:       {cfg.gap_sl_policy}",
        f"| conflict policy:     {cfg.same_bar_conflict_policy}",
        "+----------------------------",
    ]
    return "\n".join(lines)


__all__ = ["describe_engine"]
