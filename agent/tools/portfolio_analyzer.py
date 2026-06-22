import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from agent.tools.base import BaseTool, ToolResult


def _to_native(val):
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    return val


class PortfolioAnalyzerTool(BaseTool):
    """Calculate Sharpe Ratio, volatility, and max drawdown for a multi-asset portfolio."""

    def __init__(self):
        super().__init__(
            name="portfolio_analyzer",
            description="Calculate Sharpe ratio, volatility, and max drawdown for a multi-asset portfolio"
        )

    def _run(self, tickers: str, weights: str = None, period: str = "1y", risk_free_rate: float = 0.05) -> dict:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        if not ticker_list or len(ticker_list) < 2:
            raise ValueError("Need at least 2 tickers")

        # Parse weights
        if weights:
            weight_list = [float(w.strip()) for w in weights.split(",")]
            if len(weight_list) != len(ticker_list):
                raise ValueError("Weights count must match tickers count")
            if abs(sum(weight_list) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
        else:
            weight_list = [1.0 / len(ticker_list)] * len(ticker_list)

        # Download historical data
        data = yf.download(ticker_list, period=period, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("Failed to fetch historical data")

        # Handle single vs multi-ticker column structure
        if len(ticker_list) == 1:
            closes = data["Close"].to_frame(ticker_list[0])
        else:
            closes = data["Close"]

        # Validate all tickers have data (no NaN columns)
        missing_tickers = [t for t in ticker_list if closes[t].isna().all()]
        if missing_tickers:
            raise ValueError(f"No data available for tickers: {', '.join(missing_tickers)}")

        # Drop any rows with NaN (partial trading days)
        closes = closes.dropna()

        # Calculate daily returns
        returns = closes.pct_change().dropna()

        # Validate we have enough data points
        if len(returns) < 5:
            raise ValueError(f"Insufficient data points ({len(returns)}) for analysis. Need at least 5.")

        # Portfolio daily returns
        weights_arr = np.array(weight_list)
        portfolio_returns = returns.dot(weights_arr)

        # Sharpe Ratio
        excess_returns = portfolio_returns - (risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Annualized metrics
        annualized_return = portfolio_returns.mean() * 252 * 100
        annualized_volatility = portfolio_returns.std() * np.sqrt(252) * 100

        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Individual asset contributions
        asset_metrics = {}
        for i, ticker in enumerate(ticker_list):
            asset_ret = returns[ticker].mean() * 252 * 100
            asset_vol = returns[ticker].std() * np.sqrt(252) * 100
            asset_metrics[ticker] = {
                "weight": round(weight_list[i], 2),
                "annualized_return_pct": round(asset_ret, 2),
                "annualized_volatility_pct": round(asset_vol, 2),
            }

        return _to_native({
            "portfolio": {
                "tickers": ticker_list,
                "weights": [round(w, 2) for w in weight_list],
                "period": period,
                "risk_free_rate": risk_free_rate,
                "sharpe_ratio": round(sharpe_ratio, 3),
                "annualized_return_pct": round(annualized_return, 2),
                "annualized_volatility_pct": round(annualized_volatility, 2),
                "max_drawdown_pct": round(max_drawdown, 2),
            },
            "assets": asset_metrics,
            "interpretation": self._interpret(sharpe_ratio, max_drawdown),
        })

    def _interpret(self, sharpe: float, max_dd: float) -> str:
        if sharpe > 1.5:
            quality = "excellent risk-adjusted returns"
        elif sharpe > 1.0:
            quality = "good risk-adjusted returns"
        elif sharpe > 0.5:
            quality = "moderate risk-adjusted returns"
        else:
            quality = "poor risk-adjusted returns — high risk for the return"

        dd_warning = " Significant drawdown risk." if max_dd < -20 else ""
        return f"Sharpe ratio of {sharpe:.2f} indicates {quality}.{dd_warning}"


# Singleton instance
portfolio_analyzer_tool = PortfolioAnalyzerTool()