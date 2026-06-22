import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from agent.tools.base import BaseTool, ToolResult


class YahooFinanceTool(BaseTool):
    """Fetch real-time stock data, historical prices, and fundamentals from Yahoo Finance."""

    def __init__(self):
        super().__init__(
            name="yahoo_finance",
            description="Fetch live stock prices, historical returns, and fundamentals from Yahoo Finance"
        )

    def _run(self, ticker: str, operation: str = "quote", period: str = "1y") -> dict:
        stock = yf.Ticker(ticker.upper().strip())

        # Validate ticker exists — yfinance returns empty/error dict for invalid tickers
        info = stock.info
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            quick_hist = stock.history(period="5d")
            if quick_hist.empty:
                raise ValueError(f"Invalid ticker or no data available: {ticker}")

        if operation == "quote":
            return {
                "ticker": ticker.upper(),
                "name": info.get("longName", "N/A"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "currency": info.get("currency", "USD"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
            }

        elif operation == "history":
            hist = stock.history(period=period)
            if hist.empty:
                raise ValueError(f"No historical data for {ticker}")
            return {
                "ticker": ticker.upper(),
                "period": period,
                "latest_close": round(hist["Close"].iloc[-1], 2),
                "period_high": round(hist["High"].max(), 2),
                "period_low": round(hist["Low"].min(), 2),
                "avg_volume": int(hist["Volume"].mean()),
                "data_points": len(hist),
            }

        elif operation == "returns":
            hist = stock.history(period=period)
            if hist.empty or len(hist) < 2:
                raise ValueError(f"Insufficient data for {ticker}")
            daily_returns = hist["Close"].pct_change().dropna()
            total_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            volatility = daily_returns.std() * (252 ** 0.5) * 100
            return {
                "ticker": ticker.upper(),
                "period": period,
                "total_return_pct": round(total_return, 2),
                "annualized_volatility_pct": round(volatility, 2),
                "daily_returns_sample": [round(r, 4) for r in daily_returns.tail(5).tolist()],
            }

        elif operation == "fundamentals":
            return {
                "ticker": ticker.upper(),
                "revenue": info.get("totalRevenue"),
                "profit_margins": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "return_on_equity": info.get("returnOnEquity"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
            }

        else:
            raise ValueError(f"Unknown operation: {operation}")


# Singleton instance
yahoo_finance_tool = YahooFinanceTool()