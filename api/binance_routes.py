import requests
from typing import List, Dict, Optional


class BinanceAPI:
    BASE_URL = "https://api.binance.com"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1m") -> None:
        self.symbol = symbol
        self.interval = interval

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Внутрішній метод для GET-запитів"""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_price(self) -> float:
        """Отримання поточної ціни"""
        data = self._get("/api/v3/ticker/price", {"symbol": self.symbol})
        return float(data["price"])

    def get_order_book(self, limit: int = 10) -> Dict[str, List[List[str]]]:
        """Отримання глибини ринку (order book)"""
        return self._get("/api/v3/depth", {"symbol": self.symbol, "limit": limit})

    def get_klines(self, limit: int = 500) -> dict:
        """
        Отримання історичних свічок (OHLCV)

        Returns: список списків: [
            [open_time, open, high, low, close, volume, close_time, ...], ...
        ]
        """
        return self._get("/api/v3/klines", {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        })

    def get_recent_trades(self, limit: int = 100) -> dict:
        """Отримання останніх трейдів"""
        return self._get("/api/v3/trades", {
            "symbol": self.symbol,
            "limit": limit
        })

    def get_exchange_info(self) -> Dict:
        """Отримання загальної інформації про біржу"""
        return self._get("/api/v3/exchangeInfo")

    def set_symbol(self, symbol: str) -> None:
        """Оновлення торгової пари"""
        self.symbol = symbol

    def set_interval(self, interval: str) -> None:
        """Оновлення таймфрейму"""
        self.interval = interval
