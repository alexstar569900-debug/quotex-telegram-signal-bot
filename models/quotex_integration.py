"""Quotex API Integration Layer."""

import asyncio
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from QuotexAPI import QuotexAPI, TradeDirection, AccountType, AssetType
from .signal_engine import SignalData, TradeSignal, CandleData, Timeframe

logger = logging.getLogger(__name__)


class QuotexTrader:
    """Quotex trading integration."""

    def __init__(self, config):
        """Initialize Quotex trader."""
        self.config = config
        self.api = QuotexAPI(
            ssid=config.QUOTEX_SSID,
            email=config.QUOTEX_EMAIL,
            password=config.QUOTEX_PASSWORD,
            is_demo=config.QUOTEX_DEMO_MODE,
            log_level="INFO"
        )
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        self.active_trades = {}

    async def connect(self) -> bool:
        """Connect to Quotex."""
        try:
            self.logger.info("Connecting to Quotex...")
            await self.api.connect()
            profile = await self.api.login_with_ssid()
            self.is_connected = True
            self.logger.info(f"✓ Connected as: {profile.email}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Quotex."""
        try:
            await self.api.disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from Quotex")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")

    async def get_balance(self) -> Optional[float]:
        """Get current account balance."""
        try:
            balance = await self.api.get_balance()
            return balance.amount
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return None

    async def get_candles(
        self, asset: str, timeframe: Timeframe, limit: int = 100
    ) -> Optional[List[CandleData]]:
        """Fetch historical candles (mock for now)."""
        try:
            # TODO: Implement actual candle fetching from Quotex
            # For now, return mock data for testing
            candles = []
            base_time = datetime.now()
            base_price = 1.0850

            for i in range(limit, 0, -1):
                timestamp = base_time - timedelta(minutes=i)
                change = (i % 5 - 2) * 0.0005
                candles.append(
                    CandleData(
                        timestamp=timestamp,
                        open=base_price + change,
                        high=base_price + change + 0.0010,
                        low=base_price + change - 0.0010,
                        close=base_price + change + 0.0003,
                        volume=100000 + i * 1000
                    )
                )
            return candles
        except Exception as e:
            self.logger.error(f"Failed to fetch candles: {e}")
            return None

    async def place_trade(
        self, signal: SignalData, trade_amount: float
    ) -> Optional[str]:
        """Place trade based on signal."""
        if not self.is_connected:
            self.logger.warning("Not connected to Quotex")
            return None

        try:
            # Validate trade amount
            if trade_amount > self.config.MAX_TRADE_AMOUNT:
                self.logger.warning(f"Trade amount exceeds max: {trade_amount} > {self.config.MAX_TRADE_AMOUNT}")
                return None

            # Determine trade direction
            if signal.signal in [TradeSignal.STRONG_BUY, TradeSignal.BUY]:
                direction = TradeDirection.CALL
            elif signal.signal in [TradeSignal.STRONG_SELL, TradeSignal.SELL]:
                direction = TradeDirection.PUT
            else:
                self.logger.info(f"Neutral signal, skipping trade")
                return None

            # Place trade
            self.logger.info(
                f"Placing {direction.value} trade: {signal.asset} "
                f"${trade_amount} (Expiry: {signal.timeframe.value})"
            )

            trade = await self.api.buy(
                asset=signal.asset,
                amount=trade_amount,
                direction=direction,
                expiry=self._timeframe_to_seconds(signal.timeframe)
            )

            self.active_trades[trade.order_id] = {
                'asset': signal.asset,
                'direction': direction,
                'amount': trade_amount,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'placed_at': datetime.now(),
                'signal_confidence': signal.confidence
            }

            self.logger.info(f"✓ Trade placed: {trade.order_id}")
            return trade.order_id

        except Exception as e:
            self.logger.error(f"Failed to place trade: {e}")
            return None

    async def track_trade(self, order_id: str) -> Optional[dict]:
        """Track trade result."""
        try:
            result = await self.api.wait_for_result(order_id, timeout=600)
            return {
                'order_id': order_id,
                'result': result.result.value if result.result else None,
                'profit': result.profit,
                'close_price': result.close_price,
                'close_time': result.close_time
            }
        except Exception as e:
            self.logger.error(f"Failed to track trade {order_id}: {e}")
            return None

    async def get_open_trades(self) -> List[dict]:
        """Get list of open trades."""
        try:
            trades = await self.api.get_open_trades()
            return [{
                'order_id': t.order_id,
                'asset': t.asset,
                'direction': t.direction.value,
                'amount': t.amount,
                'status': t.status.value
            } for t in trades]
        except Exception as e:
            self.logger.error(f"Failed to get open trades: {e}")
            return []

    async def get_trade_history(self, limit: int = 50) -> List[dict]:
        """Get trade history."""
        try:
            trades = await self.api.get_trade_history(limit=limit)
            return [{
                'order_id': t.order_id,
                'asset': t.asset,
                'direction': t.direction.value,
                'amount': t.amount,
                'result': t.result.value if t.result else None,
                'profit': t.profit,
                'open_time': t.open_time.isoformat() if t.open_time else None,
                'close_time': t.close_time.isoformat() if t.close_time else None
            } for t in trades]
        except Exception as e:
            self.logger.error(f"Failed to get trade history: {e}")
            return []

    @staticmethod
    def _timeframe_to_seconds(timeframe: Timeframe) -> int:
        """Convert timeframe to seconds."""
        mapping = {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.M30: 1800,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400,
            Timeframe.D1: 86400
        }
        return mapping.get(timeframe, 300)
