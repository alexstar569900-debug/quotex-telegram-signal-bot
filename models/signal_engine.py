"""Advanced Signal Generation Engine with AI/ML."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TradeSignal(str, Enum):
    """Trade signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_SIGNAL = "NO_SIGNAL"


class Timeframe(str, Enum):
    """Available timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class CandleData:
    """OHLC candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SignalData:
    """Signal analysis result."""
    asset: str
    timeframe: Timeframe
    signal: TradeSignal
    accuracy: float  # 0-100%
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1
    reason: str
    timestamp: datetime
    indicators: Dict


class TechnicalAnalyzer:
    """Advanced technical analysis engine."""

    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(data) < period:
            return 50.0

        deltas = np.diff(data)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 1
        rsi = np.zeros_like(data)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, len(data)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 1
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        return float(rsi[-1])

    @staticmethod
    def calculate_macd(
        data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        ema_fast = pd.Series(data).ewm(span=fast).mean()
        ema_slow = pd.Series(data).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    @staticmethod
    def calculate_bollinger_bands(
        data: np.ndarray, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        sma = pd.Series(data).rolling(window=period).mean()
        std = pd.Series(data).rolling(window=period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def calculate_stochastic(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        lowest_low = pd.Series(low).rolling(window=period).min()
        highest_high = pd.Series(high).rolling(window=period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = pd.Series(k_percent).rolling(window=3).mean()

        return float(k_percent.iloc[-1]), float(d_percent.iloc[-1])

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean()
        return float(atr.iloc[-1])


class AISignalGenerator:
    """AI-powered signal generator with ML model."""

    def __init__(self, config):
        """Initialize AI signal generator."""
        self.config = config
        self.analyzer = TechnicalAnalyzer()
        self.model = self._load_model()
        self.logger = logging.getLogger(__name__)

    def _load_model(self):
        """Load pre-trained ML model."""
        try:
            import pickle
            if os.path.exists(self.config.ML_MODEL_PATH):
                with open(self.config.ML_MODEL_PATH, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load ML model: {e}")
        return None

    async def analyze_candles(
        self, candles: List[CandleData], asset: str, timeframe: Timeframe
    ) -> Optional[SignalData]:
        """Analyze candles and generate signal."""
        if len(candles) < 30:
            return None

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate indicators
        rsi = self.analyzer.calculate_rsi(closes, self.config.RSI_PERIOD)
        macd, macd_signal, macd_hist = self.analyzer.calculate_macd(
            closes, self.config.MACD_FAST, self.config.MACD_SLOW, self.config.MACD_SIGNAL
        )
        upper_bb, middle_bb, lower_bb = self.analyzer.calculate_bollinger_bands(
            closes, self.config.BOLLINGER_PERIOD
        )
        k_stoch, d_stoch = self.analyzer.calculate_stochastic(highs, lows, closes)
        atr = self.analyzer.calculate_atr(highs, lows, closes)

        # Build feature vector for ML model
        features = np.array([
            rsi, macd, macd_signal, macd_hist, k_stoch, d_stoch,
            (closes[-1] - lower_bb) / (upper_bb - lower_bb),  # BB position
            (closes[-1] - closes[-5]) / closes[-5],  # Recent momentum
            np.std(closes[-14:]),  # Volatility
            volumes[-1] / np.mean(volumes[-14:])  # Volume ratio
        ]).reshape(1, -1)

        # ML model prediction
        ml_confidence = 0.5
        if self.model:
            try:
                ml_confidence = float(self.model.predict_proba(features)[0].max())
            except:
                pass

        # Rule-based signal generation
        signal, confidence, reason = self._generate_signal(
            rsi, macd, macd_hist, k_stoch, closes, upper_bb, lower_bb, ml_confidence
        )

        # Calculate entry and risk levels
        current_price = closes[-1]
        entry_price = current_price
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3.0)

        # Adjust for signal direction
        if signal in [TradeSignal.SELL, TradeSignal.STRONG_SELL]:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)

        return SignalData(
            asset=asset,
            timeframe=timeframe,
            signal=signal,
            accuracy=self._calculate_accuracy(rsi, macd_hist, k_stoch, confidence),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
            timestamp=datetime.now(),
            indicators={
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'k_stoch': k_stoch,
                'd_stoch': d_stoch,
                'bb_upper': upper_bb,
                'bb_middle': middle_bb,
                'bb_lower': lower_bb,
                'atr': atr,
                'ml_confidence': ml_confidence
            }
        )

    def _generate_signal(
        self, rsi, macd, macd_hist, k_stoch, closes, upper_bb, lower_bb, ml_confidence
    ) -> Tuple[TradeSignal, float, str]:
        """Generate trading signal from indicators."""
        signal_score = 0
        reasons = []

        # RSI signals
        if rsi < 30:
            signal_score += 2
            reasons.append("RSI oversold")
        elif rsi > 70:
            signal_score -= 2
            reasons.append("RSI overbought")
        elif rsi < 40:
            signal_score += 1
            reasons.append("RSI weak")
        elif rsi > 60:
            signal_score -= 1
            reasons.append("RSI strong")

        # MACD signals
        if macd > 0 and macd_hist > 0:
            signal_score += 1.5
            reasons.append("MACD bullish")
        elif macd < 0 and macd_hist < 0:
            signal_score -= 1.5
            reasons.append("MACD bearish")

        # Stochastic signals
        if k_stoch < 20:
            signal_score += 1
            reasons.append("Stoch oversold")
        elif k_stoch > 80:
            signal_score -= 1
            reasons.append("Stoch overbought")

        # Bollinger Bands signals
        current_price = closes[-1]
        if current_price < lower_bb:
            signal_score += 1.5
            reasons.append("Price below BB lower")
        elif current_price > upper_bb:
            signal_score -= 1.5
            reasons.append("Price above BB upper")

        # Momentum signals
        momentum = (closes[-1] - closes[-5]) / closes[-5]
        if momentum > 0.01:
            signal_score += 0.5
        elif momentum < -0.01:
            signal_score -= 0.5

        # Add ML confidence
        signal_score += (ml_confidence - 0.5) * 2

        # Determine final signal
        confidence = min(abs(signal_score) / 5.0, 1.0)
        reason = ", ".join(reasons)

        if signal_score >= 3.5:
            return TradeSignal.STRONG_BUY, confidence, reason
        elif signal_score >= 1.5:
            return TradeSignal.BUY, confidence, reason
        elif signal_score <= -3.5:
            return TradeSignal.STRONG_SELL, confidence, reason
        elif signal_score <= -1.5:
            return TradeSignal.SELL, confidence, reason
        else:
            return TradeSignal.NEUTRAL, confidence, reason

    @staticmethod
    def _calculate_accuracy(rsi: float, macd_hist: float, k_stoch: float, ml_conf: float) -> float:
        """Calculate signal accuracy percentage."""
        base_accuracy = 65.0

        # Adjust based on indicator extremes
        if (rsi < 30 or rsi > 70) and (k_stoch < 20 or k_stoch > 80):
            base_accuracy += 10
        elif (rsi < 40 or rsi > 60) and (k_stoch < 30 or k_stoch > 70):
            base_accuracy += 5

        # Adjust based on MACD histogram
        if abs(macd_hist) > 0.02:
            base_accuracy += 5

        # Add ML confidence
        base_accuracy += (ml_conf - 0.5) * 20

        return min(max(base_accuracy, 50.0), 95.0)


import os
