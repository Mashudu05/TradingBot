# trading_bot.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config_path='config.json'):
        """Initialize the trading bot with configuration"""
        self.config = self.load_config(config_path)
        self.positions = {}
        self.trade_history = []
        self.daily_pl = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.daily_limit_reached = False
        self.allowed_instruments = [
            'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
            'USDCAD', 'NZDUSD', 'US30', 'DE40', 'UK100'
        ]
        
        # Initialize MT5
        if not self.initialize_mt5():
            raise Exception("Failed to initialize MT5")
        
        # Get account info
        self.account_info = mt5.account_info()
        if self.account_info is None:
            raise Exception("Failed to get account info")
        
        logger.info(f"Account Balance: ${self.account_info.balance}")
        logger.info(f"Account Equity: ${self.account_info.equity}")
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        default_config = {
            'default_risk_percent': 20.5,
            'high_volatility_risk_percent': 10,
            'max_daily_loss_percent': 2.0,
            'max_consecutive_losses': 3,
            'min_rr_ratio': 2.0,
            'min_setup_score': 80,
            'timeframes': {
                'analysis': ['H4', 'H1', 'M15', 'M5'],
                'entry': ['M5', 'M1']
            },
            'ema_periods': [20, 50, 200],
            'rsi_period': 14,
            'atr_period': 14,
            'max_spread': 50  # in points
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return default_config
    
    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        mt5_path = r"C:\Program Files\JustMarkets MetaTrader 5\terminal64.exe"
        logger.info(f"Attempting to connect to MT5 at: {mt5_path}")
        if not mt5.initialize(mt5_path):
            logger.error("MT5 initialization failed")
            logger.error("Please ensure MT5 is open and you are logged in")
            return False
        
        if not mt5.terminal_info():
            logger.error("MT5 terminal info not available")
            return False
            
        logger.info("MT5 initialized successfully")
        return True
    
    def get_market_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Fetch market data for a given symbol and timeframe"""
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()
        
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get data for {symbol} {timeframe}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if len(df) == 0:
            return df
        
        # EMAs
        for period in self.config['ema_periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=self.config['atr_period']).mean()
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure and identify key levels"""
        if len(df) < 50:
            return {'structure': 'unclear', 'levels': [], 'trend': 'unclear'}
        
        analysis = {
            'structure': 'unclear',
            'trend': 'unclear',
            'levels': [],
            'support_resistance': [],
            'liquidity_pools': [],
            'break_of_structure': False,
            'change_of_character': False
        }
        
        try:
            # Identify swing highs and lows
            highs = []
            lows = []
            for i in range(20, len(df) - 20):
                if df['high'].iloc[i] > df['high'].iloc[i-5:i].max() and \
                   df['high'].iloc[i] > df['high'].iloc[i+1:i+6].max():
                    highs.append((i, df['high'].iloc[i]))
                if df['low'].iloc[i] < df['low'].iloc[i-5:i].min() and \
                   df['low'].iloc[i] < df['low'].iloc[i+1:i+6].min():
                    lows.append((i, df['low'].iloc[i]))
            
            # Determine structure
            if len(highs) >= 3 and len(lows) >= 3:
                higher_highs = all(highs[i][1] > highs[i-1][1] for i in range(1, len(highs)))
                higher_lows = all(lows[i][1] > lows[i-1][1] for i in range(1, len(lows)))
                lower_highs = all(highs[i][1] < highs[i-1][1] for i in range(1, len(highs)))
                lower_lows = all(lows[i][1] < lows[i-1][1] for i in range(1, len(lows)))
                
                if higher_highs and higher_lows:
                    analysis['trend'] = 'bullish'
                    analysis['structure'] = 'uptrend'
                elif lower_highs and lower_lows:
                    analysis['trend'] = 'bearish'
                    analysis['structure'] = 'downtrend'
                else:
                    analysis['trend'] = 'ranging'
                    analysis['structure'] = 'consolidation'
            
            # Identify support/resistance levels
            sr_levels = []
            for high in highs:
                sr_levels.append(('resistance', high[1]))
            for low in lows:
                sr_levels.append(('support', low[1]))
            analysis['support_resistance'] = sr_levels
            
            # Check for Break of Structure (BOS)
            if len(df) > 100:
                recent_high = df['high'].iloc[-20:].max()
                recent_low = df['low'].iloc[-20:].min()
                prev_high = df['high'].iloc[-40:-20].max()
                prev_low = df['low'].iloc[-40:-20].min()
                
                if recent_high > prev_high:
                    analysis['break_of_structure'] = True
                if recent_low < prev_low:
                    analysis['break_of_structure'] = True
                    
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            
        return analysis
    
    def check_news_risk(self, symbol: str) -> Dict:
        """Check for upcoming major news events"""
        # This is a placeholder - in production, integrate with an economic calendar API
        # For now, we'll return a default "safe" status
        return {
            'news_risk': 'low',
            'events': [],
            'warning': False,
            'message': 'No major news detected (calendar data unavailable)'
        }
    
    def assess_setup_quality(self, symbol: str, df_h4: pd.DataFrame, df_h1: pd.DataFrame, 
                            df_m15: pd.DataFrame, df_m5: pd.DataFrame) -> Dict:
        """Assess trade setup quality using multi-timeframe analysis"""
        # This is a simplified assessment - you would implement all the detailed criteria
        # from your prompt here
        assessment = {
            'score': 0,
            'details': {},
            'decision': 'WAIT',
            'setup_type': None,
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward': 0,
            'confidence': 'low'
        }
        
        # Example: Simplified setup assessment
        # You should expand this with all your specific criteria
        
        # Check timeframe alignment
        h4_analysis = self.analyze_market_structure(df_h4)
        h1_analysis = self.analyze_market_structure(df_h1)
        m15_analysis = self.analyze_market_structure(df_m15)
        
        score = 0
        
        # 1. Higher timeframe alignment (20 points)
        if h4_analysis['trend'] == h1_analysis['trend'] and h4_analysis['trend'] != 'unclear':
            score += 20
        
        # 2. Market structure (20 points)
        if h4_analysis['structure'] != 'unclear' and h4_analysis['structure'] != 'consolidation':
            score += 20
        
        # 3. Key level/zone (15 points)
        # Check if price is near support/resistance
        current_price = df_m5['close'].iloc[-1]
        for level_type, level in h4_analysis['support_resistance'][:5]:  # Check top 5 levels
            if abs(current_price - level) / current_price < 0.005:  # Within 0.5%
                score += 15
                break
        
        # 4. Entry confirmation (15 points)
        # Check for rejection candles or break and retest
        if len(df_m5) > 10:
            last_candle = df_m5.iloc[-1]
            prev_candle = df_m5.iloc[-2]
            if abs(last_candle['close'] - last_candle['open']) > abs(prev_candle['close'] - prev_candle['open']) * 1.5:
                if last_candle['close'] > last_candle['open']:  # Bullish momentum
                    score += 10
                else:
                    score += 5
        
        # 5. Momentum/volume (10 points)
        # Simplified: Check if RSI is trending in the right direction
        if len(df_m5) > 20:
            rsi_trend = df_m5['rsi'].iloc[-1] - df_m5['rsi'].iloc[-5]
            if abs(rsi_trend) > 5:
                score += 10
        
        # 6. Risk-reward (10 points)
        # Calculate potential R:R (simplified)
        if len(df_h4) > 10:
            support_levels = [level for level_type, level in h4_analysis['support_resistance'] if level_type == 'support']
            resistance_levels = [level for level_type, level in h4_analysis['support_resistance'] if level_type == 'resistance']
            
            if support_levels and resistance_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else None
                
                if nearest_support and nearest_resistance:
                    risk = abs(current_price - nearest_support)
                    reward = abs(nearest_resistance - current_price)
                    if risk > 0 and reward / risk >= 2.0:
                        score += 10
                    elif risk > 0 and reward / risk >= 1.5:
                        score += 5
        
        # 7. Market conditions/news risk (10 points)
        news_risk = self.check_news_risk(symbol)
        if not news_risk['warning']:
            score += 10
        
        assessment['score'] = score
        assessment['details'] = {
            'h4_trend': h4_analysis['trend'],
            'h1_trend': h1_analysis['trend'],
            'structure': h4_analysis['structure']
        }
        
        # Determine decision based on score
        if score >= self.config['min_setup_score']:
            assessment['decision'] = 'BUY' if h4_analysis['trend'] == 'bullish' else 'SELL'
            assessment['confidence'] = 'high'
            assessment['setup_type'] = 'Break and Retest'
            
            # Calculate entry levels
            assessment['entry'] = current_price
            assessment['stop_loss'] = current_price * 0.99 if assessment['decision'] == 'BUY' else current_price * 1.01
            assessment['take_profit'] = current_price * 1.02 if assessment['decision'] == 'BUY' else current_price * 0.98
            assessment['risk_reward'] = 2.0
        else:
            assessment['decision'] = 'NO TRADE'
            assessment['confidence'] = 'low'
        
        return assessment
    
    def calculate_position_size(self, symbol: str, entry: float, stop_loss: float, 
                               risk_percent: float) -> float:
        """Calculate position size based on risk parameters"""
        account_equity = self.account_info.equity
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return 0.0
        
        # Calculate risk in account currency
        risk_amount = account_equity * (risk_percent / 100)
        
        # Calculate pip value and position size
        pip_size = 0.0001 if 'USD' in symbol else 0.01  # Simplified
        if symbol == 'XAUUSD':
            pip_size = 0.1
        
        risk_pips = abs(entry - stop_loss) / pip_size
        
        if risk_pips == 0:
            return 0.0
        
        # Calculate position size (lots)
        pip_value = self.get_pip_value(symbol)
        if pip_value == 0:
            return 0.0
        
        position_size = risk_amount / (risk_pips * pip_value)
        
        # Adjust for leverage and minimum/maximum
        position_size = max(position_size, symbol_info.volume_min)
        position_size = min(position_size, symbol_info.volume_max)
        
        # Round to allowed step
        step = symbol_info.volume_step
        position_size = round(position_size / step) * step
        
        return position_size
    
    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0
        
        # Simplified pip value calculation
        if 'USD' in symbol:
            return 1.0 if symbol.startswith('USD') else 10.0 / symbol_info.trade_tick_value
        
        return symbol_info.trade_tick_value
    
    def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on the signal"""
        if signal['decision'] not in ['BUY', 'SELL']:
            logger.info(f"No trade: {signal['decision']}")
            return False
        
        # Check daily limits
        if self.daily_limit_reached:
            logger.warning("Daily limit reached - no trades allowed")
            return False
        
        symbol = signal.get('symbol', 'EURUSD')
        direction = signal['decision']
        entry = signal['entry']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        risk_percent = signal.get('risk_percent', self.config['default_risk_percent'])
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol, entry, stop_loss, risk_percent)
        if position_size == 0:
            logger.error("Invalid position size calculation")
            return False
        
        # Prepare order
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": order_type,
            "price": entry,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Bot Trade {datetime.now().strftime('%Y%m%d')}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return False
        
        logger.info(f"Order executed: {symbol} {direction} {position_size} lots at {entry}")
        logger.info(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        
        # Record trade
        self.positions[result.order] = {
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'open_time': datetime.now(),
            'order_id': result.order
        }
        
        self.trades_today += 1
        
        return True
    
    def check_risk_limits(self) -> bool:
        """Check if trading should continue based on risk limits"""
        # Calculate daily P&L
        self.update_daily_pl()
        
        # Check daily loss limit
        account_equity = self.account_info.equity
        daily_loss_percent = abs(self.daily_pl) / account_equity * 100
        
        if daily_loss_percent >= self.config['max_daily_loss_percent']:
            logger.warning(f"Daily loss limit reached: {daily_loss_percent:.2f}%")
            self.daily_limit_reached = True
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            self.daily_limit_reached = True
            return False
        
        return True
    
    def update_daily_pl(self):
        """Update daily P&L"""
        # This would calculate P&L from open positions
        # For simplicity, we'll use the account equity change
        current_equity = self.account_info.equity
        # This would need to be tracked properly in production
        self.daily_pl = current_equity - self.account_info.balance
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return
        
        for position in positions:
            if position.comment and 'Bot Trade' in position.comment:
                # Check if position should be adjusted
                self.manage_position(position)
    
    def manage_position(self, position):
        """Manage an open position"""
        # Check if position has hit target
        current_price = position.price_current
        entry_price = position.price_open
        
        if position.type == mt5.POSITION_TYPE_BUY:
            # Check for take profit
            if current_price >= position.tp:
                logger.info(f"Take profit hit for {position.symbol}")
            # Move to breakeven if 50% of target reached
            elif current_price >= entry_price + (position.tp - entry_price) * 0.5:
                # Move stop to breakeven
                self.move_stop_to_breakeven(position)
        else:
            # Check for take profit
            if current_price <= position.tp:
                logger.info(f"Take profit hit for {position.symbol}")
            # Move to breakeven if 50% of target reached
            elif current_price <= entry_price - (entry_price - position.tp) * 0.5:
                self.move_stop_to_breakeven(position)
    
    def move_stop_to_breakeven(self, position):
        """Move stop loss to breakeven"""
        # This would modify the position's stop loss
        # For safety, we'd check current market conditions first
        pass
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Complete analysis of a symbol"""
        try:
            # Get data for all timeframes
            df_h4 = self.get_market_data(symbol, 'H4', 200)
            df_h1 = self.get_market_data(symbol, 'H1', 200)
            df_m15 = self.get_market_data(symbol, 'M15', 200)
            df_m5 = self.get_market_data(symbol, 'M5', 100)
            
            if df_h4.empty or df_h1.empty or df_m15.empty or df_m5.empty:
                logger.warning(f"Insufficient data for {symbol}")
                return {'decision': 'NO TRADE', 'reason': 'insufficient_data'}
            
            # Check spread
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'decision': 'NO TRADE', 'reason': 'symbol_not_found'}
            
            spread = symbol_info.spread
            if spread > self.config['max_spread']:
                logger.warning(f"Spread too high for {symbol}: {spread}")
                return {'decision': 'NO TRADE', 'reason': 'high_spread'}
            
            # Assess setup quality
            assessment = self.assess_setup_quality(
                symbol, df_h4, df_h1, df_m15, df_m5
            )
            
            # Calculate risk based on volatility
            risk_percent = self.config['default_risk_percent']
            if len(df_m5) > 50 and df_m5['atr'].iloc[-1] > df_m5['atr'].iloc[-20:].mean() * 1.5:
                risk_percent = self.config['high_volatility_risk_percent']
                logger.info(f"High volatility detected, reducing risk to {risk_percent}%")
            
            assessment['symbol'] = symbol
            assessment['risk_percent'] = risk_percent
            
            # Return final decision
            return assessment
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'decision': 'NO TRADE', 'reason': 'analysis_error'}
    
    def run_trading_session(self):
        """Main trading loop"""
        logger.info("Starting trading session")
        
        while not self.daily_limit_reached:
            try:
                # Check if we should continue
                if not self.check_risk_limits():
                    break
                
                # Check each allowed instrument
                for symbol in self.allowed_instruments:
                    if self.daily_limit_reached:
                        break
                    
                    # Skip if we already have a position in this symbol
                    if any(p.symbol == symbol for p in mt5.positions_get()):
                        continue
                    
                    # Analyze symbol
                    signal = self.analyze_symbol(symbol)
                    
                    # Print detailed analysis
                    self.print_signal_analysis(signal)
                    
                    # Execute if decision is BUY or SELL
                    if signal['decision'] in ['BUY', 'SELL']:
                        if self.execute_trade(signal):
                            logger.info(f"Trade executed for {symbol}")
                            # Wait a bit after opening a trade
                            time.sleep(60)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Wait before next iteration
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading session: {e}")
                time.sleep(60)
        
        if self.daily_limit_reached:
            logger.info("TRADING HALTED — DAILY RISK LIMIT REACHED")
        
        self.cleanup()
    
    def print_signal_analysis(self, signal: Dict):
        """Print detailed signal analysis"""
        if signal['decision'] in ['BUY', 'SELL']:
            logger.info(f"""
            {'='*60}
            INSTRUMENT: {signal.get('symbol', 'N/A')}
            DIRECTION: {signal['decision']}
            TIMEFRAME: Multi-timeframe (H4/H1/M15/M5)
            MARKET STRUCTURE: {signal.get('details', {}).get('structure', 'N/A')}
            SETUP: {signal.get('setup_type', 'N/A')}
            ENTRY: {signal.get('entry', 'N/A')}
            STOP LOSS: {signal.get('stop_loss', 'N/A')}
            TAKE PROFIT 1: {signal.get('take_profit', 'N/A')}
            TAKE PROFIT 2: {signal.get('take_profit', 'N/A') * 1.5 if signal.get('take_profit') else 'N/A'}
            RISK/REWARD: {signal.get('risk_reward', 'N/A')}
            RISK %: {signal.get('risk_percent', 'N/A')}%
            SETUP SCORE: {signal.get('score', 0)}/100
            CONFIDENCE: {signal.get('confidence', 'low')}
            DECISION: {signal['decision']}
            {'='*60}
            """)
        else:
            reason = signal.get('reason', 'No reason provided')
            logger.info(f"NO TRADE — {signal['decision']} — {reason}")
    
    def cleanup(self):
        """Clean up resources"""
        mt5.shutdown()
        logger.info("Trading bot stopped")
    
    def backtest(self, symbol: str, start_date: str, end_date: str):
        """Backtest the strategy"""
        # This would implement backtesting logic
        # For production, use MT5's strategy tester or a dedicated backtesting framework
        pass

# Configuration file creator
def create_config_file():
    """Create a sample configuration file"""
    config = {
        'default_risk_percent': 1.0,
        'high_volatility_risk_percent': 0.5,
        'max_daily_loss_percent': 2.0,
        'max_consecutive_losses': 3,
        'min_rr_ratio': 2.0,
        'min_setup_score': 80,
        'timeframes': {
            'analysis': ['H4', 'H1', 'M15', 'M5'],
            'entry': ['M5', 'M1']
        },
        'ema_periods': [20, 50, 200],
        'rsi_period': 14,
        'atr_period': 14,
        'max_spread': 50
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("Config file created: config.json")

# Main execution
if __name__ == "__main__":
    # Create config file
    create_config_file()
    
    # Initialize and run bot
    try:
        bot = TradingBot()
        bot.run_trading_session()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        mt5.shutdown()
