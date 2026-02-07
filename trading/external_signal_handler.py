"""
External Signal Handler Module
Handles signals coming from MT4/MT5 through file-based communication
"""
import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from threading import Thread, Event
import asyncio

from broker.deriv import DerivAPIWrapper
from utils.config import Config
from trading.executor import TradeExecutor

logger = logging.getLogger("bot.external_signal")


@dataclass
class ExternalSignal:
    """Represents an external signal from MT4/MT5"""
    symbol: str
    direction: str  # "CALL" or "PUT"
    timestamp: float  # Unix timestamp
    expiry_seconds: int
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if the signal is valid"""
        if self.direction not in ["CALL", "PUT"]:
            return False
        if self.expiry_seconds <= 0:
            return False
        if self.confidence is not None and (self.confidence < 0 or self.confidence > 1):
            return False
        return True
    
    def is_stale(self, max_age_seconds: int = 30) -> bool:
        """Check if the signal is too old to be processed"""
        return (time.time() - self.timestamp) > max_age_seconds


class ExternalSignalHandler:
    """Handles external signals from MT4/MT5"""
    
    def __init__(self, api: DerivAPIWrapper, cfg: Config, executor: TradeExecutor):
        self.api = api
        self.cfg = cfg
        self.executor = executor
        self.signals_dir = Path("signals")
        self.signals_dir.mkdir(exist_ok=True)
        self.signal_file = self.signals_dir / "signals.json"
        self.last_processed_timestamp = 0
        self.processed_signal_ids = set()
        self.shutdown_event = Event()
        
        # Initialize the signal file if it doesn't exist
        if not self.signal_file.exists():
            self.signal_file.write_text(json.dumps([]))
        
        logger.info(f"External signal handler initialized. Watching: {self.signal_file}")
    
    def read_signals_from_file(self) -> List[Dict]:
        """Read signals from the JSON file"""
        try:
            content = self.signal_file.read_text()
            if not content.strip():
                return []
            
            signals_data = json.loads(content)
            if not isinstance(signals_data, list):
                logger.warning("Signal file doesn't contain a list, resetting")
                return []
                
            return signals_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in signal file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading signal file: {e}")
            return []
    
    def process_external_signal(self, signal_data: Dict) -> bool:
        """Process a single external signal"""
        try:
            # Validate signal data
            required_fields = ['symbol', 'direction', 'timestamp', 'expiry_seconds']
            if not all(field in signal_data for field in required_fields):
                logger.warning(f"Invalid signal format: missing required fields in {signal_data}")
                return False
            
            signal = ExternalSignal(
                symbol=signal_data['symbol'],
                direction=signal_data['direction'].upper(),
                timestamp=signal_data['timestamp'],
                expiry_seconds=signal_data['expiry_seconds'],
                confidence=signal_data.get('confidence'),
                strategy=signal_data.get('strategy', 'mt4_mt5_signal')
            )
            
            # Validate the signal
            if not signal.is_valid():
                logger.warning(f"Invalid signal: {signal}")
                return False
            
            # Check if signal is stale
            if signal.is_stale():
                logger.info(f"Stale signal ignored: {signal}")
                return False
            
            # Check if we've already processed this signal
            signal_id = f"{signal.symbol}_{signal.direction}_{int(signal.timestamp)}"
            if signal_id in self.processed_signal_ids:
                logger.info(f"Duplicated signal ignored: {signal_id}")
                return False
            
            # Add to processed set
            self.processed_signal_ids.add(signal_id)
            
            # Log the received signal
            logger.info(f"Processing external signal: {signal}")
            
            # Execute the trade using existing executor logic
            return asyncio.run(self.execute_signal_trade(signal))
            
        except Exception as e:
            logger.error(f"Error processing external signal: {e}")
            return False
    
    async def execute_signal_trade(self, signal: ExternalSignal) -> bool:
        """Execute a trade based on the external signal using existing logic"""
        try:
            # Get current balance
            balance = await self.api.get_balance()
            
            # Check minimum balance
            if balance < self.cfg.min_balance:
                logger.warning(f"Balance too low: {balance} < {self.cfg.min_balance}")
                return False
            
            # Calculate stake based on configuration
            stake = min(balance * (self.cfg.trade_risk_percent / 100), self.cfg.max_stake)
            if stake < 1:
                stake = 1  # Minimum stake of $1
            
            # Prepare proposal arguments
            proposal_args = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": signal.direction,
                "currency": "USD",
                "duration": signal.expiry_seconds,
                "duration_unit": "s",
                "symbol": signal.symbol
            }
            
            logger.info(f"Executing trade: {signal.direction} on {signal.symbol} for ${stake} expiring in {signal.expiry_seconds}s")
            
            # Execute the trade using the existing executor method
            buy_result = await self.executor.execute_trade(proposal_args)
            
            if not buy_result:
                logger.error("Trade execution failed")
                return False
            
            contract_id = buy_result["buy"]["contract_id"]
            logger.info(f"Trade executed successfully | Contract ID={contract_id}")
            
            # Set up contract monitoring
            finished = asyncio.Event()
            
            def on_update(msg: Dict):
                self.executor._on_contract_update(
                    msg, signal.direction, contract_id, [signal.strategy], stake)
                poc = msg.get("proposal_open_contract", {})
                if poc.get("is_sold"):
                    finished.set()
            
            subscription = await self.api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            })
            subscription.subscribe(on_update)
            
            logger.info("Waiting for trade to settle...")
            await finished.wait()
            logger.info("Trade settled successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def process_all_signals(self):
        """Process all signals in the file"""
        signals = self.read_signals_from_file()
        
        for signal_data in signals:
            # Only process signals newer than the last processed timestamp
            if signal_data.get('timestamp', 0) > self.last_processed_timestamp:
                success = self.process_external_signal(signal_data)
                if success:
                    # Update the last processed timestamp
                    self.last_processed_timestamp = max(
                        self.last_processed_timestamp,
                        signal_data.get('timestamp', 0)
                    )
    
    def cleanup_old_signals(self):
        """Remove old processed signals from the file to prevent accumulation"""
        try:
            signals = self.read_signals_from_file()
            current_time = time.time()
            
            # Keep only signals from the last 5 minutes
            recent_signals = [
                signal for signal in signals
                if current_time - signal.get('timestamp', 0) <= 300  # 5 minutes
            ]
            
            # Write back the filtered signals
            self.signal_file.write_text(json.dumps(recent_signals))
            
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {e}")
    
    def start_monitoring(self, interval: int = 5):
        """Start monitoring the signal file in a separate thread"""
        def monitor_loop():
            logger.info("Starting external signal monitoring...")
            while not self.shutdown_event.is_set():
                try:
                    self.process_all_signals()
                    self.cleanup_old_signals()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in signal monitoring loop: {e}")
                    time.sleep(interval)
            
            logger.info("External signal monitoring stopped")
        
        monitor_thread = Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.shutdown_event.set()