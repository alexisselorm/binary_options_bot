"""
External Signal Handler Module - Production-Grade MT4/MT5 Bridge

Handles signals from MT4/MT5 through file-based communication with:
- Strict schema validation
- Atomic file operations
- Persistent deduplication
- Symbol mapping
- Confidence threshold gating
- Async-safe dispatch
"""
import json
import os
import time
import hashlib
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from threading import Thread, Event, Lock
import asyncio

from broker.deriv import DerivAPIWrapper
from utils.config import Config

# Lazy import to avoid circular dependencies
if TYPE_CHECKING:
    from trading.executor import TradeExecutor

logger = logging.getLogger("bot.external_signal")


# Valid directions for binary options
VALID_DIRECTIONS = {"CALL", "PUT"}

# Required fields in signal schema
REQUIRED_SIGNAL_FIELDS = {
    "source", "symbol", "direction", "timestamp",
    "expiry_seconds", "signal_id"
}

# Optional fields
OPTIONAL_SIGNAL_FIELDS = {"confidence", "strategy", "metadata"}


@dataclass
class ExternalSignal:
    """
    Represents an external signal from MT4/MT5 with strict schema.

    Schema:
    - source: Origin of signal (e.g., "mt5_ea", "mt4_indicator")
    - symbol: Trading symbol from MT platform
    - direction: "CALL" or "PUT"
    - timestamp: Unix timestamp in seconds
    - expiry_seconds: Trade duration in seconds
    - signal_id: Unique identifier for deduplication
    - confidence: Optional confidence score (0.0-1.0)
    - strategy: Optional strategy name
    - metadata: Optional additional data
    """
    source: str
    symbol: str
    direction: str
    timestamp: float
    expiry_seconds: int
    signal_id: str
    confidence: Optional[float] = None
    strategy: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize and validate fields after initialization"""
        if self.direction:
            self.direction = self.direction.upper()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Tuple[Optional["ExternalSignal"], List[str]]:
        """
        Create ExternalSignal from dictionary with validation.

        Returns:
            Tuple of (signal, errors) - signal is None if validation fails
        """
        errors = []

        # Check required fields
        missing_fields = REQUIRED_SIGNAL_FIELDS - set(data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            return None, errors

        # Validate and extract fields
        source = str(data.get("source", "")).strip()
        if not source:
            errors.append("source must be non-empty string")

        symbol = str(data.get("symbol", "")).strip()
        if not symbol:
            errors.append("symbol must be non-empty string")

        direction = str(data.get("direction", "")).upper().strip()
        if direction not in VALID_DIRECTIONS:
            errors.append(
                f"direction must be one of {VALID_DIRECTIONS}, got: {direction}")

        # Validate timestamp
        try:
            timestamp = float(data.get("timestamp", 0))
            if timestamp <= 0:
                errors.append("timestamp must be positive unix timestamp")
        except (TypeError, ValueError):
            errors.append("timestamp must be numeric")
            timestamp = 0

        # Validate expiry
        try:
            expiry_seconds = int(data.get("expiry_seconds", 0))
            if expiry_seconds <= 0:
                errors.append("expiry_seconds must be positive integer")
        except (TypeError, ValueError):
            errors.append("expiry_seconds must be integer")
            expiry_seconds = 0

        # Validate signal_id
        signal_id = str(data.get("signal_id", "")).strip()
        if not signal_id:
            errors.append("signal_id must be non-empty string")

        # Validate optional confidence
        confidence = None
        if "confidence" in data and data["confidence"] is not None:
            try:
                confidence = float(data["confidence"])
                if not (0.0 <= confidence <= 1.0):
                    errors.append(
                        f"confidence must be between 0.0 and 1.0, got: {confidence}")
            except (TypeError, ValueError):
                errors.append("confidence must be numeric")

        # Validate optional strategy
        strategy = data.get("strategy")
        if strategy is not None:
            strategy = str(strategy).strip()

        # Validate optional metadata
        metadata = data.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            errors.append("metadata must be a dictionary")
            metadata = None

        if errors:
            return None, errors

        return cls(
            source=source,
            symbol=symbol,
            direction=direction,
            timestamp=timestamp,
            expiry_seconds=expiry_seconds,
            signal_id=signal_id,
            confidence=confidence,
            strategy=strategy,
            metadata=metadata or {}
        ), []

    def is_valid(self) -> bool:
        """Check if the signal passes basic validation"""
        return self.direction in VALID_DIRECTIONS and self.expiry_seconds > 0

    def is_stale(self, max_age_seconds: int) -> bool:
        """Check if the signal is too old to be processed"""
        age = time.time() - self.timestamp
        return age > max_age_seconds

    def passes_confidence_threshold(self, threshold: float) -> bool:
        """
        Check if signal passes confidence threshold.
        If confidence is not set, passes by default.
        """
        if self.confidence is None:
            return True
        return self.confidence >= threshold

    def generate_hash(self) -> str:
        """Generate a unique hash for this signal for deduplication"""
        content = f"{self.source}:{self.symbol}:{self.direction}:{int(self.timestamp)}:{self.signal_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "source": self.source,
            "symbol": self.symbol,
            "direction": self.direction,
            "timestamp": self.timestamp,
            "expiry_seconds": self.expiry_seconds,
            "signal_id": self.signal_id,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "metadata": self.metadata
        }


class SymbolMapper:
    """
    Configurable mapping from MT4/MT5 symbols to Deriv symbols.
    Supports exact matches and pattern-based mapping.
    """

    # Default mappings for common symbols
    DEFAULT_MAPPINGS = {
        "EURUSD": "frxEURUSD",
        "GBPUSD": "frxGBPUSD",
        "USDJPY": "frxUSDJPY",
        "AUDUSD": "frxAUDUSD",
        "USDCAD": "frxUSDCAD",
        "USDCHF": "frxUSDCHF",
        "NZDUSD": "frxNZDUSD",
        "EURGBP": "frxEURGBP",
        "EURJPY": "frxEURJPY",
        "GBPJPY": "frxGBPJPY",
        # Synthetic indices
        "R_100": "R_100",
        "R_75": "R_75",
        "R_50": "R_50",
        "R_25": "R_25",
        "R_10": "R_10",
        "1HZ10V": "1HZ10V",
        "1HZ100V": "1HZ100V",
        "1HZ50V": "1HZ50V",
        "1HZ200V": "1HZ200V",
    }

    def __init__(self, custom_mappings: Optional[Dict[str, str]] = None):
        self.mappings = {**self.DEFAULT_MAPPINGS}
        if custom_mappings:
            self.mappings.update(custom_mappings)
        logger.debug(
            f"Symbol mapper initialized with {len(self.mappings)} mappings")

    def map_symbol(self, mt_symbol: str) -> Optional[str]:
        """
        Map MT symbol to Deriv symbol.

        Args:
            mt_symbol: Symbol from MT4/MT5

        Returns:
            Deriv symbol or None if not mappable
        """
        # Try exact match first
        if mt_symbol in self.mappings:
            return self.mappings[mt_symbol]

        # Try case-insensitive match
        mt_upper = mt_symbol.upper()
        for src, dst in self.mappings.items():
            if src.upper() == mt_upper:
                return dst

        # Try common suffix/prefix patterns
        # e.g., EURUSD.a -> frxEURUSD
        base_symbol = mt_symbol.split('.')[0].upper()
        if base_symbol in self.mappings:
            return self.mappings[base_symbol]

        # No mapping found
        return None

    def add_mapping(self, mt_symbol: str, deriv_symbol: str):
        """Add a custom mapping"""
        self.mappings[mt_symbol] = deriv_symbol
        logger.debug(f"Added symbol mapping: {mt_symbol} -> {deriv_symbol}")


class PersistentDeduplicator:
    """
    File-based deduplication that survives bot restarts.
    Stores processed signal hashes in a JSON file.
    """

    def __init__(self, storage_path: Path, max_history_hours: int = 24):
        self.storage_path = storage_path
        self.max_history_seconds = max_history_hours * 3600
        self.processed_hashes: Dict[str, float] = {}  # hash -> timestamp
        self.lock = Lock()
        self._load()

    def _load(self):
        """Load processed hashes from file"""
        try:
            if self.storage_path.exists():
                data = json.loads(self.storage_path.read_text())
                self.processed_hashes = {
                    h: float(t) for h, t in data.items()
                }
                # Clean old entries on load
                self._cleanup()
                logger.debug(
                    f"Loaded {len(self.processed_hashes)} processed signal hashes")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not load deduplication storage: {e}")
            self.processed_hashes = {}
        except Exception as e:
            logger.error(f"Error loading deduplication storage: {e}")
            self.processed_hashes = {}

    def _save(self):
        """Save processed hashes to file atomically"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename (atomic on POSIX)
            fd, temp_path = tempfile.mkstemp(
                suffix='.json.tmp',
                dir=str(self.storage_path.parent)
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self.processed_hashes, f, indent=2)
                shutil.move(temp_path, str(self.storage_path))
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            logger.error(f"Error saving deduplication storage: {e}")

    def _cleanup(self):
        """Remove old entries"""
        cutoff = time.time() - self.max_history_seconds
        self.processed_hashes = {
            h: t for h, t in self.processed_hashes.items()
            if t > cutoff
        }

    def is_processed(self, signal_hash: str) -> bool:
        """Check if signal has already been processed"""
        with self.lock:
            return signal_hash in self.processed_hashes

    def mark_processed(self, signal: ExternalSignal) -> bool:
        """
        Mark signal as processed.

        Returns:
            True if newly marked, False if already processed
        """
        signal_hash = signal.generate_hash()

        with self.lock:
            if signal_hash in self.processed_hashes:
                return False

            self.processed_hashes[signal_hash] = time.time()
            self._cleanup()
            self._save()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed signals"""
        with self.lock:
            current_time = time.time()
            cutoff_1h = current_time - 3600
            cutoff_24h = current_time - 86400

            return {
                "total_processed": len(self.processed_hashes),
                "last_hour": sum(1 for t in self.processed_hashes.values() if t > cutoff_1h),
                "last_24h": sum(1 for t in self.processed_hashes.values() if t > cutoff_24h),
            }


class ExternalSignalHandler:
    """
    Production-grade handler for external signals from MT4/MT5.

    Features:
    - Strict schema validation
    - Atomic file reads/writes
    - Persistent deduplication
    - Symbol mapping
    - Confidence threshold gating
    - Async-safe dispatch to event loop
    """

    def __init__(
        self,
        api: DerivAPIWrapper,
        cfg: Config,
        executor: "TradeExecutor",
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        self.api = api
        self.cfg = cfg
        self.executor = executor
        self.loop = loop or asyncio.get_event_loop()

        # Signal file configuration
        self.signals_dir = Path(cfg.external_signal_dir)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.signal_file = self.signals_dir / cfg.external_signal_file
        self.backup_signal_file = self.signals_dir / \
            f".{cfg.external_signal_file}.backup"

        # Deduplication storage
        self.dedup_storage_path = self.signals_dir / ".processed_signals.json"
        self.deduplicator = PersistentDeduplicator(
            self.dedup_storage_path,
            max_history_hours=24
        )

        # Symbol mapper
        self.symbol_mapper = SymbolMapper(cfg.external_symbol_map)

        # Configuration
        self.max_age_seconds = cfg.external_signal_max_age_seconds
        self.confidence_threshold = cfg.external_signal_confidence_threshold
        self.processing_interval = cfg.external_signal_interval

        # Thread safety
        self.shutdown_event = Event()
        self.file_lock = Lock()
        self.monitor_thread: Optional[Thread] = None

        # Statistics
        self.stats = {
            "signals_received": 0,
            "signals_validated": 0,
            "signals_rejected": 0,
            "signals_stale": 0,
            "signals_duplicate": 0,
            "signals_low_confidence": 0,
            "signals_symbol_error": 0,
            "trades_executed": 0,
            "trades_failed": 0,
        }
        self.stats_lock = Lock()

        # Initialize the signal file if it doesn't exist
        self._init_signal_file()

        logger.info(
            f"External signal handler initialized | "
            f"file={self.signal_file} | "
            f"max_age={self.max_age_seconds}s | "
            f"confidence_threshold={self.confidence_threshold}"
        )

    def _init_signal_file(self):
        """Initialize signal file with empty array if not exists"""
        try:
            if not self.signal_file.exists():
                self._atomic_write_signal_file([])
                logger.debug(f"Initialized signal file: {self.signal_file}")
        except Exception as e:
            logger.error(f"Failed to initialize signal file: {e}")

    def _atomic_write_signal_file(self, signals: List[Dict]) -> bool:
        """
        Write signals to file atomically using temp file + rename.

        Args:
            signals: List of signal dictionaries

        Returns:
            True if successful, False otherwise
        """
        try:
            self.signal_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                suffix='.json.tmp',
                dir=str(self.signal_file.parent)
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(signals, f, indent=2)

                # Atomic rename
                shutil.move(temp_path, str(self.signal_file))
                return True
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            return False

    def _read_signals_from_file(self) -> List[Dict]:
        """
        Read signals from JSON file with error recovery.

        Returns:
            List of signal dictionaries, empty list on error
        """
        try:
            with self.file_lock:
                if not self.signal_file.exists():
                    return []

                content = self.signal_file.read_text().strip()
                if not content:
                    return []

                # Try to parse JSON
                try:
                    signals_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error, trying backup: {e}")

                    # Try backup file
                    if self.backup_signal_file.exists():
                        try:
                            content = self.backup_signal_file.read_text().strip()
                            signals_data = json.loads(content)
                            logger.info("Recovered from backup file")
                        except Exception:
                            logger.error("Backup file also corrupted")
                            return []
                    else:
                        return []

                if not isinstance(signals_data, list):
                    logger.warning("Signal file is not a list, resetting")
                    return []

                return signals_data

        except Exception as e:
            logger.error(f"Error reading signal file: {e}")
            return []

    def _validate_and_parse_signal(
        self,
        signal_data: Dict[str, Any]
    ) -> Tuple[Optional[ExternalSignal], str]:
        """
        Validate and parse a signal dictionary.

        Returns:
            Tuple of (signal, rejection_reason) - signal is None if invalid
        """
        # Parse with validation
        signal, errors = ExternalSignal.from_dict(signal_data)

        if errors:
            return None, f"Schema validation failed: {'; '.join(errors)}"

        # Check staleness
        if signal.is_stale(self.max_age_seconds):
            age = time.time() - signal.timestamp
            return None, f"Signal stale (age={age:.1f}s, max={self.max_age_seconds}s)"

        # Check confidence threshold
        if not signal.passes_confidence_threshold(self.confidence_threshold):
            return None, (
                f"Confidence {signal.confidence} below threshold {self.confidence_threshold}"
            )

        # Map symbol
        deriv_symbol = self.symbol_mapper.map_symbol(signal.symbol)
        if deriv_symbol is None:
            return None, f"Symbol mapping failed: {signal.symbol} -> unknown"

        # Store mapped symbol in metadata for logging
        signal.metadata["deriv_symbol"] = deriv_symbol

        return signal, ""

    def _increment_stat(self, stat_name: str):
        """Thread-safe stat increment"""
        with self.stats_lock:
            self.stats[stat_name] = self.stats.get(stat_name, 0) + 1

    def _get_stats(self) -> Dict[str, int]:
        """Thread-safe stat retrieval"""
        with self.stats_lock:
            return dict(self.stats)

    def _dispatch_to_loop(
        self,
        coro
    ):
        """
        Safely dispatch a coroutine to the event loop.

        This is the correct way to schedule coroutines from threads.
        """
        if self.loop.is_closed():
            logger.error("Event loop is closed, cannot dispatch")
            future = asyncio.Future()
            future.set_exception(RuntimeError("Event loop closed"))
            return future

        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _execute_signal_trade(self, signal: ExternalSignal) -> bool:
        """
        Execute a trade based on the external signal.

        Uses existing executor logic with proper risk management.
        """
        deriv_symbol = signal.metadata.get("deriv_symbol", signal.symbol)

        try:
            # Get current balance
            balance = await self.api.get_balance()

            # Check minimum balance
            if balance < self.cfg.min_balance:
                logger.warning(
                    f"Balance too low for trade: {balance} < {self.cfg.min_balance}"
                )
                return False

            # Calculate stake
            stake = min(
                balance * (self.cfg.trade_risk_percent / 100),
                self.cfg.max_stake
            )
            stake = max(stake, 1.0)  # Minimum $1 stake

            # Validate duration is tradable
            if not self.executor.valid_durations:
                self.executor.valid_durations = await self.api.fetch_trading_durations(
                    deriv_symbol
                )

            if signal.expiry_seconds not in self.executor.valid_durations:
                logger.warning(
                    f"Expiry {signal.expiry_seconds}s not valid for {deriv_symbol}. "
                    f"Valid: {self.executor.valid_durations[:10]}..."
                )
                # Use configured fallback expiry
                expiry_to_use = self.cfg.expiry * 60  # Convert minutes to seconds
                if expiry_to_use not in self.executor.valid_durations:
                    # Use first available duration
                    if self.executor.valid_durations:
                        expiry_to_use = self.executor.valid_durations[0]
                    else:
                        logger.error(f"No valid durations for {deriv_symbol}")
                        return False
                logger.info(f"Using fallback expiry: {expiry_to_use}s")
            else:
                expiry_to_use = signal.expiry_seconds

            # Prepare proposal
            proposal_args = {
                "proposal": 1,
                "amount": stake,
                "basis": "stake",
                "contract_type": signal.direction,
                "currency": "USD",
                "duration": expiry_to_use,
                "duration_unit": "s",
                "symbol": deriv_symbol
            }

            strategy_name = signal.strategy or "external_signal"
            logger.info(
                f"Executing external trade | "
                f"signal_id={signal.signal_id} | "
                f"direction={signal.direction} | "
                f"symbol={deriv_symbol} | "
                f"expiry={expiry_to_use}s | "
                f"stake=${stake:.2f} | "
                f"strategy={strategy_name} | "
                f"confidence={signal.confidence}"
            )

            # Execute trade
            buy_result = await self.executor.execute_trade(proposal_args)

            if not buy_result:
                self._increment_stat("trades_failed")
                logger.error("Trade execution failed")
                return False

            contract_id = buy_result["buy"]["contract_id"]
            self._increment_stat("trades_executed")

            logger.info(
                f"Trade executed successfully | "
                f"contract_id={contract_id} | "
                f"signal_id={signal.signal_id}"
            )

            # Set up contract monitoring
            finished = asyncio.Event()
            strategies_list = [strategy_name]

            def on_update(msg: Dict):
                self.executor._on_contract_update(
                    msg,
                    signal.direction,
                    contract_id,
                    strategies_list,
                    stake,
                    source=signal.source,
                    signal_id=signal.signal_id
                )
                poc = msg.get("proposal_open_contract", {})
                if poc.get("is_sold"):
                    finished.set()

            subscription = await self.api.subscribe({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            })
            subscription.subscribe(on_update)

            logger.debug(f"Waiting for trade settlement: {contract_id}")
            await finished.wait()
            logger.info(f"Trade settled: {contract_id}")

            return True

        except Exception as e:
            self._increment_stat("trades_failed")
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False

    def _process_signal(self, signal_data: Dict) -> bool:
        """
        Process a single signal with full validation and execution.

        Returns:
            True if trade was executed, False otherwise
        """
        self._increment_stat("signals_received")

        # Validate and parse
        signal, rejection_reason = self._validate_and_parse_signal(signal_data)

        if signal is None:
            self._increment_stat("signals_rejected")
            logger.debug(
                f"Signal rejected: {rejection_reason} | data={signal_data}")

            # Track specific rejection reasons
            if "stale" in rejection_reason.lower():
                self._increment_stat("signals_stale")
            elif "confidence" in rejection_reason.lower():
                self._increment_stat("signals_low_confidence")
            elif "symbol" in rejection_reason.lower():
                self._increment_stat("signals_symbol_error")

            return False

        # Check for duplicate (persistent)
        if self.deduplicator.is_processed(signal.generate_hash()):
            self._increment_stat("signals_duplicate")
            logger.debug(f"Duplicate signal ignored: {signal.signal_id}")
            return False

        # Mark as processed before execution (prevents double-execution)
        self.deduplicator.mark_processed(signal)
        self._increment_stat("signals_validated")

        logger.info(
            f"Processing validated signal | "
            f"id={signal.signal_id} | "
            f"source={signal.source} | "
            f"symbol={signal.symbol}->{signal.metadata.get('deriv_symbol')} | "
            f"direction={signal.direction} | "
            f"expiry={signal.expiry_seconds}s"
        )

        # Dispatch to event loop safely from thread
        future = self._dispatch_to_loop(self._execute_signal_trade(signal))

        try:
            # Wait for result with timeout
            result = future.result(timeout=30.0)
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"Trade execution timeout for signal {signal.signal_id}")
            return False
        except Exception as e:
            logger.error(f"Error in trade dispatch: {e}")
            return False

    def _process_all_signals(self):
        """Process all signals in the file"""
        signals = self._read_signals_from_file()

        if not signals:
            return

        processed_count = 0

        for signal_data in signals:
            if not isinstance(signal_data, dict):
                continue

            if self._process_signal(signal_data):
                processed_count += 1

        if processed_count > 0:
            logger.debug(f"Processed {processed_count} signals")

        # Cleanup old signals from file periodically
        self._cleanup_signal_file()

    def _cleanup_signal_file(self):
        """Remove old processed signals from the file"""
        try:
            signals = self._read_signals_from_file()
            current_time = time.time()

            # Keep only recent unprocessed signals (last 5 minutes)
            recent_signals = []
            for signal_data in signals:
                if not isinstance(signal_data, dict):
                    continue

                timestamp = signal_data.get("timestamp", 0)
                if current_time - timestamp <= 300:  # 5 minutes
                    recent_signals.append(signal_data)

            # Write back filtered signals
            self._atomic_write_signal_file(recent_signals)

        except Exception as e:
            logger.error(f"Error cleaning up signal file: {e}")

    def start_monitoring(self):
        """Start monitoring the signal file in a separate thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Signal monitoring already running")
            return

        self.shutdown_event.clear()

        def monitor_loop():
            logger.info("Starting external signal monitoring thread")

            while not self.shutdown_event.is_set():
                try:
                    self._process_all_signals()
                except Exception as e:
                    logger.error(
                        f"Error in signal monitoring: {e}", exc_info=True)

                # Wait with interrupt check
                self.shutdown_event.wait(self.processing_interval)

            logger.info("External signal monitoring thread stopped")

        self.monitor_thread = Thread(
            target=monitor_loop,
            daemon=True,
            name="ExternalSignalMonitor"
        )
        self.monitor_thread.start()
        logger.debug("Signal monitoring thread started")

    def stop_monitoring(self):
        """Stop the monitoring thread gracefully"""
        logger.info("Stopping external signal monitoring...")
        self.shutdown_event.set()

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning(
                    "Signal monitoring thread did not stop gracefully")
            else:
                logger.debug("Signal monitoring thread stopped")

        self.monitor_thread = None

    def get_status(self) -> Dict[str, Any]:
        """Get current handler status"""
        stats = self._get_stats()
        dedup_stats = self.deduplicator.get_stats()

        return {
            "running": self.monitor_thread is not None and self.monitor_thread.is_alive(),
            "signal_file": str(self.signal_file),
            "signal_file_exists": self.signal_file.exists(),
            "stats": stats,
            "deduplication": dedup_stats,
            "config": {
                "max_age_seconds": self.max_age_seconds,
                "confidence_threshold": self.confidence_threshold,
                "processing_interval": self.processing_interval,
            }
        }
