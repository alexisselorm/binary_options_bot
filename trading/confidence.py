# trading/confidence.py

import json
import os
from datetime import datetime

CONF_PATH = "data/strategy_confidence.json"
DEFAULT_CONFIDENCE = 0.5
DECAY = 0.5  # You can tune this globally or per strategy

_conf_data = {}


def _load_conf():
    global _conf_data
    if os.path.exists(CONF_PATH):
        with open(CONF_PATH, "r") as f:
            _conf_data = json.load(f)
    else:
        _conf_data = {}


def _save_conf():
    with open(CONF_PATH, "w") as f:
        json.dump(_conf_data, f, indent=2)


def get_confidence(strategy_name: str) -> float:
    _load_conf()
    entry = _conf_data.get(strategy_name, {})
    return entry.get("confidence", DEFAULT_CONFIDENCE)


def record_result(strategy_names: list, won: bool, decay: float = DECAY):
    _load_conf()

    for stra in strategy_names:
        old_entry = _conf_data.get(stra, {})
        old_conf = old_entry.get("confidence", DEFAULT_CONFIDENCE)

        result = 1.0 if won else 0.0
        new_conf = round(old_conf * decay + result * (1 - decay), 4)

        _conf_data[stra] = {
            "confidence": new_conf,
            "last_updated": datetime.utcnow().isoformat()
        }

        _save_conf()


def reset_confidence(strategy_name: str = None):
    _load_conf()
    if strategy_name:
        _conf_data[strategy_name] = {
            "confidence": DEFAULT_CONFIDENCE,
            "last_updated": datetime.utcnow().isoformat()
        }
    else:
        for strat in _conf_data:
            _conf_data[strat] = {
                "confidence": DEFAULT_CONFIDENCE,
                "last_updated": datetime.utcnow().isoformat()
            }
    _save_conf()
