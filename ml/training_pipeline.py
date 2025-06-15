# ml/training_pipeline.py
"""
Enhanced Trading Model Training Pipeline
Fetch candles ➜ advanced feature engineering ➜ train LSTM with Attention ➜ train XGBoost
Saves:
  • data/historical_candles.csv             (raw)
  • data/historical_features.csv            (with enhanced TA)
  • models/lstm_model.keras                 (Bidirectional LSTM with Attention)
  • models/xgb_model.json                   (optimized XGBoost)
  • models/lstm_training.png                (loss/acc/auc curves)
  • models/xgb_feature_importance.png       (bar chart)
  • models/performance_metrics.txt          (validation metrics)
"""

import asyncio
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, Bidirectional, Attention, Concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.optimizers import AdamW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier, plot_importance
import joblib
import pandas_ta as ta

from broker.deriv import DerivAPIWrapper
from utils.config import Config

# Constants
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RAW_CSV = DATA_DIR / "historical_candles.csv"
FEAT_CSV = DATA_DIR / "historical_features.csv"
LSTM_SEQ_LEN = 100
PREDICTION_WINDOW = 5
THRESHOLD = 0.002  # 0.2% price movement threshold

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 1. Data Fetching
# ----------------------------------------------------------------------


async def download_candles(cfg: Config, count: int = 50000) -> pd.DataFrame:
    """Fetch historical candles from Deriv API"""
    api = DerivAPIWrapper(cfg.app_id)
    await api.authorize(cfg.api_token)
    resp = await api.get_candles(cfg.asset, count=count,
                                 granularity=cfg.timeframe * 60)
    candles: List[dict] = resp.get("candles", [])
    if not candles:
        raise RuntimeError("No candles returned from Deriv API")
    df = pd.DataFrame(candles)
    df.to_csv(RAW_CSV, index=False)
    print(
        f"📁 Raw candles saved → {RAW_CSV.relative_to(ROOT)} (rows={len(df)})")
    return df

# ----------------------------------------------------------------------
# 2. Advanced Feature Engineering
# ----------------------------------------------------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators and features"""
    df = df.copy()

    # Price transformations
    df["log_close"] = np.log(df["close"])
    df["price_change"] = df["close"].pct_change()
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]

    # Trend indicators
    df.ta.ema(length=8, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.ichimoku(append=True)

    # Momentum indicators
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)

    # Volatility indicators
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2.0, append=True)
    df.ta.kc(length=20, scalar=2, append=True)
    df["volatility"] = df["close"].rolling(14).std()

    # Pattern recognition
    df.ta.cdl_pattern(name="doji", append=True)
    df.ta.cdl_pattern(name="engulfing", append=True)
    df.ta.cdl_pattern(name="hammer", append=True)

    # Time-based features
    df["hour"] = pd.to_datetime(df["epoch"], unit="s").dt.hour
    df["day_of_week"] = pd.to_datetime(df["epoch"], unit="s").dt.dayofweek

    # Drop NA values from indicators
    df.dropna(inplace=True)
    df.to_csv(FEAT_CSV, index=False)
    print(
        f"📁 Feature CSV saved → {FEAT_CSV.relative_to(ROOT)} (rows={len(df)})")
    return df

# ----------------------------------------------------------------------
# 3. Enhanced LSTM Model with Attention
# ----------------------------------------------------------------------


def prepare_sequences(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences with max 200 timesteps"""
    MAX_SEQ_LENGTH = 200  # Hard limit for memory safety

    prices = df["close"].values
    scaler = MinMaxScaler()
    ohlc_scaled = scaler.fit_transform(df[["open", "high", "low", "close"]])

    sequences, labels = [], []

    for i in range(MAX_SEQ_LENGTH, len(df) - PREDICTION_WINDOW):
        # Take only the most recent 200 timesteps
        seq = ohlc_scaled[i-MAX_SEQ_LENGTH:i]
        future_return = (prices[i + PREDICTION_WINDOW] - prices[i]) / prices[i]
        label = 1 if future_return > THRESHOLD else 0
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


class EfficientAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.query_dense = Dense(self.units)
        self.key_dense = Dense(self.units)
        self.value_dense = Dense(self.units)

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Efficient attention (reduces O(n²) memory)
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(scores, value)

    def get_config(self):
        return {'units': self.units}


def train_lstm(X: np.ndarray, y: np.ndarray) -> Model:
    """Memory-optimized LSTM for 200-step sequences"""
    inputs = Input(shape=(200, 4))  # Fixed input shape

    # 1. Efficient BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 2. Lightweight Attention
    query = Dense(32)(x)
    key = Dense(32)(x)
    value = Dense(32)(x)
    attention = tf.keras.layers.Attention()([query, value, key])
    x = Concatenate()([x, attention])

    # 3. Final layers
    x = Bidirectional(LSTM(32))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="swish")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Callbacks (keep your existing ones)
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=2,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5),
        ModelCheckpoint(str(MODELS_DIR / "lstm_best.keras"),
                        save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(MODELS_DIR / "logs"))
    ]

    # Train with class weights
    hist = model.fit(
        X, y,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight={0: 1, 1: 1.2}
    )

    # Plotting (keep your existing code)
    plot_training_history(hist)

    return model


def plot_training_history(history):
    """Plot training curves"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "lstm_training.png")
    plt.close()
# ----------------------------------------------------------------------
# 4. Optimized XGBoost Model
# ----------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get relevant feature columns from dataframe"""
    # Exclude raw price columns and time columns
    exclude_cols = {"open", "high", "low", "close", "epoch", "log_close"}
    return [col for col in df.columns if col not in exclude_cols]


def train_xgb(df: pd.DataFrame, lstm_model: Model) -> XGBClassifier:
    """Train XGBoost model with LSTM features (updated for 200-timestep compatibility)"""
    # Get feature columns (UNCHANGED)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.astype(float)

    # Scale features (UNCHANGED)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Add LSTM predictions as features - CRITICAL UPDATE
    seqs = []
    for i in range(200, len(df)):  # Changed from LSTM_SEQ_LEN to 200
        seq = df[["open", "high", "low", "close"]
                 ].iloc[i-200:i].values  # Fixed 200 timesteps
        seqs.append(seq)
    seqs = np.array(seqs)

    # Verify sequence shape matches LSTM expectations
    if seqs.shape[1] != 200:
        raise ValueError(f"LSTM expects 200 timesteps, got {seqs.shape[1]}")

    lstm_preds = lstm_model.predict(seqs).flatten()
    X = np.hstack([X[200:], lstm_preds.reshape(-1, 1)])  # Adjusted offset

    # Create target (UNCHANGED except adjusted indices)
    prices = df["close"].values
    future_returns = (np.roll(prices, -PREDICTION_WINDOW) - prices) / prices
    y = np.where(future_returns > THRESHOLD, 1, 0)
    y = y[200:-PREDICTION_WINDOW]  # Adjusted for 200-timestep window
    X = X[:len(y)]  # Ensure alignment

    # Time-based train-test split (UNCHANGED)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Optimized XGBoost parameters (UNCHANGED)
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.2,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric=["auc", "error"],
        tree_method="hist",
        early_stopping_rounds=50,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
    )

    # Train with validation (UNCHANGED)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # Evaluate (UNCHANGED)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Save metrics (UNCHANGED)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    with open(MODELS_DIR / "performance_metrics.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nAUC-ROC: {metrics['roc_auc']:.4f}")

    # Save model (UNCHANGED)
    model.save_model(str(MODELS_DIR / "xgb_model.json"))
    print(f"✅ XGBoost saved → {MODELS_DIR / 'xgb_model.json'}")

    # Plot feature importance (UNCHANGED)
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=20)
    plt.tight_layout()
    out_png = MODELS_DIR / "xgb_feature_importance.png"
    plt.savefig(out_png)
    plt.close()
    print(f"🖼️ XGB importance → {out_png.relative_to(ROOT)}")

    return model

# ----------------------------------------------------------------------
# 5. Main Pipeline
# ----------------------------------------------------------------------


async def async_main():
    """Main async training workflow"""
    cfg = Config()

    # 1. Fetch data
    print("🚀 Starting data download...")
    raw_df = await download_candles(cfg, count=50000)

    # 2. Feature engineering
    print("🔧 Engineering features...")
    feat_df = add_indicators(raw_df)

    # 3. Train LSTM
    print("🧠 Training LSTM model...")
    X_seq, y_seq = prepare_sequences(feat_df)
    lstm_mdl = train_lstm(X_seq, y_seq)

    # 4. Train XGBoost
    print("🌲 Training XGBoost model...")
    xgb_mdl = train_xgb(feat_df, lstm_mdl)

    print("🎉 Training pipeline completed!")


def main():
    """Run the training pipeline"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
