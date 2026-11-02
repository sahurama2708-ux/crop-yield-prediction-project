import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "ultimate_crop_yield_dl_model.h5"
FEATURE_PATH = PROJECT_DIR / "artifacts" / "feature_columns.pkl"
RAW_DATA_PATH = PROJECT_DIR / "data" / "row" / "yield_df.csv"
DB_PATH = APP_DIR / "history.db"

# These are the exact input feature names used by the trained model.
NUMERIC_FEATURES = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
COUNTRY_FEATURE = "Area_India"
CROP_FEATURES = {
    "Maize": "Item_Maize",
    "Plantains and others": "Item_Plantains and others",
    "Potatoes": "Item_Potatoes",
    "Rice": "Item_Rice, paddy",
    "Sorghum": "Item_Sorghum",
    "Soybeans": "Item_Soybeans",
    "Sweet potatoes": "Item_Sweet potatoes",
    "Wheat": "Item_Wheat",
    "Yams": "Item_Yams",
}

model = None
feature_columns = None
numeric_stats = None
model_load_error = None


def load_artifacts():
    """Load model metadata and derive the numeric standardization from raw training data."""
    global model, feature_columns, numeric_stats, model_load_error

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(f"Feature metadata not found: {FEATURE_PATH.name}")
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found: {RAW_DATA_PATH.name}")

        model = load_model(str(MODEL_PATH), compile=False)
        feature_columns = joblib.load(FEATURE_PATH)

        raw = pd.read_csv(RAW_DATA_PATH)
        required = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"Training data is missing columns: {', '.join(missing)}")

        # The processed training table stores these four fields standardized.
        # The scaler.pkl in this project is an identity scaler, so it cannot
        # transform raw form values. Rebuild the standardization from raw training data.
        stats = {}
        for col in required:
            values = pd.to_numeric(raw[col], errors="coerce").dropna()
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid training-data statistics for {col}")
            stats[col] = (mean, std)
        numeric_stats = stats

        if len(feature_columns) != int(model.input_shape[-1]):
            raise ValueError(
                f"Feature mismatch: metadata has {len(feature_columns)} columns, "
                f"model expects {model.input_shape[-1]}"
            )
        missing_model_features = [
            col for col in [*NUMERIC_FEATURES, COUNTRY_FEATURE, *CROP_FEATURES.values()]
            if col not in feature_columns
        ]
        if missing_model_features:
            raise ValueError(
                "Feature metadata is missing model inputs: "
                + ", ".join(missing_model_features)
            )

        model_load_error = None
        app.logger.info("Prediction model and feature metadata loaded.")
    except Exception as exc:
        model = None
        feature_columns = None
        numeric_stats = None
        model_load_error = str(exc)
        app.logger.exception("Could not load prediction artifacts")


load_artifacts()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                location TEXT,
                crop TEXT,
                season TEXT,
                soil TEXT,
                area REAL,
                yield_val REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


init_db()


def build_model_input(data):
    """Build the 113-column input in the exact order used during training."""
    crop = str(data.get("crop", "Rice")).strip()
    if crop not in CROP_FEATURES:
        raise ValueError(
            f"Unsupported crop '{crop}'. Choose one of: {', '.join(CROP_FEATURES)}"
        )

    values = {
        "Year": 2013.0,  # latest year represented in this model's training dataset
        "average_rain_fall_mm_per_year": float(data.get("rainfall", 1200)),
        "pesticides_tonnes": float(data.get("pesticides", 45)),
        "avg_temp": float(data.get("temperature", 27.5)),
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a valid number")
    if values["average_rain_fall_mm_per_year"] <= 0:
        raise ValueError("Rainfall must be greater than zero.")
    if values["pesticides_tonnes"] < 0:
        raise ValueError("Pesticides cannot be negative.")
    if not -5 <= values["avg_temp"] <= 55:
        raise ValueError("Temperature must be between -5°C and 55°C.")

    row = pd.DataFrame(0.0, index=[0], columns=feature_columns, dtype=np.float32)
    for name, raw_value in values.items():
        mean, std = numeric_stats[name]
        row.at[0, name] = (raw_value - mean) / std

    # The training data has country-level Area_* features, not district names.
    # This application is configured for Indian crop estimates.
    row.at[0, COUNTRY_FEATURE] = 1.0
    row.at[0, CROP_FEATURES[crop]] = 1.0
    return row.to_numpy(dtype=np.float32), crop, values


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "message": "Model ready" if model is not None else model_load_error,
    }), (200 if model is not None else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or feature_columns is None or numeric_stats is None:
        return jsonify({
            "status": "error",
            "message": f"Prediction model is not ready: {model_load_error or 'unknown error'}",
        }), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Send prediction inputs as JSON."}), 400

    try:
        email = str(data.get("email") or "guest@gmail.com").strip().lower()
        location = str(data.get("location") or "India").strip()
        season = str(data.get("season") or "").strip()
        soil = str(data.get("soil") or "").strip()
        area = float(data.get("area", 1))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Field area must be greater than zero.")

        model_input, crop, _ = build_model_input(data)
        raw_prediction = np.asarray(model.predict(model_input, verbose=0)).reshape(-1)
        if raw_prediction.size != 1 or not np.isfinite(raw_prediction[0]):
            raise ValueError("The model returned an invalid prediction.")

        # Model target is hg/ha. Convert to quintals per acre:
        # 1 hg/ha = 0.00040468564224 quintals/acre.
        yield_hg_ha = float(raw_prediction[0])
        if yield_hg_ha <= 0:
            raise ValueError(
                "The model returned a non-positive yield. Please check the input values."
            )
        yield_q_acre = round(yield_hg_ha * 0.00040468564224, 2)
        total_yield_q = round(yield_q_acre * area, 2)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions
                (email, location, crop, season, soil, area, yield_val)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (email, location, crop, season, soil, area, yield_q_acre),
            )

        return jsonify({
            "status": "success",
            "crop": crop,
            "season": season,
            "soil": soil,
            "area": area,
            "raw_model_output_hg_ha": round(yield_hg_ha, 2),
            "estimated_yield_q_acre": yield_q_acre,
            "estimated_total_yield_quintals": total_yield_q,
            "confidence": None,
            "note": (
                "The trained model uses year, rainfall, pesticide use, temperature, "
                "country and crop. Season and soil are saved in history but are not "
                "model features in this training dataset."
            ),
        })
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "status": "error",
            "message": "Prediction failed on the server. Check Render logs for details.",
        }), 500


@app.route("/history", methods=["GET"])
def get_history():
    email = request.args.get("email", "").strip().lower()
    if not email:
        return jsonify({"status": "error", "message": "Email required"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT location, crop, season, soil, area, yield_val, timestamp
            FROM predictions WHERE email = ? ORDER BY id DESC
            """,
            (email,),
        ).fetchall()

    history = [
        {
            "location": row[0],
            "crop": row[1],
            "season": row[2],
            "soil": row[3],
            "area": row[4],
            "yield_val": row[5],
            "timestamp": row[6],
        }
        for row in rows
    ]
    return jsonify({"status": "success", "history": history})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
