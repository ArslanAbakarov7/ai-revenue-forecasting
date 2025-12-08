"""
Flask API exposing /train, /predict and /log endpoints.
"""

from flask import Flask, request, jsonify
from api.data_ingest import ingest_all_jsons
from api.features_and_model import build_monthly_features, train_select_and_save, load_artifact, predict_next_month_global
from api.logger import write_log, read_log

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Revenue Forecasting API"})

@app.route("/train", methods=["POST", "GET"])
def train_endpoint():
    """Run ingestion -> features -> train pipeline and save model artifact."""
    try:
        # ingestion step assumes JSONs already in project/data
        ingest_all_jsons()
        build_monthly_features()
        model_path, maes, best = train_select_and_save()
        write_log(f"train completed model={model_path} best={best} maes={maes}")
        return jsonify({"status":"success", "model_path": model_path, "best": best, "mae": maes})
    except Exception as e:
        write_log(f"train error: {e}")
        return jsonify({"status":"error", "message": str(e)}), 500

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    """Return next-month revenue prediction; optional country filter not implemented here."""
    try:
        artifact = load_artifact()
        pred = predict_next_month_global(artifact)
        write_log(f"predict global: {pred}")
        return jsonify({"status":"success", "prediction_next_month": pred})
    except Exception as e:
        write_log(f"predict error: {e}")
        return jsonify({"status":"error", "message": str(e)}), 500

@app.route("/logfile", methods=["GET"])
def logfile_endpoint():
    return jsonify({"log": read_log()})
