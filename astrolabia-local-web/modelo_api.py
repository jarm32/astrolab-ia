from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import traceback
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"

@app.route('/status', methods=['GET'])
def status():
    """Permite verificar si el modelo se cargó correctamente"""
    exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "modelo_encontrado": exists,
        "ruta": os.path.abspath(MODEL_PATH)
    })

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    traceback.print_exc()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "No se pudo cargar el modelo en el servidor.",
            "detalle": "Verifica que el archivo .pkl exista y sea compatible con joblib/sklearn."
        }), 500

    try:
        data = request.get_json(force=True)

        X = np.array([[
            data.get("pl_radio", 0),
            data.get("profundidad", 0),
            data.get("periodo_orbital", 0),
            data.get("insolacion", 0),
            data.get("duracion_transito", 0),
            data.get("st_radio", 0),
            data.get("st_temperatura", 0),
            data.get("pl_temperatura_eq", 0),
            data.get("st_gravedad", 0)
        ]])

        # Intentar predicción según capacidades del modelo
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X)
            pred = float(y_pred[0][1] * 100)
        elif hasattr(model, "predict"):
            y_pred = model.predict(X)
            pred = float(y_pred[0])
            if 0 <= pred <= 1:
                pred *= 100
        else:
            raise AttributeError("El modelo no tiene método predict ni predict_proba")

        return jsonify({"prediccion": pred})

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(debug=True)