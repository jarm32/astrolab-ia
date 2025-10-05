from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # llamadas desde localhost:8000

model_path = "model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            "pl_radio", "profundidad", "periodo_orbital", "insolacion",
            "duracion_transito", "st_radio", "st_temperatura",
            "pl_temperatura_eq", "st_gravedad"
        ]

        values = [float(data.get(f, 0)) for f in features]
        X = np.array(values).reshape(1, -1)

        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500

        prediction = model.predict(X)
        result = float(prediction[0])

        return jsonify({"prediccion": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)