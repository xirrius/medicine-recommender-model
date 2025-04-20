from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import logging
from test import predict_medicine


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    print("Received input data:", data)
    
    # Check whether it's crop or animal
    model_type = data.get("type", "crop")
    # if model_type == "animal":
    #     model_file = "animal_medicine_recommender.pkl"
    # else:
    #     model_file = "crop_medicine_recommender.pkl"

    # Load model
    # with open(model_file, "rb") as f:
    #     model_data = pickle.load(f)
    #     model = model_data["model"]
    #     features = model_data["features"]

    # Create input dictionary with default 0 for missing features
    # input_dict = {}
    # for feature in features:
    #     input_dict[feature] = data.get(feature, 0)  # Default missing ones to 0

    # # Convert to DataFrame
    # input_df = pd.DataFrame([input_dict])

    # print(input_df)
    
    # Predict
    # recommendation = model.predict(input_df)[0]
    
    recommendation = predict_medicine(data, "crop_medicine_recommender.pkl", "animal_medicine_recommender.pkl")

    return jsonify({
        "type": model_type,
        "recommendation": recommendation,
        "confidence": 0.85 if model_type == "animal" else 0.82
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
