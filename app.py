from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

# 🔥 Load YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🌿 Disease Info (ADD YOUR FULL DATA HERE)
disease_info = {
    "Chilli_LeafCurl": {
        "description": "Leaf curl disease causes curling and yellowing of leaves.",
        "treatment": "Spray neem oil weekly.",
        "fertilizer": "Apply balanced NPK fertilizer.",
        "tamil": "இலை சுருங்கும் நோய்"
    },
    "Chilli_Healthy": {
        "description": "Plant is healthy.",
        "treatment": "No treatment needed.",
        "fertilizer": "Maintain regular nutrients.",
        "tamil": "ஆரோக்கியமான மிளகாய்"
    },
    "Tomato_LateBlight": {
        "description": "Late blight causes dark spots and rotting.",
        "treatment": "Use fungicide spray.",
        "fertilizer": "Use potassium-rich fertilizer.",
        "tamil": "தக்காளி பிந்தைய அழுகல்"
    },
    "Tomato_Healthy": {
        "description": "Plant is healthy.",
        "treatment": "No treatment needed.",
        "fertilizer": "Maintain nutrients.",
        "tamil": "ஆரோக்கியமான தக்காளி"
    },
    "Groundnut_Healthy": {
        "description": "Plant is healthy.",
        "treatment": "No treatment needed.",
        "fertilizer": "Maintain nutrients.",
        "tamil": "ஆரோக்கியமான நிலக்கடலை"
    }
}

@app.route("/")
def home():
    return "🌱 PlantIQ API Running..."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files["image"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 🔥 YOLO prediction
        results = model(filepath)

        probs = results[0].probs
        names = results[0].names

        prediction = names[probs.top1]
        confidence = float(probs.top1conf)

        # 🔥 Get disease info
        info = disease_info.get(prediction, {})

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "description": info.get("description", ""),
            "treatment": info.get("treatment", ""),
            "fertilizer": info.get("fertilizer", ""),
            "tamil": info.get("tamil", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
