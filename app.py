from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

# 🔥 Load model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🌿 ENHANCED PLANT DISEASE DATABASE
disease_info = {
    "Chilli_LeafCurl": {
        "description": "Leaf Curl disease is caused by viruses transmitted by whiteflies. It leads to curling, yellowing, and stunted growth of chilli leaves.",
        "treatment": "1. Remove infected leaves immediately.\n2. Spray neem oil every 5–7 days.\n3. Use yellow sticky traps for whiteflies.\n4. Apply recommended viral control pesticide if severe.",
        "fertilizer": "Use balanced NPK (10:10:10) fertilizer. Add micronutrients like Zinc and Magnesium for recovery.",
        "routine": "🌱 Daily: Inspect leaves for pests\n🌿 Weekly: Spray neem oil\n💧 Irrigation: Keep soil moist but not waterlogged",
        "tamil": "இலை சுருங்கும் நோய் - வெள்ளை ஈ தாக்கத்தால் ஏற்படும் வைரஸ் நோய்"
    },

    "Chilli_Healthy": {
        "description": "The chilli plant is healthy with no visible disease symptoms.",
        "treatment": "No treatment required. Maintain good hygiene in the field.",
        "fertilizer": "Continue regular NPK fertilization every 15 days.",
        "routine": "🌱 Daily: Check soil moisture\n🌿 Weekly: Inspect leaf underside\n💧 Watering: 2–3 times per week depending on climate",
        "tamil": "ஆரோக்கியமான மிளகாய் செடி"
    },

    "Tomato_LateBlight": {
        "description": "Late Blight is a serious fungal disease causing dark spots, leaf rot, and fruit decay in tomato plants.",
        "treatment": "1. Spray copper-based fungicide every 7 days.\n2. Remove infected leaves immediately.\n3. Avoid overhead watering.\n4. Improve air circulation between plants.",
        "fertilizer": "Apply potassium-rich fertilizer (0-0-50). Add calcium nitrate for fruit strength.",
        "routine": "🌱 Daily: Check for dark spots\n🌿 Weekly: Fungicide spray\n💧 Water early morning only\n🌬 Ensure good spacing between plants",
        "tamil": "தக்காளி பிந்தைய அழுகல் நோய்"
    },

    "Tomato_Healthy": {
        "description": "Tomato plant is healthy and growing well without disease symptoms.",
        "treatment": "No treatment required. Maintain preventive care.",
        "fertilizer": "Apply NPK (20:20:20) every 2 weeks.",
        "routine": "🌱 Daily: Observe leaf color\n🌿 Weekly: Check for pests\n💧 Water 2–3 times per week",
        "tamil": "ஆரோக்கியமான தக்காளி செடி"
    },

    "Groundnut_Healthy": {
        "description": "Groundnut plant is healthy with good leaf and root development.",
        "treatment": "No treatment required. Maintain field hygiene.",
        "fertilizer": "Use phosphorus-rich fertilizer for pod development.",
        "routine": "🌱 Daily: Soil check\n🌿 Weekly: Pest inspection\n💧 Moderate irrigation required",
        "tamil": "ஆரோக்கியமான நிலக்கடலை"
    }
}

@app.route("/")
def home():
    return "🌱 PlantIQ AI API Running..."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        if not os.path.exists(filepath):
            return jsonify({"error": "File upload failed"}), 500

        # 🔥 YOLO prediction
        results = model(filepath)
        r = results[0]

        if r.probs is None:
            return jsonify({"error": "No prediction returned from model"}), 500

        prediction_index = int(r.probs.top1)
        confidence = float(r.probs.top1conf)
        prediction = r.names[prediction_index]

        info = disease_info.get(prediction, {
            "description": "No data available",
            "treatment": "No data available",
            "fertilizer": "No data available",
            "routine": "No data available",
            "tamil": ""
        })

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "description": info["description"],
            "treatment": info["treatment"],
            "fertilizer": info["fertilizer"],
            "routine": info["routine"],
            "tamil": info["tamil"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
