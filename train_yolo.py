from ultralytics import YOLO
import os

# ==============================
# 📁 DATASET PATH
# ==============================
# Change this to your dataset path
DATASET_PATH = r"D:\PlantDiseaseApp\PlantIQ\PlantDiseaseApp\backend\dataset file"

# ==============================
# 🔍 CHECK DATASET
# ==============================
if not os.path.exists(DATASET_PATH):
    raise Exception("❌ Dataset path not found")

if not os.path.exists(os.path.join(DATASET_PATH, "train")):
    raise Exception("❌ 'train' folder missing")

if not os.path.exists(os.path.join(DATASET_PATH, "val")):
    raise Exception("❌ 'val' folder missing")

print("✅ Dataset structure OK")

# ==============================
# 🧠 LOAD YOLO MODEL
# ==============================
# yolov8n-cls = classification model (fast & lightweight)
model = YOLO("yolov8n-cls.pt")

# ==============================
# 🚀 TRAIN MODEL
# ==============================
model.train(
    data=DATASET_PATH,   # dataset folder (train + val inside)
    epochs=15,           # increase for better accuracy
    imgsz=224,           # image size
    batch=16,            # reduce if RAM low
    name="plant_disease_model"
)

print("🎉 Training Completed!")

# ==============================
# 💾 MODEL SAVE LOCATION
# ==============================
print("📁 Model saved at:")
print("runs/classify/plant_disease_model/weights/best.pt")