import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import json

# --------------------
# App Configuration
# --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------
# Load Model & Class Map
# --------------------
try:
    model = load_model("vegetable_classifier.h5", compile=False)
except:
    model = None
    print("Warning: Model file not found!")

# Create sample class map if it doesn't exist
if os.path.exists("class_map.pkl"):
    with open("class_map.pkl", "rb") as f:
        class_map = pickle.load(f)
else:
    # Sample vegetables for demonstration
    class_map = {
        0: "Tomato",
        1: "Carrot",
        2: "Broccoli",
        3: "Potato",
        4: "Cabbage",
        5: "Cucumber",
        6: "Bell Pepper",
        7: "Spinach",
        8: "Pumpkin",
        9: "Lettuce"
    }
    # Save it for future use
    with open("class_map.pkl", "wb") as f:
        pickle.dump(class_map, f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------
# Routes
# --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("prediction.html", 
                             result="No image uploaded", 
                             error=True,
                             image_path=None)

    file = request.files["image"]
    if file.filename == "":
        return render_template("prediction.html", 
                             result="No image selected", 
                             error=True,
                             image_path=None)

    if not allowed_file(file.filename):
        return render_template("prediction.html", 
                             result="Only PNG, JPG, JPEG, and GIF files are allowed", 
                             error=True,
                             image_path=None)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Image preprocessing
        img = tf.keras.utils.load_img(filepath, target_size=(150, 150))
        img_arr = tf.keras.utils.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        # Prediction
        if model:
            prediction = model.predict(img_arr, verbose=0)
            pred_index = np.argmax(prediction)
            confidence = float(prediction[0][pred_index]) * 100
            result = class_map.get(pred_index, "Unknown")
            
            return render_template("prediction.html", 
                                 result=result,
                                 confidence=f"{confidence:.2f}",
                                 image_url=f"/uploads/{filename}",
                                 error=False)
        else:
            return render_template("prediction.html", 
                                 result="Model not available", 
                                 error=True,
                                 image_path=None)
    
    except Exception as e:
        return render_template("prediction.html", 
                             result=f"Error processing image: {str(e)}", 
                             error=True,
                             image_path=None)


@app.route("/logout")
def logout():
    return render_template("logout.html")


# --------------------
# Run App
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
