from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load all models
models = {
    "cherry": tf.keras.models.load_model("../Models/Cherry_Training_v1"),
    "apple": tf.keras.models.load_model("../Models/Apple_Training_v1"),
    "pepper_bell": tf.keras.models.load_model("../Models/BellP_Training_v1"),
    "grape": tf.keras.models.load_model("../Models/Grape_Training_v1"),
    "peach": tf.keras.models.load_model("../Models/Peach_Training_v1"),
    "tomato": tf.keras.models.load_model("../Models/Tomato_Training_v1"),
    "potato": tf.keras.models.load_model("../Models/Potato_Training_v1"),
    "strawberry": tf.keras.models.load_model("../Models/Strawberry_Training_v1")
}

# Define class names for each model if they differ
class_names = {
    "cherry": ["Cherry_Powdery_mildew", "Cherry_healthy"],
    "apple": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"],
    "pepper_bell": ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"],
    "grape": ["Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy"],
    "peach": ["Peach___Bacterial_spot", "Peach___healthy"],
    "tomato": ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato_Target_Spot", "Tomato_Tomato_YellowLeaf__Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy"],
    "potato": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
    "strawberry": ["Strawberry___Leaf_scorch", "Strawberry___healthy"]
}

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    results = []
    for plant, model in models.items():
        predictions = model.predict(img_batch)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[plant][predicted_class_index]
        confidence = np.max(predictions[0])
        results.append({
            "plant": plant,
            "class": predicted_class,
            "confidence": float(confidence)
        })
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
