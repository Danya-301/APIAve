from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

# Cargar el modelo .keras al iniciar la app
MODEL_PATH = 'model_VGG16_v4.keras'
model = load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((224, 224))  # Ajustar tamaño según tu modelo
    image = np.array(image) / 255.0   # Normalización
    return np.expand_dims(image, axis=0)  # Añadir dimensión batch

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener clase con mayor probabilidad
        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
