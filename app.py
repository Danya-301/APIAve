from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np

app = FastAPI()

# Cargar el modelo al iniciar la app
MODEL_PATH = 'model_VGG16_v4.keras'
model = load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesar la imagen para el modelo."""
    image = image.resize((224, 224))  # Ajustar tamaño
    image = np.array(image) / 255.0   # Normalización a [0, 1]
    if image.shape[-1] != 3:  # Asegurar que la imagen tenga 3 canales (RGB)
        raise ValueError("La imagen debe ser RGB con 3 canales.")
    return np.expand_dims(image, axis=0)  # Añadir dimensión batch

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Verificar tipo MIME del archivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
        
        # Intentar abrir la imagen
        try:
            image = Image.open(file.file).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="No se pudo identificar la imagen.")

        # Preprocesar y predecir
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction, axis=1)[0])  # Clase con mayor probabilidad
        
        return {"predicted_class": predicted_class}

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")
    finally:
        file.file.close()  # Asegurar cierre del archivo
