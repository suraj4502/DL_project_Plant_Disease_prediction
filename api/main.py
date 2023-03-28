from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf



MODEL = tf.keras.models.load_model("C:/Users/91775/DATA SCIENCE/projects/plant_disease/models/vg_final.h5")

CLASS_NAMES = [ "Pepper__bell___Bacterial_spot",
                "Pepper__bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Late_blight", 
                "Tomato_Leaf_Mold", 
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites_Two_spotted_spider_mite", 
                "Tomato__Target_Spot", 
                "Tomato__Tomato_YellowLeaf__Curl_Virus",
                "Tomato__Tomato_mosaic_virus",
                "Tomato_healthy"]


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return "HELLO this is my app"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    
    img_batch = np.expand_dims(image, 0)

    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }




if __name__=="__main__":
    uvicorn.run(app,host = 'localhost',port = 8000)