from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

class_names = [ "Pepper__bell___Bacterial_spot",
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

BUCKET_NAME = "dlmodel20" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/vg_final.h5",
            "/tmp/vg_final.h5",
        )
        model = tf.keras.models.load_model("/tmp/vg_final.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}

