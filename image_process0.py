import numpy as np
import cv2
from keras.models import load_model
from io import BytesIO
from info0 import *
from rembg import remove
import copy
import time

# from tensorflow.keras.applications.resnet50 import ResNet50
leafornot_model = load_model(
    "C:/Users/phadn/OneDrive/Desktop/Virtual_envn_project/MainProject/model/LeafOrNot.h5",
    compile=False,
)

disease_model = load_model(
    "C:/Users/phadn/OneDrive/Desktop/Virtual_envn_project/MainProject/model/keras2.h5",
    compile=False,
)
np.set_printoptions(suppress=True)


def image_processing(file):
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Remove the background using rembg
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_copy = copy.copy(img)
    x = np.array([img / 255])
    prediction = leafornot_model.predict(x)
    y = np.argmax(prediction)
    global response
    response = " "
    if y == 1:
        response = "Sorry, the image is not a leaf."
        filename = (
            "C:/Users/phadn/OneDrive/Desktop/Virtual_envn_project/MainProject/Images/notleaf/notleaf_"
            + str(int(time.time()))
            + ".jpg"
        )
        cv2.imwrite(filename, img_copy)
    else:
        # Classify the disease if the image is a leaf using the disease_model
        output_img_rgba = remove(img_copy)
        img = cv2.cvtColor(output_img_rgba, cv2.COLOR_RGBA2BGR)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_copy_seg = copy.copy(img)
        filename = (
            "C:/Users/phadn/OneDrive/Desktop/Virtual_envn_project/MainProject/Images/leaf/leaf_"
            + str(int(time.time()))
            + ".jpg"
        )
        cv2.imwrite(filename, img_copy_seg)
        x = np.array([img / 255])
        prediction = disease_model.predict(x)
        y = np.argmax(prediction)
        global english_response
        english_response = " "
        english_response = (
            f"Disease Predicted: {disease_english[y]}\n"
            f"\n"
            f"Symptons: {symptoms_english[y]}\n"
            f"\n"
            f"Cure: {cure_english[y]}\n"
            f"\n"
        )


def get_response_english():
    if response == "Sorry, the image is not a leaf.":
        return response
    else:
        return english_response
