# # """
# # DeepLung AI — Flask inference backend.

# # Endpoints:
# #     POST /predict   — accepts multipart 'image' field, returns predicted class + probabilities.
# #     GET  /health    — health check.

# # The DenseNet121 model is intentionally NOT included. Place your trained
# # `.h5` / `.keras` weights at `model/weights/densenet121_pneumonia.h5` and
# # uncomment the loader below.
# # """
# # from __future__ import annotations
# # from flask import Flask, render_template
# # import tensorflow as tf
# # import spaces
# # from huggingface_hub import hf_hub_download
# # from tensorflow.keras.models import load_model as keras_load_model
# # import io
# # import os
# # import cv2
# # from typing import Any
# # import gradio as gr 
# # from flask import Flask, jsonify, request
# # from flask_cors import CORS
# # from PIL import Image
# # import numpy as np

# # app = Flask(__name__)
# # CORS(app)  # Allow requests from the React frontend

# # CLASS_NAMES = ["covid19", "normal","pneumonia"]
# # IMG_SIZE = (224, 224)

# # MODEL_PATH = hf_hub_download(
# #     repo_id="iqrakhawar/chest_xray_deep_learning",
# #     filename="10-Final_chest_xray_AI_model.keras"
# # )

# # model = tf.keras.models.load_model(MODEL_PATH,compile=False)
# # # ------------------------------------------------------------------
# # # Model loading — leave empty until you drop trained weights in place.
# # # ------------------------------------------------------------------


# # # _model: Any = None


# # # _model = None

# # # def load_model():
# # #     global _model

# # #     if _model is not None:
# # #         return _model

# # #     try:
# # #         model_path = hf_hub_download(
# # #             repo_id=HF_REPO_ID,
# # #             filename=HF_FILENAME
# # #         )

# # #         _model = keras_load_model(
# # #             model_path,
# # #             compile=False
# # #         )
# # #         print(_model.input_shape)
      
# # #         app.logger.info("Chest X-ray model loaded successfully.")

# # #     except Exception as e:
# # #         app.logger.error(f"Error loading model: {e}")
# # #         return None

# # #     return _model

# # severity_map = {

# #     "normal": {
# #         "severity": "No Disease",
# #         "risk": "Low",
# #         "follow_up": (
# #             "The chest X-ray does not show any significant signs of pneumonia or COVID-19 infection. "
# #             "Maintain a healthy lifestyle, stay hydrated, and continue practicing good respiratory hygiene. "
# #             "If symptoms such as persistent cough, fever, chest pain, or difficulty breathing develop or worsen, "
# #             "consult a healthcare professional for further clinical evaluation. Routine medical follow-up is "
# #             "recommended only if symptoms persist or new respiratory complaints arise."
# #         )
# #     },

# #     "pneumonia": {
# #         "severity": "Moderate",
# #         "risk": "Medium",
# #         "follow_up": (
# #             "The chest X-ray findings are suggestive of pneumonia. It is recommended to consult a physician or "
# #             "pulmonologist as soon as possible for a detailed clinical assessment and confirmation of the diagnosis. "
# #             "Additional investigations such as blood tests, sputum culture, or repeat chest imaging may be required "
# #             "depending on the patient's condition. Follow the prescribed treatment plan, take medications exactly "
# #             "as directed, ensure adequate rest and hydration, and seek immediate medical attention if symptoms such "
# #             "as high fever, severe chest pain, worsening cough, or shortness of breath become more severe."
# #         )
# #     },

# #     "covid19": {
# #         "severity": "Critical",
# #         "risk": "High",
# #         "follow_up": (
# #             "The chest X-ray findings are highly suggestive of COVID-19-related lung involvement. Immediate medical "
# #             "evaluation is strongly recommended. The patient should promptly consult a qualified healthcare provider "
# #             "or visit the nearest hospital for further assessment and appropriate management. Additional laboratory "
# #             "tests, oxygen saturation monitoring, and confirmatory diagnostic testing may be required. If symptoms "
# #             "such as severe breathing difficulty, persistent chest pain, confusion, bluish lips or face, or low "
# #             "oxygen saturation are present, emergency medical care should be sought without delay. Follow all medical "
# #             "advice, isolation guidelines, and treatment recommendations provided by healthcare professionals."
# #         )
# #     }

# # }

# # def preprocess(img):
    
# #     img = img.convert("L")

# #     img = img.resize((224,224))

# #     img = np.array(img)

# #     img = img.astype(np.float32) / 255.0

# #     img = np.expand_dims(img, axis=-1)

# #     img = np.expand_dims(img, axis=0)

# #     print(img.shape)

# #     return img

# # @app.route('/')
# # def home():
# #     return render_template('test.html')

# # @app.get("/health")
# # def health():
# #     return jsonify({"status": "ok", "model_loaded": load_model() is not None})


# # # @app.post("/predict")
# # # def predict():
 
    
# # #     if "image" not in request.files:
# # #         return jsonify({"error": "No 'image' file provided"}), 400

# # #     file = request.files["image"]

# # #     try:
# # #         img = Image.open(io.BytesIO(file.read()))
# # #     except Exception as exc:
# # #         return jsonify({"error": f"Invalid image: {exc}"}), 400

# # #     model = load_model()

# # #     if model is None:
# # #         return jsonify({"error": "Model could not be loaded."}), 500

# # #     x = preprocess(img)

# # #     probs = model.predict(x, verbose=0)[0]

# # #     print("PROBS:", probs)
# # #     print("INPUT MIN:", x.min())
# # #     print("INPUT MAX:", x.max())
# # #     print("INPUT SHAPE:", x.shape)
# # #     print("CLASS_NAMES:", CLASS_NAMES)
# # #     print("PROBS:", probs)
# # #     idx = int(np.argmax(probs))

# # #     label = CLASS_NAMES[idx]

# # #     confidence = float(probs[idx] * 100)

# # #     info = severity_map[label]

# # #     return jsonify({

# # #         "prediction": label,

# # #         "confidence": round(confidence, 2),

# # #         "estimated_severity": info["severity"],

# # #         "risk_level": info["risk"],

# # #         "follow_up": info["follow_up"],

# # #         "probabilities": {

# # #             name: round(float(p) * 100, 2)

# # #             for name, p in zip(CLASS_NAMES, probs)

# # #         }

# # #     })
# # # load_model()
# # spaces.GPU
# # def predict(image):
    
# #     x = preprocess(image)


# #     probs = model.predict(
# #         x,
# #         verbose=0
# #     )[0]


# #     idx = int(np.argmax(probs))


# #     label = CLASS_NAMES[idx]


# #     info = severity_map[label]


# #     result = {
# #         "prediction": label,
# #         "confidence": round(float(probs[idx])*100,2),
# #         "estimated_severity": info["severity"],
# #         "risk_level": info["risk"],
# #         "follow_up": info["follow_up"],

# #         "probabilities":{
# #             CLASS_NAMES[i]:
# #             round(float(probs[i])*100,2)
# #             for i in range(3)
# #         }
# #     }


# #     return result



# # demo = gr.Interface(

# #     fn=predict,

# #     inputs=gr.Image(
# #         type="pil",
# #         label="Upload Chest X-Ray"
# #     ),

# #     outputs=gr.JSON(
# #         label="DeepLung AI Result"
# #     ),

# #     title="DeepLung AI",

# #     description=
# #     "AI based chest X-ray analysis using DenseNet121"
# # )


# # demo.launch()

# # # if __name__ == "__main__":
# # #     app.run(host="0.0.0.0", port=5000, debug=True)
# import gradio as gr
# import tensorflow as tf
# from huggingface_hub import hf_hub_download
# from PIL import Image
# import numpy as np

# CLASS_NAMES = ["covid19", "normal", "pneumonia"]

# MODEL_PATH = hf_hub_download(
#     repo_id="iqrakhawar/chest_xray_deep_learning",
#     filename="10-Final_chest_xray_AI_model.keras"
# )

# model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# severity_map = {
#     "normal": {
#         "severity": "No Disease",
#         "risk": "Low",
#         "follow_up": (
#            "The chest X-ray does not show any significant signs of pneumonia or COVID-19 infection. "
#             "Maintain a healthy lifestyle, stay hydrated, and continue practicing good respiratory hygiene. "
#             "If symptoms such as persistent cough, fever, chest pain, or difficulty breathing develop or worsen, "
#             "consult a healthcare professional for further clinical evaluation. Routine medical follow-up is "
#             "recommended only if symptoms persist or new respiratory complaints arise."
#         ),
#     },
#     "pneumonia": {
#         "severity": "Moderate",
#         "risk": "Medium",
#         "follow_up": (
#            "The chest X-ray findings are suggestive of pneumonia. It is recommended to consult a physician or "
#             "pulmonologist as soon as possible for a detailed clinical assessment and confirmation of the diagnosis. "
#             "Additional investigations such as blood tests, sputum culture, or repeat chest imaging may be required "
#             "depending on the patient's condition. Follow the prescribed treatment plan, take medications exactly "
#             "as directed, ensure adequate rest and hydration, and seek immediate medical attention if symptoms such "
#             "as high fever, severe chest pain, worsening cough, or shortness of breath become more severe."
#         ),
#     },
#     "covid19": {
#         "severity": "Critical",
#         "risk": "High",
#         "follow_up": (
#            "The chest X-ray findings are highly suggestive of COVID-19-related lung involvement. Immediate medical "
#             "evaluation is strongly recommended. The patient should promptly consult a qualified healthcare provider "
#             "or visit the nearest hospital for further assessment and appropriate management. Additional laboratory "
#             "tests, oxygen saturation monitoring, and confirmatory diagnostic testing may be required. If symptoms "
#             "such as severe breathing difficulty, persistent chest pain, confusion, bluish lips or face, or low "
#             "oxygen saturation are present, emergency medical care should be sought without delay. Follow all medical "
#             "advice, isolation guidelines, and treatment recommendations provided by healthcare professionals."
#         ),
#     },
# }


# def preprocess(image: Image.Image):
#     image = image.convert("RGB")          # 3 channels
#     image = image.resize((224, 224))
#     image = np.array(image).astype("float32") / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image


# def predict(image):
#     x = preprocess(image)

#     probs = model.predict(x, verbose=0)[0]

#     idx = int(np.argmax(probs))
#     label = CLASS_NAMES[idx]

#     info = severity_map[label]

#     return {
#         "prediction": label,
#         "confidence": round(float(probs[idx]) * 100, 2),
#         "estimated_severity": info["severity"],
#         "risk_level": info["risk"],
#         "follow_up": info["follow_up"],
#         "probabilities": {
#             CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
#             for i in range(len(CLASS_NAMES))
#         },
#     }


# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
#     outputs=gr.JSON(label="Prediction"),
#     title="DeepLung AI",
#     description="Chest X-ray classification using DenseNet121",
# )

# if __name__ == "__main__":
#     demo.launch()

import gradio as gr
import spaces
import tensorflow as tf
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download


# -----------------------------
# Classes
# -----------------------------

CLASS_NAMES = [
    "covid19",
    "normal",
    "pneumonia"
]


# -----------------------------
# Download Model
# -----------------------------

MODEL_PATH = hf_hub_download(
    repo_id="iqrakhawar/chest_xray_deep_learning",
    filename="10-Final_chest_xray_AI_model.keras"
)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("MODEL LOADED")
print("INPUT SHAPE:", model.input_shape)


# -----------------------------
# Severity Information
# -----------------------------

severity_map = {

    "normal": {
        "severity": "No Disease",
        "risk": "Low",
        "follow_up":  "The chest X-ray does not show any significant signs of pneumonia or COVID-19 infection. "
            "Maintain a healthy lifestyle, stay hydrated, and continue practicing good respiratory hygiene. "
            "If symptoms such as persistent cough, fever, chest pain, or difficulty breathing develop or worsen, "
            "consult a healthcare professional for further clinical evaluation. Routine medical follow-up is "
            "recommended only if symptoms persist or new respiratory complaints arise."
    },

    "pneumonia": {
        "severity": "Moderate",
        "risk": "Medium",
        "follow_up":            "The chest X-ray findings are suggestive of pneumonia. It is recommended to consult a physician or "
            "pulmonologist as soon as possible for a detailed clinical assessment and confirmation of the diagnosis. "
            "Additional investigations such as blood tests, sputum culture, or repeat chest imaging may be required "
            "depending on the patient's condition. Follow the prescribed treatment plan, take medications exactly "
            "as directed, ensure adequate rest and hydration, and seek immediate medical attention if symptoms such "
            "as high fever, severe chest pain, worsening cough, or shortness of breath become more severe."
    },

    "covid19": {
        "severity": "Critical",
        "risk": "High",
        "follow_up": "The chest X-ray findings are highly suggestive of COVID-19-related lung involvement. Immediate medical "
            "evaluation is strongly recommended. The patient should promptly consult a qualified healthcare provider "
            "or visit the nearest hospital for further assessment and appropriate management. Additional laboratory "
            "tests, oxygen saturation monitoring, and confirmatory diagnostic testing may be required. If symptoms "
            "such as severe breathing difficulty, persistent chest pain, confusion, bluish lips or face, or low "
            "oxygen saturation are present, emergency medical care should be sought without delay. Follow all medical "
            "advice, isolation guidelines, and treatment recommendations provided by healthcare professionals."
    }
}


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess(image):

    image = image.convert("RGB")

    image = image.resize(
        (224,224)
    )

    image = np.array(image).astype(
        "float32"
    ) / 255.0


    image = np.expand_dims(
        image,
        axis=0
    )


    print(
        "INPUT:",
        image.shape
    )

    return image



# -----------------------------
# Prediction
# -----------------------------

@spaces.GPU
def predict(image):

    x = preprocess(image)


    probs = model.predict(
        x,
        verbose=0
    )[0]


    idx = int(
        np.argmax(probs)
    )


    label = CLASS_NAMES[idx]


    info = severity_map[label]


    return {

        "prediction": label,

        "confidence": round(
            float(probs[idx])*100,
            2
        ),

        "estimated_severity":
            info["severity"],

        "risk_level":
            info["risk"],

        "follow_up":
            info["follow_up"],


        "probabilities": {

            CLASS_NAMES[i]:
            round(float(probs[i])*100,2)

            for i in range(
                len(CLASS_NAMES)
            )
        }
    }



# -----------------------------
# Gradio App
# -----------------------------

demo = gr.Interface(

    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload Chest X-Ray"
    ),

    outputs=gr.JSON(
        label="DeepLung AI Result"
    ),

    title="DeepLung AI",

    description=
    "Chest X-ray classification using DenseNet121"
)


demo.launch()