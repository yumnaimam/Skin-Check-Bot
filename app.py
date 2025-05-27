import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import random
from google.colab import files

# Upload required files
files.upload()  # Upload skin_cancer_model.keras
files.upload()  # Upload filtered_skin_cancer_with_images_and_treatments.csv and skin_cancer_patients.csv

# Load model and class labels
model = tf.keras.models.load_model("skin_cancer_model.keras")
class_names = ["benign", "malignant"]

# Load clinical data
filtered_df = pd.read_csv("filtered_skin_cancer_with_images_and_treatments.csv", low_memory=False)
patients_df = pd.read_csv("skin_cancer_patients.csv", low_memory=False)
skin_patients = patients_df[patients_df["cases_primary_site"].str.lower() == "skin"]

def analyze(image, message):
    response = ""

    if image:
        img = image.resize((180, 180))
        img_array = np.array(img)

        avg_color = np.mean(img_array, axis=(0, 1))
        std_color = np.std(img_array, axis=(0, 1))

        if avg_color[0] < 80 and avg_color[1] < 80 and avg_color[2] < 80:
            return "⚠️ This image is too dark to be a skin lesion."
        elif std_color.mean() > 60:
            return "⚠️ This image looks too complex to be a skin lesion. Please upload a clear photo of skin."

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]

        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])
        cancer_type = class_names[class_idx]

        response += f"Prediction: {cancer_type.capitalize()} ({confidence:.2%} confident)\n\n"

        if not skin_patients.empty:
            row = skin_patients.sample(1).iloc[0]
            response += "Example Patient Info:\n"
            response += f"- Project ID: {row['project_project_id']}\n"
            response += f"- Diagnosis: {row['cases_disease_type']}\n"
            response += f"- Primary Site: {row['cases_primary_site']}\n"
            if 'treatments_treatment_type' in row and pd.notna(row['treatments_treatment_type']):
                response += f"- Treatment: {row['treatments_treatment_type']}\n"

    if message:
        q = message.lower()
        if "melanoma" in q:
            response += "\nMelanoma is a serious form of skin cancer that starts in melanocytes."
        elif "benign" in q:
            response += "\nBenign means non-cancerous. These lesions usually don't spread or become harmful."
        elif "malignant" in q:
            response += "\nMalignant means cancerous. These lesions can grow, invade nearby tissues, or spread."
        elif "symptoms" in q or "signs" in q:
            response += "\nLook for changes in size, shape, or color of a mole, especially if it's asymmetric or has irregular borders."
        else:
            response += "\nI’m here to help with skin cancer questions! Try asking about melanoma, benign, or malignant."

    return response

image_input = gr.Image(type="pil", label="Upload Skin Image")
chat_input = gr.Textbox(lines=2, placeholder="Ask a skin cancer question...")
output = gr.Text(label="Result")

app = gr.Interface(
    fn=analyze,
    inputs=[image_input, chat_input],
    outputs=output,
    title="SkinCheck Bot",
    description="Upload a skin lesion image and ask skin cancer questions. This is a prototype and may give inaccurate results if used improperly."
)

app.launch()
