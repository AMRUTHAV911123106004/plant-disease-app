import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# -----------------------------
# Load model
# -----------------------------
model_path = os.path.join("model", "plant_disease_model.h5")
model = tf.keras.models.load_model(model_path)

# -----------------------------
# Class names (match training order)
# -----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# -----------------------------
# Treatments
# -----------------------------
treatments = {
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicides and remove infected leaves.",
    "Pepper__bell___healthy": "Plant is healthy. Maintain proper watering.",
    "Potato___Early_blight": "Apply fungicide and avoid overhead irrigation.",
    "Potato___Late_blight": "Use certified fungicides and remove infected plants.",
    "Potato___healthy": "Healthy crop. Maintain soil nutrition.",
    "Tomato___Bacterial_spot": "Remove infected leaves and use bactericides.",
    "Tomato___Early_blight": "Apply fungicides and crop rotation recommended.",
    "Tomato___Late_blight": "Use certified fungicides and remove infected plants.",
    "Tomato___Leaf_Mold": "Improve air circulation and apply fungicide.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and use fungicide sprays.",
    "Tomato___Spider_mites": "Spray miticide and maintain humidity.",
    "Tomato___Target_Spot": "Use fungicide and remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants; control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants; disinfect tools.",
    "Tomato___healthy": "Healthy tomato leaf. Maintain proper care."
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Plant Disease Detection System")

# Plant selection
plant_choice = st.selectbox(
    "Select Plant Type",
    ["Select", "Tomato", "Potato", "Pepper"]
)

if plant_choice == "Select":
    st.warning("Please select a plant type first.")
    st.stop()

# Variant/disease selection
variants = [cls for cls in class_names if plant_choice.lower() in cls.lower()]
variant_choice = st.selectbox("Select Leaf Variant/Disease", ["Auto Detect"] + variants)

# Image upload
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction / Manual Selection
    # -----------------------------
    if variant_choice == "Auto Detect":
        prediction = model.predict(img_array)[0]
        confidence = float(np.max(prediction) * 100)
        index = int(np.argmax(prediction))
        predicted_class = class_names[index]

        THRESHOLD = 35
        is_supported = plant_choice.lower() in predicted_class.lower()

        if confidence < THRESHOLD or not is_supported:
            st.warning("⚠️ Unsupported or unknown plant detected.")
            st.write("This system supports only **Tomato, Potato, and Pepper** leaves.")
            st.write("Please upload the correct plant leaf image.")
        else:
            st.success(f"✅ Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")
            if predicted_class in treatments:
                st.subheader("🌿 Recommended Treatment")
                st.write(treatments[predicted_class])
    else:
        # Manual selection
        st.success(f"✅ Selected Variant: {variant_choice}")
        st.info("Confidence: Manual Selection")
        if variant_choice in treatments:
            st.subheader("🌿 Recommended Treatment")
            st.write(treatments[variant_choice])
