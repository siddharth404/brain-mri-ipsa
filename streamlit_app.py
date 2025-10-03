# streamlit_app.py
"""
Streamlit app for Brain MRI classification + Grad-CAM visualization.
Put best_model.h5 in the same directory as this script.
Run: streamlit run streamlit_app.py
"""

import os
from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

st.set_page_config(page_title="Brain Tumor Detector (DS556)", layout="centered")

# -------------------------
# Helpers: preprocessing & Grad-CAM
# -------------------------
IMG_SIZE = (128, 128)  # (width, height) - same as training

@st.cache_resource(show_spinner=False)
def load_brain_model(model_path="best_model.h5"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Place best_model.h5 in this folder.")
        return None
    model = load_model(model_path)
    # make sure model is built (safe)
    try:
        _ = model.input
    except Exception:
        model.build(input_shape=(None, IMG_SIZE[1], IMG_SIZE[0], 1))
    return model

def preprocess_image(pil_image, target_size=IMG_SIZE):
    img = pil_image.resize(target_size)
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)  # H, W, 1
    arr = np.expand_dims(arr, axis=0)   # 1, H, W, 1
    return arr

def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        name = layer.name.lower()
        class_name = layer.__class__.__name__.lower()
        if 'conv' in name or class_name.startswith('conv'):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: (1, H, W, C)
    returns heatmap as 2D numpy array normalized to [0,1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)
    if last_conv_layer_name is None:
        raise ValueError("No conv layer found in model for Grad-CAM.")

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Weight the channels by corresponding gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.sum(conv_outputs, axis=-1)

    # Relu + normalize
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max() if heatmap.max() != 0 else 1e-10
    heatmap /= max_val
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def overlay_heatmap_on_image(pil_img_rgb, heatmap, alpha=0.4, colormap='jet'):
    """
    pil_img_rgb: PIL.Image in RGB
    heatmap: 2D array (H, W) values in [0,1]
    returns PIL.Image overlay
    """
    if pil_img_rgb.mode != 'RGB':
        pil_img_rgb = pil_img_rgb.convert('RGB')

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_img_rgb.size, resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img) / 255.0  # H,W

    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_arr)[:, :, :3]  # H,W,3 in 0..1
    colored = np.uint8(255 * colored)

    overlay = (alpha * colored + (1 - alpha) * np.array(pil_img_rgb)).astype(np.uint8)
    return Image.fromarray(overlay)

# -------------------------
# UI
# -------------------------
st.title("Brain Tumor Detector — DS556 Mini Project")
st.write("Upload a brain MRI image (jpg/png). The model predicts the class and shows Grad-CAM explainability.")

col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

with col2:
    st.write("Model & classes")
    model_file = st.text_input("Model file path", value="best_model.h5")
    # default class names - change if your label mapping differs
    default_classes = "glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor"
    class_names_input = st.text_input("Class names (comma-separated, in model order)", value=default_classes)
    CLASS_NAMES = [c.strip() for c in class_names_input.split(",") if c.strip()]

st.markdown("---")

model = load_brain_model(model_file)
if model is None:
    st.stop()

# show small model info
with st.expander("Model summary (show/hide)", expanded=False):
    st.text(model.summary())

# Quick sample images toggle
with st.expander("Tips & sample images", expanded=False):
    st.write("- Model expects grayscale MRI resized to 128x128 and normalized to [0,1].")
    st.write("- If predicted class names look wrong, update the 'Class names' input to match the LabelEncoder used during training (order matters).")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        st.stop()

    st.subheader("Uploaded Image")
    st.image(image, use_column_width=False, width=300)

    # Preprocess
    x = preprocess_image(image, target_size=IMG_SIZE)

    # Predict
    preds = model.predict(x)
    pred_idx = int(np.argmax(preds[0]))
    pred_prob = float(preds[0][pred_idx])
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"

    st.success(f"Prediction: **{pred_class}** — Probability: **{pred_prob:.4f}**")

    # Grad-CAM (wrap in spinner)
    with st.spinner("Generating Grad-CAM..."):
        try:
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=None, pred_index=pred_idx)
            overlay = overlay_heatmap_on_image(image.convert("RGB"), heatmap, alpha=0.45, colormap='jet')

            # Show side-by-side
            st.subheader("Grad-CAM overlay")
            col_a, col_b = st.columns([1,1])
            col_a.image(image.resize((300,300)), caption="Original", use_column_width=False)
            col_b.image(overlay.resize((300,300)), caption="Grad-CAM overlay", use_column_width=False)

            # Download overlay
            buf = BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download overlay (PNG)", data=buf, file_name="gradcam_overlay.png", mime="image/png")
        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

    # Show full prediction probs table
    st.subheader("Full probabilities")
    prob_df = { CLASS_NAMES[i] if i<len(CLASS_NAMES) else f"Class_{i}": float(preds[0][i]) for i in range(len(preds[0])) }
    st.table(pd.DataFrame.from_dict(prob_df, orient='index', columns=['probability']).sort_values('probability', ascending=False))

else:
    st.info("Upload an MRI image to run prediction and Grad-CAM.")
