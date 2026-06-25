import os

# Workaround for OpenMP duplicate runtime conflicts on some Windows setups.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from src.config import DEVICE, IMAGE_SIZE
from src.model import PlantDiseaseCNN


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7fbf7;
        }
        .hero {
            background: #eaf7ee;
            color: #1f5130;
            border: 1px solid #cfe8d6;
            padding: 1rem 1.1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .soft-card {
            background: white;
            border: 1px solid #d8eadc;
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.7rem;
        }
        .status-ok {
            color: #1d6b3a;
            font-weight: 600;
        }
        .status-warn {
            color: #8a5a18;
            font-weight: 600;
        }
        .small-note {
            color: #40634b;
            font-size: 0.93rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(model, class_names, image, unknown_threshold):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

    predicted_class = class_names[pred_idx]
    if confidence < unknown_threshold:
        predicted_class = "Unknown / Uncertain"
    return predicted_class, confidence, probs


def pretty_label(label):
    return label.replace("___", " - ").replace("_", " ")


def main():
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="Leaf",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">Plant Disease Detection System 🌿</h2>
            <p style="margin:0.4rem 0 0 0;">
                CNN From Scratch - Simple Leaf Health Checker
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Controls")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="models/best_model.pth",
            help="Path to trained model checkpoint file.",
        )
        unknown_threshold = st.slider(
            "Uncertain Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            help="If top confidence is below this, output is marked uncertain.",
        )
        st.caption(f"Running on: {DEVICE.upper()} | Input Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        st.markdown(
            """
            ---
            <p class="small-note">
            Tip: Use a clear close-up leaf image for better prediction.
            </p>
            """
            ,
            unsafe_allow_html=True
        )

    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    model, class_names = load_model(checkpoint_path)
    st.success("Model loaded successfully.")

    left_col, right_col = st.columns([1.1, 1.0], gap="large")

    with left_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        predicted_class, confidence, probs = predict_image(
            model, class_names, image, unknown_threshold
        )

        with left_col:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.progress(int(confidence * 100))
            st.caption(f"Model confidence meter: {confidence * 100:.2f}%")

        with right_col:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Prediction Summary")

            display_class = pretty_label(predicted_class)
            if predicted_class == "Unknown / Uncertain":
                st.markdown(
                    '<p class="status-warn">Prediction: Unknown / Uncertain</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p class="status-ok">Prediction: {display_class}</p>',
                    unsafe_allow_html=True,
                )

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Confidence", f"{confidence * 100:.2f}%")
            metric_col2.metric("Threshold", f"{unknown_threshold * 100:.0f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Top 3 Probable Classes 🌱")
            ranked = sorted(
                [
                    (pretty_label(class_names[i]), float(probs[i]) * 100)
                    for i in range(len(class_names))
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            st.table(
                [
                    {"Rank": idx + 1, "Class": cls, "Probability": f"{prob:.2f}%"}
                    for idx, (cls, prob) in enumerate(ranked)
                ]
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Class Probability Distribution")
        prob_data = {pretty_label(class_names[i]): float(probs[i]) for i in range(len(class_names))}
        st.bar_chart(prob_data)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        with right_col:
            st.info("Upload an image to view prediction insights and confidence analytics.")


if __name__ == "__main__":
    main()
