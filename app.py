import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import gdown
import os
import torch.nn.functional as F
from PIL import Image

# ----- Step 1: Download model from Google Drive -----
@st.cache_resource
def download_and_load_model():
    file_id = "1Cj3orbD3B7dHVQHNSjG04tXkAuYjodXR"  # 👈 REPLACE with your actual file ID
    model_path = "siamesemodel2.pth"

    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

    model = timm.create_model("xception", pretrained=False)

    # Modify input to grayscale
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, model.conv1.kernel_size,
                            model.conv1.stride, model.conv1.padding, bias=model.conv1.bias)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = download_and_load_model()

# ----- Step 2: Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(img1, img2, threshold=0.01):
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    with torch.no_grad():
        out1 = model.forward_features(img1)
        out2 = model.forward_features(img2)
        distance = F.pairwise_distance(out1, out2).item()

    prediction = "✔ Genuine" if distance <= threshold else "❌ Forged"
    return prediction, distance

# ----- Step 3: Streamlit UI -----
st.title("🖊 Signature Verification App")
st.markdown("Upload two signature images to check if they're from the **same person**.")

col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"])
with col2:
    img2_file = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("L")
    img2 = Image.open(img2_file).convert("L")

    st.image([img1, img2], caption=["Signature 1", "Signature 2"], width=200)

    if st.button("🔍 Verify Signature"):
        result, score = predict(img1, img2)
        st.markdown(f"### Result: {result}")
        st.markdown(f"**Similarity Score**: `{score:.4f}`")
