import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import timm
import gdown
import os

# ✅ Setup page
st.set_page_config(page_title="Signature Verification", layout="centered")
st.title("✍️ Signature Verification Dashboard")
st.markdown("Upload two signature images to verify whether they are **Genuine** or **Forged**.")

# ✅ Download model from Google Drive if not present
model_path = "siamesemodel2.pth"
file_id = "1Cj3orbD3B7dHVQHNSjG04tXkAuYjodXR"  # Replace with your actual file ID
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    st.info("⏳ Downloading model from Google Drive...")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error("❌ Failed to download model. Make sure it's shared as 'Anyone with the link'")
        st.stop()

# ✅ Define the model
class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.model = timm.create_model("xception", pretrained=False)
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        with torch.no_grad():
            self.model.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model.forward_features(x)

# ✅ Load model only once using Streamlit's caching
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Prediction function
def predict(img1, img2, similarity_threshold=95.0):
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output1 = model(img1)
        output2 = model(img2)
        distance = F.pairwise_distance(output1, output2).item()
        similarity = max(0, (1 - distance / 2)) * 100
        prediction = "✔ Genuine" if similarity >= similarity_threshold else "❌ Forged"
        return similarity, prediction

# ✅ Upload images
img1 = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"])
img2 = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"])

if img1 and img2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Signature 1", use_column_width=True)
    with col2:
        st.image(img2, caption="Signature 2", use_column_width=True)

    image1 = Image.open(img1).convert("L")
    image2 = Image.open(img2).convert("L")

    if st.button("🔍 Verify"):
        similarity, result = predict(image1, image2, similarity_threshold=95.0)
        st.markdown(f"### 🔢 Similarity: `{similarity:.2f}%`")
        if result == "✔ Genuine":
            st.success("🟢 Prediction: ✔ Genuine Signature")
        else:
            st.error("🔴 Predicti

