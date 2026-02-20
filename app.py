import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ================= CONFIG =================
MODEL_PATH = "models/fracture_model.pth"
CLASS_NAMES = ["Mild", "Moderate", "Other", "Severe"]
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ================= GRAD-CAM =================
def generate_gradcam(model, image, class_idx):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.conv_head
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * acts, axis=0)

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam / cam.max()

    return cam

# ================= UI =================
st.title("🦴 Bone Fracture Classification")
st.write("Upload an X-ray image to classify fracture severity")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    st.success(f"Prediction: **{CLASS_NAMES[pred_class]} Fracture**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    st.subheader("Prediction Probabilities")
    prob_data = {CLASS_NAMES[i]: float(probs[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_data)

    # ===== GradCAM =====
    st.subheader("🔍 Grad-CAM Visualization")

    cam = generate_gradcam(model, image, pred_class)

    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_np

    st.image(overlay.astype(np.uint8), caption="Fracture Region Highlight")

st.warning("⚠️ This system is for educational purposes only.")
