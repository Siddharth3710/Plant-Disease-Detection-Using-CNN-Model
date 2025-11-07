import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# -----------------------------------
# 1️⃣ App Config
# -----------------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")
st.title("🌱 Plant Disease Detection Dashboard")
st.write("Upload a plant leaf image to identify possible diseases and view remedies.")

# -----------------------------------
# 2️⃣ Class Labels
# -----------------------------------
classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# -----------------------------------
# 3️⃣ Disease Info
# -----------------------------------
disease_info = {
    'Pepper__bell___Bacterial_spot': {
        'description': 'Dark lesions on leaves and fruit caused by *Xanthomonas campestris*.',
        'remedy': 'Use copper-based fungicides, avoid overhead watering, and rotate crops.'
    },
    'Potato___Early_blight': {
        'description': 'Dark brown concentric spots on leaves caused by *Alternaria solani*.',
        'remedy': 'Apply chlorothalonil fungicide and remove infected debris.'
    },
    'Potato___Late_blight': {
        'description': 'Water-soaked lesions spreading fast in cool, wet weather.',
        'remedy': 'Use resistant varieties and apply metalaxyl-based fungicides.'
    },
    'Tomato_Bacterial_spot': {
        'description': 'Small brown lesions on leaves and fruits.',
        'remedy': 'Use pathogen-free seeds and ensure good air circulation.'
    },
    'Tomato_healthy': {
        'description': 'No disease detected.',
        'remedy': 'Maintain balanced watering and fertilization.'
    },
}

# -----------------------------------
# 4️⃣ Load Model
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
num_classes = len(classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load("best_plant_disease_model.pt", map_location=device))
model.eval()

# -----------------------------------
# 5️⃣ Image Transform
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------
# 6️⃣ Upload + Prediction
# -----------------------------------
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    filename = uploaded_file.name.strip().lower()
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        st.error("❌ Invalid file type. Please upload a JPG or PNG image.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            # 🔆 Enhance image
            image = ImageEnhance.Contrast(image).enhance(1.2)
            image = ImageEnhance.Brightness(image).enhance(1.1)
            st.image(image, caption="Uploaded Image (Enhanced)", use_column_width=True)

        with col2:
            st.write("### 🔎 Analyzing...")
            img_t = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Get top-3 predictions
            top3_prob, top3_idx = torch.topk(probs, 3)
            top3_prob = top3_prob.cpu().numpy() * 100
            top3_classes = [classes[i] for i in top3_idx.cpu().numpy()]

            # 🧠 Show Top Prediction
            st.success(f"**Predicted Disease:** {top3_classes[0]}")
            st.info(f"**Confidence:** {top3_prob[0]:.2f}%")

            # 📊 Confidence Chart
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(top3_classes[::-1], top3_prob[::-1], color='seagreen')
            ax.set_xlabel('Confidence (%)')
            ax.set_xlim(0, 100)
            st.pyplot(fig)

            # 🩺 Disease Info
            top_pred = top3_classes[0]
            if top_pred in disease_info:
                info = disease_info[top_pred]
                st.write(f"### 🩺 Description\n{info['description']}")
                st.write(f"### 🌾 Suggested Remedy\n{info['remedy']}")
            else:
                st.warning("ℹ️ No extra information available for this disease.")
