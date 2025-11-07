🌱 Plant Disease Detection using Deep Learning (ResNet50 + Streamlit)

An AI-powered plant disease detection system built using ResNet50 and Streamlit, capable of identifying multiple crop diseases from leaf images with over 96% accuracy.
The system provides confidence levels, disease details, and suggested remedies, all in an interactive dashboard.

🧠 Features

✅ Trained ResNet50 CNN model on the PlantVillage dataset
✅ Achieved ~96% accuracy on test data
✅ Displays Top-3 predictions with confidence bar charts
✅ Includes disease description and remedy suggestions
✅ Streamlit-based web interface for easy, real-time predictions
✅ Automatic image enhancement (brightness & contrast) for better accuracy

| Component            | Technology                            |
| -------------------- | ------------------------------------- |
| **Model**            | ResNet50 (PyTorch)                    |
| **Frontend**         | Streamlit                             |
| **Dataset**          | PlantVillage                          |
| **Image Processing** | Pillow (PIL), TorchVision             |
| **Metrics**          | Accuracy, Precision, Recall, F1-Score |

⚙️ Setup Instructions
1️⃣ Clone this repository
git clone https://github.com/Siddharth3710/Plant-Disease-Detection-Using-CNN-Model.git
cd Plant-Disease-Detection-Using-CNN-Model

2️⃣ Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, you can install manually:

pip install torch torchvision streamlit pillow matplotlib scikit-learn

3️⃣ Run the Streamlit app
streamlit run app.py


Then open the local URL shown (usually http://localhost:8501
) in your browser 🌿

🧾 Model Performance
Metric	Value
Accuracy	95.9%
Precision	95.6%
Recall	95.2%
F1 Score	95.3%
🌾 Supported Diseases

Pepper Bell: Bacterial Spot, Healthy

Potato: Early Blight, Late Blight, Healthy

Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

🩺 Example Output

Input: Uploaded leaf image
Output:

🧠 Predicted Disease: Tomato_Late_Blight

🔍 Confidence: 94.8%

🩺 Description: Water-soaked lesions spreading fast in cool, wet weather.

🌾 Remedy: Use resistant varieties and apply metalaxyl-based fungicides.

📦 Project Structure
Plant-Disease-Detection-Using-CNN-Model/
│
├── app.py                     # Streamlit frontend
├── best_plant_disease_model.pt # Trained PyTorch model
├── requirements.txt            # Dependencies (optional)
├── README.md                   # Project documentation
└── dataset/                    # (optional) Local dataset folder

🚀 Future Improvements

📲 Mobile-friendly UI (Streamlit Cloud / HuggingFace Spaces)

📸 Real-time camera-based detection

🧩 Add more plant species

📑 Generate downloadable PDF reports of predictions

👨‍💻 Author

Siddharth Jha
🌐 GitHub Profile

💬 Passionate about AI, ML, and Deep Learning applications in agriculture 🌿

🌟 Show Your Support

If you like this project, please ⭐ star the repo — it helps a lot!
Feel free to fork, improve, and share your feedback 💚
