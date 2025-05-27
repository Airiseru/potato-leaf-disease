import streamlit as st
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import torch
from torch import nn
from torchvision import transforms
import pickle
from copy import deepcopy

# === CONSTANTS ===
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["Bacteria", "Fungi", "Healthy", "Pest", "Phytopthora", "Virus"]
svm_model_dir = "./models/svm_model_final.pkl"
cnn_model_dir = "./models/dino_model_final.pth"

ORIG_IMG_SIZE = (1500,1500)
RESIZE_IMG = (420, 420)
BATCH_SIZE = 8
seed_value = 42

# Set seed
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# === CLASSES === 
color_dict = {
    0:"gray",
    1:"red",
    2:"green",
    3:"violet",
    4:"orange",
    5:"blue"
}

if "model_to_use" not in st.session_state:
    st.session_state["model_to_use"] = "Traditional"

if "save_results" not in st.session_state:
    st.session_state["save_results"] = False

if "save_file_name" not in st.session_state:
    st.session_state["save_file_name"] = "results.txt"

# DINOv2
if "dino_model" not in st.session_state:
    st.session_state["dino_model"] = Dinov2Model.from_pretrained("facebook/dinov2-large")
    st.session_state["dino_model"] = st.session_state["dino_model"].to(device)

if "dino_model_transformer" not in st.session_state:
    st.session_state["dino_model_transformer"] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

if "processor" not in st.session_state:
    st.session_state["processor"] = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)

# Load the models
if "svm_model" not in st.session_state:
    with open(Path(svm_model_dir), 'rb') as file:
        st.session_state["svm_model"] = pickle.load(file)

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(st.session_state["dino_model_transformer"])
        self.classifier = nn.Sequential(nn.Dropout(0.7), nn.ReLU(), nn.Linear(in_features=384, out_features=len(classes), bias=True))

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

if "transformer_model" not in st.session_state:
    st.session_state["transformer_model"] = DinoVisionTransformerClassifier()
    st.session_state["state_dict"] = torch.load("models/dino_model_state_dict.pth", weights_only=True, map_location="cpu")
    st.session_state["transformer_model"].load_state_dict(st.session_state["state_dict"])
    st.session_state["transformer_model"].eval() 

# === FUNCTIONS ===
def preprocess_image_trad(img):
    all_features = []
    batch = Image.fromarray(np.clip(Image.open(img), 0, 255).astype(np.uint8))

    # Preprocess and move to device
    inputs = st.session_state["processor"](images=batch, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = st.session_state["dino_model"](**inputs)
        features = outputs.pooler_output
        print("\tFeatures taken using  DINOv2")

    all_features.append(features.cpu().numpy())
    print("\tCompleted all batches for  DINOv2\n")

    # Combine all batches into one
    return np.vstack(all_features)

def preprocess_image_transformer(img):
    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_IMG),
        transforms.ToTensor()
    ])
    return train_transform(img).unsqueeze(0)

# === APPLICATION ===
st.title("Potato Leaf Disease Detection")
st.markdown("`by DeeDyL`")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["model_to_use"] = st.radio(
        "What prediction model do you want to use?",
        ["Traditional", "Transformer"],
        captions=[
            "Just some regular old SVM",
            "Deep learning magic",
        ],
    )

# Main application
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_file != None:
    # Open image
    st.image(Image.open(uploaded_file),width=500)


if st.button("Classify", icon="🤖") and uploaded_file:
    with st.spinner("Classifying the image(s)...", show_time=True):
        # Preprocess images and make predictions
        predictions = []
        match st.session_state['model_to_use']:
            case 'Traditional':
                preprocessed_file = preprocess_image_trad(uploaded_file)
                predictions = st.session_state["svm_model"].predict(preprocessed_file)
            case 'Transformer':
                preprocessed_file = preprocess_image_transformer(Image.open(uploaded_file))
                output = st.session_state["transformer_model"](preprocessed_file)
                predictions = output.argmax(dim=1).tolist()
    
    # Output the predictions
    st.header("Results")
    st.markdown(f"The model predicts that the image is from the **:{color_dict[predictions[0]]}-badge[{classes[predictions[0]]}]** class.")

    st.toast("Done!", icon='🎉')

    # Empty the uploader?
    uploaded_files = None
