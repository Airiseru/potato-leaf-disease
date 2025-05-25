import streamlit as st
import cv2
import cv2.typing as cv_typing
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from transformers import AutoImageProcessor, Dinov2Model
import torch
from torch import nn
from torchvision.io import decode_image, ImageReadMode
from torchvision import transforms
import pickle
from tqdm import tqdm
import typing
from io import StringIO
from copy import deepcopy

# === CONSTANTS ===
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["Bacteria", "Fungi", "Healthy", "Pest", "Phytopthora", "Virus"]
svm_model_dir = "./models/svm_model_final.pkl"
cnn_model_dir = "./models/dino_model_state_dict.pth"

ORIG_IMG_SIZE = (1500,1500)
RESIZE_IMG = (420, 420)
BATCH_SIZE = 8
seed_value = 42

# Set seed
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)

# === CLASSES === 
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = deepcopy(dino_model)
        # self.classifier = nn.Sequential(nn.Linear(384, 384), nn.ReLU(), nn.Linear(384, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.7), nn.ReLU(), nn.Linear(in_features=384, out_features=len(classes), bias=True))
        # self.classifier = nn.Linear(in_features=384, out_features=len(classes), bias=True)

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

color_dict = {
    0:"gray",
    1:"red",
    2:"green",
    3:"violet",
    4:"orange",
    5:"yellow"
}

# Load the model
with open(Path(svm_model_dir), 'rb') as file:
    svm_model = pickle.load(file)

transformer_model = DinoVisionTransformerClassifier()
transformer_model.load_state_dict(torch.load(cnn_model_dir, map_location='cpu'))

if "model_to_use" not in st.session_state:
    st.session_state["model_to_use"] = "Traditional"

if "save_results" not in st.session_state:
    st.session_state["save_results"] = False

if "save_file_name" not in st.session_state:
    st.session_state["save_file_name"] = "results.txt"

# === FUNCTIONS ===
def load_images():
    pass

def preprocess_image_trad(img):
    all_features = []
    batch = Image.fromarray(np.clip(Image.open(img), 0, 255).astype(np.uint8))
    # Preprocess and move to GPU
    inputs = processor(images=batch, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = dino_model(**inputs)
        features = outputs.pooler_output
        print("\tFeatures taken using  DINOv2")

    all_features.append(features.cpu().numpy())
    print("\tCompleted all batches for  DINOv2\n")

    # Combine all batches into one
    return np.vstack(all_features)

def preprocess_image_transformer(img):
    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_IMG),
        transforms.RandomHorizontalFlip(p=0.5), # Random flip with 50% probability
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    return train_transform(img)

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

    st.session_state["save_results"] = st.toggle("Save classification result")
    st.caption("This saves your classification results in one text file")

    if st.session_state["save_results"]:
        st.session_state["save_file_name"] = st.text_input("Enter filename to save as", value=st.session_state["save_file_name"])

# Main application
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_file != None:
    st.image(Image.open(uploaded_file),width=500)


if st.button("Classify", icon="🤖") and uploaded_file:
    with st.spinner("Classifying the image(s)...", show_time=True):
        # Read and load the images
        
        # Preprocess images and make predictions
        preprocessed_files = []
        predictions = []
        match st.session_state['model_to_use']:
            case 'Traditional':
                preprocessed_file = preprocess_image_trad(uploaded_file)
                predictions = svm_model.predict(preprocessed_file)
            case 'Transformer':
                preprocessed_files.append(preprocess_image_transformer(Image.open(uploaded_file)))
                predictions = torch.argmax(transformer_model(preprocessed_files), dim=1).tolist()
    # Show the image
    
    # Output the predictions
    st.header("Results")
    st.markdown(f"The model predicts that the image is from the :{color_dict[predictions[0]]}-badge[{classes[predictions[0]]}] class.")

    st.toast("Done!", icon='🎉')

    # Empty the uploader?
    uploaded_files = None
