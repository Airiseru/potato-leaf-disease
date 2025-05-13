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

import pickle
from tqdm import tqdm
import typing
from io import StringIO

# === CONSTANTS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Bacteria", "Fungi", "Healthy", "Pest", "Phytopthora", "Virus"]
svm_model_dir = "./models/svm_model_final.pkl"
cnn_model_dir = "./models"

ORIG_IMG_SIZE = (1500,1500)
BATCH_SIZE = 8
seed_value = 42

# Set seed
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)

# Load the model
with open(Path(svm_model_dir), 'rb') as file:
    svm_model = pickle.load(file)

if "model_to_use" not in st.session_state:
    st.session_state["model_to_use"] = "Traditional"

if "save_results" not in st.session_state:
    st.session_state["save_results"] = False

if "save_file_name" not in st.session_state:
    st.session_state["save_file_name"] = "results.txt"

# === FUNCTIONS ===
def load_images():
    pass

def preprocess_image_trad():
    pass

# === APPLICATION ===
st.title("Potato Leaf Disease Detection")
st.markdown("`by DeeDyL`")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["model_to_use"] = st.radio(
        "What prediction model do you want to use?",
        ["Traditional", "CNN"],
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
uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Classify", icon="🤖") and uploaded_files:
    with st.spinner("Classifying the image(s)...", show_time=True):
        # Read and load the images
        print(uploaded_files)
        st.write("wait")

        # Preprocess the images

        # Predict

    # Show the image
    
    # Output the predictions

    st.toast("Done!", icon='🎉')

    # Empty the uploader?
    uploaded_files = None
