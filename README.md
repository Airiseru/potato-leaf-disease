# :potato: Potato Leaf Disease Detection

In agriculture, it is often hard to detect whether a leaf has been contaminated or is healthy for produce. As such, this project aims to use a data-driven farming approach to optimize resource use, improve crop yields, and enhance overall farm management.

This project is done in fulfillment of CS180: Artificial Intelligence done by [Denise Dee](https://github.com/Airiseru), [Denzell Dy](https://github.com/DenzDy), and [Jose Miguel Lozada](https://github.com/jslozada1221).

## :memo: Methodology

For both files, the methodology is simple:
1. Load images
2. Preprocess the images
3. Hypertune the model
4. Evaluate the model

The specific methodologies implemented are further explained in the development notebooks.

## :file_folder: File Structure

The repository contains the following structure:

```
.
├── .gitignore
├── app.py                          # File containing the code for the streamlit web application
├── requirements.txt
├── notebooks
│   ├── develop_trad.ipynb          # Notebook conaining the development code using traditional approaches
│   ├── develop_deep.ipynb          # Notebook conaining the development code using deep learning
│   ├── demo_trad.ipynb             # Notebook conaining the demo code using traditional approaches
│   └── demo_deep.ipynb             # Notebook conaining the demo code using deep learning
├── data                            # Datasets
│   ├── potato_train                # Folder containing images of potato leaves based on their class for training the model
│   │   ├── Virus
│   │   ├── Phytopthora
│   │   ├── Pest
│   │   ├── Healthy
│   │   ├── Fungi
│   │   └── Bacteria
│   └── potato_test                 # Folder containing images of potato leaves for testing the model
├── predictions                     # Prediction files
│   ├── pred_trad.csv               # File using the traditional approach
│   └── pred_deep.csv               # File using the deep learning approach
├── models                          # Final model files for prediction
│   ├── trad_model.pkl              # Pickle file for traditional model
│   └── deep_model.pkl              # Pickle file for deep learning model
└── README.md
```

Note that due to the size of the datasets, they will not be included in the repository.

## :computer: Development

This section contains the development pre-requisites to ensure that all the codes work as intended.

### :green_book: Conda Environment

The project will use an Anaconda or Miniconda Ecosystem which can be installed through their [website](https://www.anaconda.com/download). To create a conda virtual environment, use the following commands:

```bash
# Create a new conda environment
conda create --name cs180-proj python=3.12 -y

# Activate the new environment
conda activate cs180-proj

# Install Jupyter and the Kernel
pip install jupyter
ipython kernel install --name "cs180-proj" --display-name "Python 3.12 (CS180 Project)"
```

### :snake: Python Dependencies

To install the dependencies, simply run the command `pip install -r requirements.txt` using the conda environment. Note, that it is recommended to have [NVIDIA CUDA](https://docs.nvidia.com/cuda/) and [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) installed, if possible, for faster processing and model training.

## :books: Code Structure

For the development notebooks, the structure of the code is simple. It will first contain an in-depth explanation of the methodology that will be attempted for that notebook. It will then proceed with a miscellaneous section which simply contains the `pip install` and the ignore warnings. The libraries used will then be imported and the global constants will be defined. Afterwards, the notebook will proceed with the actual implementation of the methodology. This mainly includes the preprocessing the images, training the model, and evaluating the model.

## :iphone: Web Application

To run the streamlit web application locally, simply run the command `streamlit run app.py` in the command prompt.