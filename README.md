# :potato: Potato Leaf Disease Detection

In agriculture, it is often hard to detect whether a leaf has been contaminated or is healthy for produce. As such, this project aims to use a data-driven farming approach to optimize resource use, improve crop yields, and enhance overall farm management.

This project is done in fulfillment of CS180: Artificial Intelligence done by [Denise Dee](), [Denzell Dy](), and [Jose Miguel Lozada]().

## :memo: Methodology

work in progress

## :file_folder: File Structure

The repository contains the following structure:

```
.
├── .gitignore
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
│   └── potato_test                 # Folder containing images of potato leaves based on their class for testing the model
│   │   ├── Virus
│   │   └── ...
├── predictions                     # Prediction files
│   ├── pred_trad.csv               # File using the traditional approach
│   └── pred_deep.csv               # File using the deep learning approach
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

To install the dependencies, simply run the command `pip install -r requirements.txt` using the conda environment.
## :books: Code Structure

work in progress