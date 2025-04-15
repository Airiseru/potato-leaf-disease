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

The project will use [uv](https://docs.astral.sh/uv/), which is a Python package and project manager. To install uv, check their [installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more details.

To use the project, it is recommended to use the virtual environment. To activate and deacivate the virtual environment, run the following commands:

```bash
# To Activate
source .venv/bin/activate

# To Deactivate
deactivate
```

To install or add a dependency to the project, simply run the command `uv add <module_name>`

```bash
# Example: installing numpy
uv add numpy
```

To run codes, there are multiple ways to do so. The simplest way is to run the following commands:

```bash
# To open the project in Jupyter Lab
uv run --with jupyter jupyter lab

# To run a python script
uv run main.py
```

However, if using Visual Studio Code, the jupyter notebook can be ran as is using the kernel `.venv/bin/python`

### :snake: Python Dependencies

Note that since uv is being used, all the dependencies are immediately installed. However, a requirements.txt file is provided in the off chance that uv doesn't work. To install the dependencies, simply run the command `pip install -r requirements.txt`

## :books: Code Structure

work in progress