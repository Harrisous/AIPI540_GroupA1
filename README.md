# AIPI540 Group A1 Project

## Project Description
[Add your project description here]

## Setup and Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run setup:
```bash
python setup.py
```
ps: there should be a model_scripted.pt in the "models" folder, but the model was too large and rejected by GitHub, so the user may want to run the deep-learning script to generate the model file before run main.py
4. Run the project:
```bash
streamlit run main.py
```

## Project Structure
```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── setup.py               <- script to set up project
├── main.py                <- main script to run project UI
├── scripts/               <- directory for pipeline scripts
│   ├── make_dataset.py    <- script to get data
│   ├── build_features.py  <- script to generate features
│   └── model.py           <- script to train model and predict
├── models/                <- directory for trained models
├── data/                  <- directory for project data
│   ├── raw/              <- directory for raw data
│   ├── processed/        <- directory for processed data
│   └── outputs/          <- directory for output data
└── notebooks/            <- directory for exploration notebooks
```

## Team Members

Haochen Li, Harshitha Rasamsetty, Dave Wang, Xiaoquan Kong