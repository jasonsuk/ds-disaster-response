# Disaster Response Pipeline

## Table of contents

1. [Project motivation](#motivation)
2. [File description](#file)
3. [Deliverables](#deliverables)
4. [Installation](#installation)
5. [Acknowledgement](#acknowledgement)

<a id="motivation"></a>

## Motivation

For this project, I built pipelines that are ready for an API that will classifcy disaster messages into relevant categories.

The primary purpose of this project is to build ETL-pipeline and ML-pipeline. Therefore the model performance is not the focus.

<a id="file"></a>

## File description

```bash
├── app
│   ├── templates
│   │   ├── go.html
│   │   └── master.html
│   └── run.py
├── data
│   ├── categories.csv
│   ├── DisasterResponse.db
│   ├── ETL_pipeline.ipynb
│   ├── etl_pipeline.py
│   └── messages.csv
├── models
│   ├── classifier.pkl
│   ├── ML_pipeline.ipynb
│   └── train_classifier.py
├── .gitignore
└── README.md
```

> `app` folder : files to run a flask web app.

> `data` folder <br>
>
> -   **etl_pipeline.py :** Python script to preprocess and store raw data
> -   **ETL_pipeline.ipynb :** Jupyter notebook to test codes with logics used to build a ETL pipeline
> -   **DisasterResponse.db :** SQL database produced from the data pipeline
> -   **categories.csv**, **messages.csv** : Raw datasets

> `models`
>
> -   **train_classifier.py** : Python script used for a machine learning classification modeling
> -   **ML_pipeline.ipynb** : Jupyter notebook to test codes with logics used to build a ML pipeline
> -   **classifier.pkl** : Pickle file that saves the trained model.

<a id="deliverables"></a>

## Deliverables

The two Python scripts are written to operate data processing and modeling and ready to use for a web app built inside `app/run.py`

<a id="installation"></a>

## Installation

The code here basically run on the Anaconda distribution of Python (version 3.\*).

<a id="acknowledgement"></a>

## Acknowledgement

This project is a part of Udacity's Data Scientist Nano Degree program.
Udacity provides the contents need to run the web app (`app` folder), and Figure Eight provided raw data in csv format.
