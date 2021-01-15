# Disaster Response Pipeline

## Table of contents

1. [Project motivation](#motivation)
2. [Packages used](#packages)
3. [File description](#file)
4. [Deliverables](#deliverables)
5. [Instructions](#instruction)
6. [Limitation](#limitation)
7. [Acknowledgement](#acknowledgement)

<a id="motivation"></a>

## Motivation

For this project, I built data and machine learning pipelines that are ready for a disaster response classifier web API.

The web app will help people to quickly identify which categories that an emergency message falls into and facilitate decision making on follow-up actions during emergency situations.

<a id="packages"></a>

## Packages used

The code here basically run on the Anaconda distribution of Python (version 3.\*).

You need the following Python packages to install in order to run the app smoothly. The command to install in conda environemt is :

    conda install --channel conda-forge scikit-learn, nltk, sqlalchemy, pickle, flask, plotly

<a id="file"></a>

## File description

> `app` folder : files to run a flask web app.

> `data` folder <br>
>
> -   **etl_pipeline.py :** Python script to preprocess and store raw data
> -   **ETL_pipeline.ipynb :** Jupyter notebook to test codes for a ETL pipeline
> -   **DisasterResponse.db :** SQL database produced from the data pipeline
> -   **categories.csv**, **messages.csv** : Raw datasets

> `models`
>
> -   **train_classifier.py** : Python script used for a machine learning classification modeling
> -   **ML_pipeline.ipynb** : Jupyter notebook to test codes for a ML pipeline
> -   **classifier.pkl** : Pickle file that saves the trained model.

<a id="deliverables"></a>

## Deliverables

The two Python scripts are written to operate data processing and modeling and ready to use for a web app built inside `app/run.py`

<a id="instruction"></a>

## Instructions

To run the app,

1.  Run the following commands in the root folder/repository

    **ETL pipeline to clean and store data**

        python data/etl_pipeline.py data/messages.csv data/categories.csv data/DisasterResponse.db

    **ML pipeline to train classifier and saves the model in pickle file**

        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2.  Run the following command in the `app` directory to run the app

        python run.py

3.  Type the below url on your browser to open the app

        http://0.0.0.0:3001

<a id="limitation"></a>

## Limitation

Disaster response messages can be classified into 36 categories. However, a majority of features from the data (csv files) have imbalanced records.

For example, a feature label `child_alone` contains little information but only 0 values whereas most of the labels are greatly skewed to 0 binary number (negative) as seen in the figure below.

<img src='skewed.png'>

As a result, while evaluating the model, some labels produce true positive of 0 and f1 score calculation then returns zero division errors (as recall and precision equals zero).

Therefore, the model may not be fully trained with the given data sets and require more training data to fit in.

<a id="acknowledgement"></a>

## Acknowledgement

This project is a part of Udacity's Data Scientist Nano Degree program.
Udacity provides the contents need to run the web app (`app` folder), and Figure Eight provided raw data in csv format.
