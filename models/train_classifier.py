# basic modules
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix, make_scorer, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

# nltk to perform natural language processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

# scikit-learn libraries


def load_data(database_filepath):
    ''' Load database to Pandas dataframe. 
    Assign predictor and response variables after checking 
    if child_alone column have values other than 0.

    Argument: 
        database_filepath : a path to the database

    Output: 
    predictor variables X, response variable Y, 
    a list of category names
    '''
    # Create engine to connect to 'message' table in the database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)

    # Check if child_alone column have values other than 0
    # If not, drop child_alone column
    if df['child_alone'].sum() == 0:
        df.drop(columns='child_alone', inplace=True)

    # Assign target and response variables
    X = df['message']  # disaster response raw message
    Y = df.iloc[:, 4:]  # categories

    # Create a list of category names
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    ''' Split text into tokens follwed by preprocessing processes to
    lowercase the text, remove stopwords and remove puntuations. 
    Then return lemmatized tokens.

    Arguments:
        text(string) <-- disaster response message
    Output : 
        a list that contains lemmitized tokens
    '''

    # Tokenize the text
    tokens = [
        word.strip() for word in text.lower()
        if ((word not in stopwords.words('english')) and
            word.isalnum())
    ]

    # Lemmatize the tokenized words
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmed


def build_model():
    ''' Build a Logistic regressionclassifier model with grid search
    to fit training data

    Target variable Y (messages) will be reproduced to vectorized data
    before fitting into the multiclass classification model.

    When fitting the model, grid search will be performed to generate the 
    best possible parameter evaluated by a customized scorer using micro f1 score.

    Argument: None

    Output: 
        model that is ready to fit with training data    
    '''

    # Create a pipeline
    # max_iter set to 1000 in case of convergence error
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    params = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__C': (1, 100)
    }

    scorer = make_scorer(f1_score, average='micro')
    cv = GridSearchCV(pipeline, params, verbose=3,
                      scoring=scorer, n_jobs=1, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Custome evaluation tool that returns accuracy score and 
    classification report for the fitted model

    Arguments: 
        model: trained classifier model
        X_test: test set predictor variables X
        Y_test: test set target variable Y
        category_names: category names to show to corresponding scores

    Output:
        Accuracy and classificaiton scores for the fitted model
    '''

    # When data is imbalanced, zero-division errors expected
    # Ignore errors to clearly represent scores
    import warnings
    warnings.filterwarnings('ignore')

    # Run predictions on X_test
    Y_pred = model.predict(X_test)

    # Print model accuracy score
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(f'Accuracy score: {accuracy:.2f}%', end='\n\n')
    print(f'Classification score: ')

    # Print model classificaiton report
    results = classification_report(
        Y_test, Y_pred, target_names=category_names)

    print(results)


def save_model(model, model_filepath):
    ''' Save the trained model to a pickle file (.pkl)
    Conventionally, pickle file to be saved in 'models' directory
    as 'classifier_v[version_number].pkl'

    Argument: 
        model: trained model
        model_filepath: a path to save the pickle file

    Output:
        None, a pickle file to be created at the given filepath
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
