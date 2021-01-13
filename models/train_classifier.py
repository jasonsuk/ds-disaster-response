# basic modules
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)

    df.drop(columns='child_alone', inplace=True)

    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    tokens = [
        word.strip() for word in text.lower()
        if ((word not in stopwords.words('english')) and
            word.isalnum())
    ]

    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(token) for token in tokens]

    return lemmed


def build_model():

    # max_iter set to 1000 in case of convergence error
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(C=100, max_iter=1000)))
    ])

    # params = {
    #     'tfidf__use_idf': (True, False),
    #     'clf__estimator__C': (1, 100)
    # }

    # scorer = make_scorer(f1_score, average='micro')
    # cv = GridSearchCV(pipeline, params, verbose=3,
    #                   scoring=scorer, n_jobs=1, cv=2)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names, expand=False):

    import warnings
    warnings.filters('ignore')

    Y_pred = model.predict(X_test)
    Y_test = np.array(Y_test)

    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print(f'Accuracy score: {accuracy:.2f}%', end='\n\n')
    print(f'Classification score: ')

    results = classification_report(
        Y_test, Y_pred, target_names=category_names, output_dict=True)

    df = pd.DataFrame(results)

    if expand:
        print(df)

    else:

        df = pd.DataFrame(results)
        print(df[['micro avg', 'weighted avg']])


def save_model(model, model_filepath):
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
