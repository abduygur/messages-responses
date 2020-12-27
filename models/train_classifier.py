import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from text_tokenize import text_tokenizer


def load_data(database_filepath):
    """
        Load Data from the Database

        Parameters
        ----------
        database_filepath

        Returns
        -------
        X: Independent Variables
        Y: Dependent Variables

    """
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.iloc[:, 4:]

    return X,Y



def build_model():
    """
       Create Pipeline Using CountVectorizer, TfidfTransformer and MultiOutputClassifier

       Parameters
       ----------
       None

       Returns
       -------
       pipeline

    """

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=partial(text_tokenizer))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=20, min_samples_split=10,
                                   n_jobs=-1)))
    ])

    # Create Dictionary for tuning min_samples_split and n_estimators in Random Forest Classifier
    parameters = {
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__n_estimators': [50, 100, 200]
    }

    scorer = make_scorer(performance_metric)

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=5, n_jobs=5, scoring=scorer)



    return model


def evaluate_model(model, X_test, y_test):
    """
        Calculate the Accuracy, Precision, Recall and F1 Score and Return Scores for all Targets

        Parameters
        ----------
        y_test: Actual Data
        y_pred: Predicted Data

        Returns
        -------
        df_score: Scores which includes Acc, Prec, Recall, F1 Score and True Rate for all Targets

    """

    y_pred = model.predict(X_test)

    scorelist = []

    # Iterate over y columns
    for i, target in enumerate(y_test):
        # Calculate metrics for actual and predicted data
        accuracy = accuracy_score(y_test[target], y_pred[:, i])
        precision = precision_score(y_test[target], y_pred[:, i],zero_division=0)
        recall = recall_score(y_test[target], y_pred[:, i])
        f1 = f1_score(y_test[target], y_pred[:, i])
        scorelist.append([accuracy, precision, recall, f1, round(y_test[target].mean() * 100, 2)])

    # Create df which includes all metrics for all y columns
    df_score = pd.DataFrame(data=np.array(scorelist), index=y_test.columns,
                            columns=['Acc', 'Pre', 'Rec', 'F1', 'True Rate (%)'])

    print(df_score)


def save_model(model, model_filepath):
    """
        Dump the model to model_filepath using pickle

        Parameters
        ----------
        model: Created Estimator
        model_filepath: Path for saving the model

        Returns
        -------
        None

    """
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()