import sys, pickle, re
import pandas as pd
import numpy as np
import nltk
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

nltk.download(['punkt', 'wordnet'])

df = pd.read_csv('data/messages_response.csv')
X = df['message']
Y = df.iloc[:, 4:]
Y= Y.astype(int)
category_names = list(np.array(Y.columns))


def tokenize(text):
    """
    Tokenize the text and returns the clean tokens
    """
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Model pipeline and performing GridSearch
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model performance
    """
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i], "\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' % (category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))


def save_model(model, model_filepath):
    """
    This function saves trained model as Pickle file, to be loaded later.
    """
    dump(model, 'models/classifier.joblib')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

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