from flask import Flask
import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
from joblib import load
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import pickle

application = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
df = pd.read_csv('data/messages_response.csv')

# load model
model = joblib.load("models/classifier.joblib")

#with open("models/classifier.pkl", 'rb') as pickleFile:
 #   model = pickle.load(pickleFile)
@application.route('/')
@application.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories = df[df.columns[4:]]
    category_counts = (categories.mean() * categories.shape[0]).sort_values(ascending=False)
    category_names = list(category_counts.index)

    # render web page with plotly graphs
    return render_template('master.html')


# web page that handles user query and displays model results
@application.route('/go')
def go():
        # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
    query=query,
    classification_result=classification_results
    )

if __name__ == '__main__':
    application.run()