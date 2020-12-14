#!/usr/bin/python2.6

from flask import Flask, render_template, request
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("page.html")

def run_classifier(input_text):
    clf = load("./mysite/naive_bayes.joblib")

    vectorizer = load("./mysite/vectorize.joblib")

    test_values = vectorizer.transform([input_text])

    return clf.predict(test_values)

@app.route('/evaluate', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
    else:
        review = request.args.get('review')

    predicted_score = run_classifier(review)

    return render_template('page.html', rating=predicted_score)
