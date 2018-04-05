import csv
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk import word_tokenize

def read_csv(csv_file_name):
    csv_file = open(csv_file_name, "r");
    opinions = csv_file.readlines();
    return opinions;

def tokenize(opinions):
    data = [];

    for opinion in opinions:
        data.append(word_tokenize(opinion));

    return data;

opinions_m = read_csv("data/dataset.csv");
opinions_m = tokenize(opinions_m);
class_m = read_csv("data/labels.csv");

opinions_train, opinions_test, class_train, class_test = train_test_split(opinions_m, class_m, test_size=0.33, random_state=1);

vectorizer = TfidfVectorizer();
opinions_train = vectorizer.fit_transform(opinions_train);
opinions_test = vectorizer.transform(opinions_test);

clf = MultinomialNB();
clf.fit(opinions_train, class_train);
pred = clf.predict(opinions_test);
print(metrics.accuracy_score(class_test, pred));


[string1, string2]
['Je', 'suis', 'content', ',' , 'c'est', 'cool', '.']
['content', 'cool']
content , cool
