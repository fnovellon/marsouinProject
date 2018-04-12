import csv
import string
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

def write_csv(csv_file_name, data):
    d = [];
    for da in data:
        d.append("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in da]).strip());

    csv_file = open(csv_file_name, "w");
    writer = csv.writer(csv_file, delimiter='\n');
    writer.writerow(d);

opinions_m = read_csv("data/dataset2.csv");
class_m = read_csv("data/labels.csv");
test = read_csv("data/test_data.csv");

opinions_train, opinions_test, class_train, class_test = train_test_split(opinions_m, class_m, test_size=0.0);
opinions_test_test, opinions_test_train, class_test_test, class_test_train = train_test_split(test, test, test_size=0.0);

vectorizer = TfidfVectorizer();
opinions_train = vectorizer.fit_transform(opinions_train);
class_test_test = vectorizer.transform(class_test_test);

clf = MultinomialNB();
clf.fit(opinions_train, class_train);
pred = clf.predict(class_test_test);
print(pred);
write_csv("data/test_label.csv", pred);
