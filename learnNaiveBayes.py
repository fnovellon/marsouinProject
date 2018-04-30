import csv
import string
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import KFold

def read_csv(csv_file_name):
    csv_file = open(csv_file_name, "r");
    opinions = csv_file.readlines();
    return opinions;

def write_csv(csv_file_name, data):
    d = [];
    for da in data:
        d.append("".join([""+i if not i.startswith("'") and i not in string.punctuation else i for i in da]).strip());

    csv_file = open(csv_file_name, "w");
    writer = csv.writer(csv_file, delimiter='\n');
    writer.writerow(d);



vectorizer = TfidfVectorizer();
print("read...");
opinions_m = read_csv("data/dataset2.csv"); #Avis d'entrainement
class_m = read_csv("data/labels.csv"); #Resultats d'entrainement

test = read_csv("data/test_data2.csv"); #Avis du challenge
class_test = read_csv("data/test_labels.csv"); #Resultats du challenge

# kf = KFold(n_splits=3);
clf = MultinomialNB();
# clf = KNeighborsClassifier(n_neighbors=10);
# clf = svm.SVC(kernel='linear');

opinions_train = vectorizer.fit_transform(opinions_m);
opinions_test = vectorizer.transform(test);

########################
### Cross validation ###
########################

# print("cross validation...");
# score = cross_val_score(clf, opinions_train, class_m, cv=10);
# print(score);

print("fit...");
clf.fit(opinions_train, class_m);
print("prediction...");
pred = clf.predict(opinions_test);
print(pred);
accuracy = metrics.accuracy_score(pred, class_test);
print(accuracy);
recall = metrics.recall_score(class_test, pred, average='weighted');
print(recall);
print(2*(accuracy * recall)/(accuracy + recall));
print(metrics.f1_score(class_test, pred, average='weighted'));

#################
### Write csv ###
#################

# print("write csv...");
# write_csv("data/res_label.csv", pred);

# print("Score avec cross validation :" + metrics.accuracy_score(pred, class_test));
# print("Score avec cross validation :" + metrics.accuracy_score(class_test, pred));
