import string
import csv
import nltk
from nltk import word_tokenize
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

def to_lower(opinions):
    data = [];

    for opinion in opinions:
        data.append(opinion.lower());

    return data;

def tokenize(opinions):
    data = [];

    for opinion in opinions:
        data.append(word_tokenize(opinion));

    return data;

def lemmatize(opinions):
    wnl = nltk.WordNetLemmatizer();

    data = [];

    for opinion in opinions:
        lemmatize = [wnl.lemmatize(t, pos='v') for t in opinion];
        lemmatize = [wnl.lemmatize(t) for t in lemmatize];
        data.append(lemmatize);

    return data;

def remove_stop_words(opinions):
    stop_words_file = open("rsc/stopwords.csv", "r");
    stop_words_list = stop_words_file.readlines();
    stop_words = [];

    for element in stop_words_list:
        stop_words.append(element.rstrip("\n"));

    data = [];

    for opinion in opinions:
        tmp = [];

        for word in opinion:
            if word not in stop_words:
                tmp.append(word);

        data.append(tmp);

    return data;

def untokenize(opinions):
    data = [];
    for opinion in opinions:
        data.append("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in opinion]).strip());

    return data;

def split_sentences(opinions):
    data = [];

    for opinion in opinions:
        tmp = [];
        tmp2 = [];

        for word in opinion:
            if word == '.':
                tmp2.append(tmp);
                tmp = [];
            else:
                tmp.append(word);

        data.append(tmp2);
        tmp2 = [];

    return data;

def learn(opinions_m):
    class_m = read_csv("data/labels.csv");

    opinions_train, opinions_test, class_train, class_test = train_test_split(opinions_m, class_m, test_size=0.33, random_state=1);

    vectorizer = TfidfVectorizer();
    opinions_train = vectorizer.fit_transform(opinions_train);
    opinions_test = vectorizer.transform(opinions_test);

    clf = MultinomialNB();
    clf.fit(opinions_train, class_train);
    pred = clf.predict(opinions_test);
    print(metrics.accuracy_score(class_test, pred));

print("begin read\n");
opinions = read_csv("data/dataset.csv");
print("end read\n");
# print(opinions[0]);
# print("");
print("begin lower\n");
opinions = to_lower(opinions);
print("end lower\n");
# print(opinions[0]);
# print("");
print("begin tokenize\n");
opinions = tokenize(opinions);
print("end tokenize\n");
# print(opinions[0]);
# print("");
print("begin lemmatize\n");
opinions = lemmatize(opinions);
print("end lemmatize\n");
# print(opinions[0]);
# print("");
print("begin stop words\n");
opinions = remove_stop_words(opinions);
print("end stop words\n");
# print(opinions[0]);
# print("");
# opinions = split_sentences(opinions);
# for element in opinions[0]:
#     print(element);
#     print("");
# print(opinions[0]);
# print("");
print("begin untokenize\n");
opinions = untokenize(opinions);
print("end untokenize\n");
# print(opinions[0]);
# print("");
print("begin learn\n");
learn(opinions);
print("end learn\n");
