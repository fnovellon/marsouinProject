import csv
import string

import nltk
from nltk import word_tokenize
import re
import treetaggerwrapper

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

# Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def read_csv(csv_file_name):
    csv_file = open(csv_file_name, "r");
    data = csv_file.readlines();
    return data;

def tokenize(opinions):
    datas = [];

    for opinion in opinions:
        newTokens = [];
        tokens = word_tokenize(opinion);
        for token in tokens:
            if "." in token:
                m = re.search('[a-z0-9]\.[A-Z]', token);
                if m:
                    buffer = token.split(".");
                    newTokens.append(buffer[0]);
                    newTokens.append(".");
                    newTokens.append(buffer[1]);

                else:
                    newTokens.append(token);

            else:
                newTokens.append(token);

        datas.append(newTokens);

    return datas;

def to_lower(opinions):
    datas = [];

    for opinion in opinions:
        tokens = [];

        for token in opinion:
            tokens.append(token.lower());

        datas.append(tokens);

    return datas;

def lemmatize(opinions):
    wnl = nltk.WordNetLemmatizer();

    data = [];

    for opinion in opinions:
        lemmatize = [wnl.lemmatize(t, pos='v') for t in opinion];
        lemmatize = [wnl.lemmatize(t) for t in lemmatize];
        data.append(lemmatize);

    return data;

def remove_tags_from_list(opinions, tags):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='./tt');
    new_opinions = [];
    # i = 0;
    for opinion in opinions:
        # print("iteration " , i)
        phrase = " ".join(opinion);
        tagged = tagger.tag_text(phrase);
        new_opinion = [];

        for element in tagged:
            strings = [];
            strings = element.split("\t");
            try:
                if strings[1] not in tags:
                    new_opinion.append(strings[0])
            except:
                    continue;
        new_opinions.append(new_opinion)
        # i = i + 1;
    # print("result : ")

    return new_opinions;

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

def write_csv(csv_file_name, opinions):
    csv_file = open(csv_file_name, "w");
    writer = csv.writer(csv_file, delimiter='\n');
    writer.writerow(opinions);

#################
### Variables ###
#################

print("Generate classifier...");
classifier_name = ["Naive Bayes", "1-Nearest Neighbors", "3-Nearest Neighbors", "5-Nearest Neighbors", "10-Nearest Neighbors", "SVC"];
cross_validation_brut = [];
average_brut = [];
predictions_brut = [];
cross_validation_lemm = [];
average_lemm = [];
predictions_lemm = [];
cross_validation_stop = [];
average_stop = [];
predictions_stop = [];
cross_validation_tree = [];
average_tree = [];
predictions_tree = [];

vectorizer = TfidfVectorizer();

print("Read all .csv files...");
# data train
opinions_train = read_csv("data/dataset.csv");
# labels train
labels_train = read_csv("data/labels.csv");
# data challenge
opinions_challenge = read_csv("data/test_data.csv");
# labels challenge
labels_challenge = read_csv("data/test_labels.csv");

##############################
### FIRST STEP : BRUT TEXT ###
##############################

print("BRUTE TEXT");
print("");

classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), svm.SVC(kernel='linear')];

# transform text
opinions_train_vect = vectorizer.fit_transform(opinions_train);
opinions_challenge_vect = vectorizer.transform(opinions_challenge);

# cross validation
for i in range(len(classifier)):
    print("Cross validation " + classifier_name[i] + "...");
    cross_validation_brut.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_brut.append(0.0);
    for j in range(10):
        average_brut[i] += cross_validation_brut[i][j];
    average_brut[i] /= 10;

# fit
for i in range(len(classifier)):
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);

# predict
for i in range(len(classifier)):
    print("Predict " + classifier_name[i] + "...");
    predictions_brut.append(classifier[i].predict(opinions_challenge_vect));

# f-mesure
for i in range(len(classifier)):
    print("f-mesure " + classifier_name[i] + " : " + str(metrics.f1_score(labels_challenge, predictions_brut[i], average='weighted')));

####################################
### SECOND STEP : LEMMATIZE TEXT ###
####################################

print("Tokenize...");
opinions_train = tokenize(opinions_train);
opinions_challenge = tokenize(opinions_challenge);
print("Lower...");
opinions_train = to_lower(opinions_train);
opinions_challenge = to_lower(opinions_challenge);
print("Lemmatize...");
opinions_train = lemmatize(opinions_train);
opinions_challenge = lemmatize(opinions_challenge);
print("Untokenize...\n");
opinions_train = untokenize(opinions_train);
opinions_challenge = untokenize(opinions_challenge);

print("LEMMATIZE TEXT");
print("");

classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), svm.SVC(kernel='linear')];

# transform text
opinions_train_vect = vectorizer.fit_transform(opinions_train);
opinions_challenge_vect = vectorizer.transform(opinions_challenge);

# cross validation
for i in range(len(classifier)):
    print("Cross validation " + classifier_name[i] + "...");
    cross_validation_lemm.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_lemm.append(0.0);
    for j in range(10):
        average_lemm[i] += cross_validation_lemm[i][j];
    average_lemm[i] /= 10;

# fit
for i in range(len(classifier)):
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);

# predict
for i in range(len(classifier)):
    print("Predict " + classifier_name[i] + "...");
    predictions_lemm.append(classifier[i].predict(opinions_challenge_vect));

# f-mesure
for i in range(len(classifier)):
    print("f-mesure " + classifier_name[i] + " : " + str(metrics.f1_score(labels_challenge, predictions_lemm[i], average='weighted')));

#####################################
### THIRD STEP : STOP WORDS TEXT ###
#####################################

print("Tokenize...");
opinions_train = tokenize(opinions_train);
opinions_challenge = tokenize(opinions_challenge);
print("Remove stop words...");
opinions_train = remove_stop_words(opinions_train);
opinions_challenge = remove_stop_words(opinions_challenge);
print("Untokenize...\n");
opinions_train = untokenize(opinions_train);
opinions_challenge = untokenize(opinions_challenge);

print("STOP WORDS TEXT");
print("");

classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), svm.SVC(kernel='linear')];

# transform text
opinions_train_vect = vectorizer.fit_transform(opinions_train);
opinions_challenge_vect = vectorizer.transform(opinions_challenge);

# cross validation
for i in range(len(classifier)):
    print("Cross validation " + classifier_name[i] + "...");
    cross_validation_stop.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_stop.append(0.0);
    for j in range(10):
        average_stop[i] += cross_validation_stop[i][j];
    average_stop[i] /= 10;

# fit
for i in range(len(classifier)):
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);

# predict
for i in range(len(classifier)):
    print("Predict " + classifier_name[i] + "...");
    predictions_stop.append(classifier[i].predict(opinions_challenge_vect));

# f-mesure
for i in range(len(classifier)):
    print("f-mesure " + classifier_name[i] + " : " + str(metrics.f1_score(labels_challenge, predictions_stop[i], average='weighted')));

######################################
### FOURTH STEP : TREE TAGGER TEXT ###
######################################

print("Tokenize...");
opinions_train = tokenize(opinions_train);
opinions_challenge = tokenize(opinions_challenge);
print("Tree Tagger...");
opinions_train = remove_tags_from_list(opinions_train, ["CC","DT","EX","FW","IN","IN/that","JJ","LS","MD","PDT","POS","PP","PP$","SENT","SYM","TO","UH","WDT","WP","WP","WRB",":","$"]);
opinions_challenge = remove_tags_from_list(opinions_challenge, ["CC","DT","EX","FW","IN","IN/that","JJ","LS","MD","PDT","POS","PP","PP$","SENT","SYM","TO","UH","WDT","WP","WP","WRB",":","$"]);
print("Untokenize...\n");
opinions_train = untokenize(opinions_train);
opinions_challenge = untokenize(opinions_challenge);

print("TREE TAGGER TEXT");
print("");

classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), svm.SVC(kernel='linear')];

# transform text
opinions_train_vect = vectorizer.fit_transform(opinions_train);
opinions_challenge_vect = vectorizer.transform(opinions_challenge);

# cross validation
for i in range(len(classifier)):
    print("Cross validation " + classifier_name[i] + "...");
    cross_validation_tree.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_tree.append(0.0);
    for j in range(10):
        average_tree[i] += cross_validation_tree[i][j];
    average_tree[i] /= 10;

# fit
for i in range(len(classifier)):
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);

# predict
for i in range(len(classifier)):
    print("Predict " + classifier_name[i] + "...");
    predictions_tree.append(classifier[i].predict(opinions_challenge_vect));

# f-mesure
for i in range(len(classifier)):
    print("f-mesure " + classifier_name[i] + " : " + str(metrics.f1_score(labels_challenge, predictions_tree[i], average='weighted')));
