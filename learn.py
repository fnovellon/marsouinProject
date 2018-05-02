import csv
import string
import time

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
from sklearn.tree import DecisionTreeClassifier
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

begin = time.time();

print("Generate classifier...");
classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
classifier_name = ["Naive Bayes", "1-Nearest Neighbors", "3-Nearest Neighbors", "5-Nearest Neighbors", "10-Nearest Neighbors", "DecisionTree", "SVC poly", "SVC linear"];
classifier_simple = ["NB", "1-NN", "3-NN", "5-NN", "10-NN", "DT", "SVC-p", "SVC-l"];
cputime = [];

# brut text
cross_validation_brut = [];
average_brut = [];
deviation_brut = [];
predictions_brut = [];
accuracy_brut = [];
recall_brut = [];
fmesure_brut = [];
cputime_brut = [];
# lemmatized text
cross_validation_lemm = [];
average_lemm = [];
deviation_lemm = [];
predictions_lemm = [];
accuracy_lemm = [];
recall_lemm = [];
fmesure_lemm = [];
cputime_lemm = [];
# stop words text
cross_validation_stop = [];
average_stop = [];
deviation_stop = [];
predictions_stop = [];
accuracy_stop = [];
recall_stop = [];
fmesure_stop = [];
cputime_stop = [];
# tree tagger text
cross_validation_tag = [];
average_tag = [];
deviation_tag = [];
predictions_tag = [];
accuracy_tag = [];
recall_tag = [];
fmesure_tag = [];
cputime_tag = [];
# all preprocessing text
cross_validation_all = [];
average_all = [];
deviation_all = [];
predictions_all = [];
accuracy_all = [];
recall_all = [];
fmesure_all= [];
cputime_all = [];

vectorizer = TfidfVectorizer();

print("Read all .csv files...");
# data train
opinions_train_brut = read_csv("data/dataset.csv");
# labels train
labels_train = read_csv("data/labels.csv");
# data challenge
opinions_challenge_brut = read_csv("data/test_data.csv");
# labels challenge
labels_challenge = read_csv("data/test_labels.csv");

#####################
### PREPROCESSING ###
#####################

t1 = time.time();
print("Tokenize...");
opinions_train_token = tokenize(opinions_train_brut);
opinions_challenge_token = tokenize(opinions_challenge_brut);
print("Lower...");
opinions_train_lower = to_lower(opinions_train_token);
opinions_challenge_lower = to_lower(opinions_challenge_token);
print("Tree Tagger...");
opinions_train_tag = remove_tags_from_list(opinions_train_lower, ["CC","DT","EX","FW","IN","IN/that","JJ","LS","MD","PDT","POS","PP","PP$","SENT","SYM","TO","UH","WDT","WP","WP","WRB",":","$"]);
opinions_challenge_tag = remove_tags_from_list(opinions_challenge_lower, ["CC","DT","EX","FW","IN","IN/that","JJ","LS","MD","PDT","POS","PP","PP$","SENT","SYM","TO","UH","WDT","WP","WP","WRB",":","$"]);
print("Lemmatize...");
opinions_train_lemm = lemmatize(opinions_train_lower);
opinions_challenge_lemm = lemmatize(opinions_challenge_lower);
opinions_train_all = lemmatize(opinions_train_tag);
opinions_challenge_all = lemmatize(opinions_challenge_tag);
print("Remove stop words...");
opinions_train_stop = remove_stop_words(opinions_train_lemm);
opinions_challenge_stop = remove_stop_words(opinions_challenge_lemm);
opinions_train_all = lemmatize(opinions_train_all);
opinions_challenge_all = lemmatize(opinions_challenge_all);
print("Untokenize...");
opinions_train_lemm = untokenize(opinions_train_lemm);
opinions_challenge_lemm = untokenize(opinions_challenge_lemm);
opinions_train_stop = untokenize(opinions_train_stop);
opinions_challenge_stop = untokenize(opinions_challenge_stop);
opinions_train_tag = untokenize(opinions_train_tag);
opinions_challenge_tag = untokenize(opinions_challenge_tag);
opinions_train_all = untokenize(opinions_train_all);
opinions_challenge_all = untokenize(opinions_challenge_all);
t2 = time.time();
print("Preprocessing time : " + str(int((t2-t1)/60)) + "m " + str(int((t2-t1)%60)) + "s");

###################
### NAIVE BAYES ###
###################

for i in range(len(classifier)):
    print("");
    print(classifier_name[i]);
    print("");
    t1 = time.time();

    print("BRUTE TEXT");
    classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
    opinions_train_vect = vectorizer.fit_transform(opinions_train_brut);
    opinions_challenge_vect = vectorizer.transform(opinions_challenge_brut);
    print("Cross validation " + classifier_name[i] + "...");
    brut1 = time.time();
    cross_validation_brut.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_brut.append(0.0);
    for j in range(len(cross_validation_brut[i])):
        average_brut[i] += cross_validation_brut[i][j];
    average_brut[i] /= len(cross_validation_brut[i]);
    average_brut[i] = average_brut[i].__round__(3);
    deviation_brut.append(max(cross_validation_brut[i]) - min(cross_validation_brut[i]));
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);
    print("Predict " + classifier_name[i] + "...");
    predictions_brut.append(classifier[i].predict(opinions_challenge_vect));
    print("accuracy " + classifier_name[i] + "...");
    accuracy_brut.append(metrics.accuracy_score(labels_challenge, predictions_brut[i]).__round__(3));
    print("recall " + classifier_name[i] + "...");
    recall_brut.append(metrics.recall_score(labels_challenge, predictions_brut[i], average='weighted').__round__(3));
    print("f-mesure " + classifier_name[i] + "...");
    fmesure_brut.append(metrics.f1_score(labels_challenge, predictions_brut[i], average='weighted').__round__(3));
    brut2 = time.time();
    cputime_brut.append((brut2 - brut1).__round__(3));
    print("");

    print("TREE TAGGER TEXT");
    classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
    opinions_train_vect = vectorizer.fit_transform(opinions_train_tag);
    opinions_challenge_vect = vectorizer.transform(opinions_challenge_tag);
    print("Cross validation " + classifier_name[i] + "...");
    tag1 = time.time();
    cross_validation_tag.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_tag.append(0.0);
    for j in range(len(cross_validation_tag[i])):
        average_tag[i] += cross_validation_tag[i][j];
    average_tag[i] /= len(cross_validation_tag[i]);
    average_tag[i] = average_tag[i].__round__(3);
    deviation_tag.append(max(cross_validation_tag[i]) - min(cross_validation_tag[i]));
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);
    print("Predict " + classifier_name[i] + "...");
    predictions_tag.append(classifier[i].predict(opinions_challenge_vect));
    print("accuracy " + classifier_name[i] + "...");
    accuracy_tag.append(metrics.accuracy_score(labels_challenge, predictions_tag[i]).__round__(3));
    print("recall " + classifier_name[i] + "...");
    recall_tag.append(metrics.recall_score(labels_challenge, predictions_tag[i], average='weighted').__round__(3));
    print("f-mesure " + classifier_name[i] + "...");
    fmesure_tag.append(metrics.f1_score(labels_challenge, predictions_tag[i], average='weighted').__round__(3));
    tag2 = time.time();
    cputime_tag.append((tag2 - tag1).__round__(3));
    print("");

    print("LEMMATIZE TEXT");
    classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
    opinions_train_vect = vectorizer.fit_transform(opinions_train_lemm);
    opinions_challenge_vect = vectorizer.transform(opinions_challenge_lemm);
    print("Cross validation " + classifier_name[i] + "...");
    lemm1 = time.time();
    cross_validation_lemm.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_lemm.append(0.0);
    for j in range(len(cross_validation_lemm[i])):
        average_lemm[i] += cross_validation_lemm[i][j];
    average_lemm[i] /= len(cross_validation_lemm[i]);
    average_lemm[i] = average_lemm[i].__round__(3);
    deviation_lemm.append(max(cross_validation_lemm[i]) - min(cross_validation_lemm[i]));
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);
    print("Predict " + classifier_name[i] + "...");
    predictions_lemm.append(classifier[i].predict(opinions_challenge_vect));
    print("accuracy " + classifier_name[i] + "...");
    accuracy_lemm.append(metrics.accuracy_score(labels_challenge, predictions_lemm[i]).__round__(3));
    print("recall " + classifier_name[i] + "...");
    recall_lemm.append(metrics.recall_score(labels_challenge, predictions_lemm[i], average='weighted').__round__(3));
    print("f-mesure " + classifier_name[i] + "...");
    fmesure_lemm.append(metrics.f1_score(labels_challenge, predictions_lemm[i], average='weighted').__round__(3));
    lemm2 = time.time();
    cputime_lemm.append((lemm2 - lemm1).__round__(3));
    print("");

    print("STOP WORDS TEXT");
    classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
    opinions_train_vect = vectorizer.fit_transform(opinions_train_stop);
    opinions_challenge_vect = vectorizer.transform(opinions_challenge_stop);
    print("Cross validation " + classifier_name[i] + "...");
    stop1 = time.time();
    cross_validation_stop.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_stop.append(0.0);
    for j in range(len(cross_validation_stop[i])):
        average_stop[i] += cross_validation_stop[i][j];
    average_stop[i] /= len(cross_validation_stop[i]);
    average_stop[i] = average_stop[i].__round__(3);
    deviation_stop.append(max(cross_validation_stop[i]) - min(cross_validation_stop[i]));
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);
    print("Predict " + classifier_name[i] + "...");
    predictions_stop.append(classifier[i].predict(opinions_challenge_vect));
    print("accuracy " + classifier_name[i] + "...");
    accuracy_stop.append(metrics.accuracy_score(labels_challenge, predictions_stop[i]).__round__(3));
    print("recall " + classifier_name[i] + "...");
    recall_stop.append(metrics.recall_score(labels_challenge, predictions_stop[i], average='weighted').__round__(3));
    print("f-mesure " + classifier_name[i] + "...");
    fmesure_stop.append(metrics.f1_score(labels_challenge, predictions_stop[i], average='weighted').__round__(3));
    stop2 = time.time();
    cputime_stop.append((stop2 - stop1).__round__(3));
    print("");

    print("PREPROCESSING TEXT");
    classifier = [MultinomialNB(), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(), svm.SVC(kernel='poly'), svm.SVC(kernel='linear')];
    opinions_train_vect = vectorizer.fit_transform(opinions_train_all);
    opinions_challenge_vect = vectorizer.transform(opinions_challenge_all);
    print("Cross validation " + classifier_name[i] + "...");
    all1 = time.time();
    cross_validation_all.append(cross_val_score(classifier[i], opinions_train_vect, labels_train, cv=10));
    average_all.append(0.0);
    for j in range(len(cross_validation_all[i])):
        average_all[i] += cross_validation_all[i][j];
    average_all[i] /= len(cross_validation_all[i]);
    average_all[i] = average_all[i].__round__(3);
    deviation_all.append(max(cross_validation_all[i]) - min(cross_validation_all[i]));
    print("Fit " + classifier_name[i] + "...");
    classifier[i].fit(opinions_train_vect, labels_train);
    print("Predict " + classifier_name[i] + "...");
    predictions_all.append(classifier[i].predict(opinions_challenge_vect));
    print("accuracy " + classifier_name[i] + "...");
    accuracy_all.append(metrics.accuracy_score(labels_challenge, predictions_all[i]).__round__(3));
    print("recall " + classifier_name[i] + "...");
    recall_all.append(metrics.recall_score(labels_challenge, predictions_all[i], average='weighted').__round__(3));
    print("f-mesure " + classifier_name[i] + "...");
    fmesure_all.append(metrics.f1_score(labels_challenge, predictions_all[i], average='weighted').__round__(3));
    all2 = time.time();
    cputime_all.append((all2 - all1).__round__(3));

    t2 = time.time();
    print("");
    cputime.append((t2 - t1).__round__(3));
    print("CPU TIME : " + str(int(cputime[i]/60)) + "m " + str(int(cputime[i]%60)) + "s");

end = time.time();

#######################
### DISPLAY RESULTS ###
#######################

print("Brut text results :");
print("Classifier : \t\t\t", end="");
print(*classifier_simple, sep='\t');
print("Cross validation average : \t", end="");
print(*average_brut, sep='\t');
print("Cross validation deviation : \t", end="");
print(*deviation_brut, sep='\t');
print("Accuracy : \t\t\t", end="");
print(*accuracy_brut, sep='\t');
print("Recall : \t\t\t", end="");
print(*recall_brut, sep='\t');
print("F-mesure : \t\t\t", end="");
print(*fmesure_brut, sep='\t');
print("");

print("Tree tagger text results :");
print("Classifier : \t\t\t", end="");
print(*classifier_simple, sep='\t');
print("Cross validation average : \t", end="");
print(*average_tag, sep='\t');
print("Cross validation deviation : \t", end="");
print(*deviation_tag, sep='\t');
print("Accuracy : \t\t\t", end="");
print(*accuracy_tag, sep='\t');
print("Recall : \t\t\t", end="");
print(*recall_tag, sep='\t');
print("F-mesure : \t\t\t", end="");
print(*fmesure_tag, sep='\t');
print("");

print("Lemmatize text results :");
print("Classifier : \t\t\t", end="");
print(*classifier_simple, sep='\t');
print("Cross validation average : \t", end="");
print(*average_lemm, sep='\t');
print("Cross validation deviation : \t", end="");
print(*deviation_lemm, sep='\t');
print("Accuracy : \t\t\t", end="");
print(*accuracy_lemm, sep='\t');
print("Recall : \t\t\t", end="");
print(*recall_lemm, sep='\t');
print("F-mesure : \t\t\t", end="");
print(*fmesure_lemm, sep='\t');
print("");

print("Stop words text results :");
print("Classifier : \t\t\t", end="");
print(*classifier_simple, sep='\t');
print("Cross validation average : \t", end="");
print(*average_stop, sep='\t');
print("Cross validation deviation : \t", end="");
print(*deviation_stop, sep='\t');
print("Accuracy : \t\t\t", end="");
print(*accuracy_stop, sep='\t');
print("Recall : \t\t\t", end="");
print(*recall_stop, sep='\t');
print("F-mesure : \t\t\t", end="");
print(*fmesure_stop, sep='\t');
print("");

print("Preprocessing text results :");
print("Classifier : \t\t\t", end="");
print(*classifier_simple, sep='\t');
print("Cross validation average : \t", end="");
print(*average_all, sep='\t');
print("Cross validation deviation : \t", end="");
print(*deviation_all, sep='\t');
print("Accuracy : \t\t\t", end="");
print(*accuracy_all, sep='\t');
print("Recall : \t\t\t", end="");
print(*recall_all, sep='\t');
print("F-mesure : \t\t\t", end="");
print(*fmesure_all, sep='\t');
print("");

print("CPU time (second) :");
print("Classifier : \t\t", end="");
print(*classifier_simple, sep='\t');
print("Brut text : \t", end="");
print(*cputime_brut, sep='\t');
print("Lemmatized text : \t", end="");
print(*cputime_lemm, sep='\t');
print("Stop words text : \t", end="");
print(*cputime_stop, sep='\t');
print("Tree tagger text : \t", end="");
print(*cputime_tag, sep='\t');
print("Preprocessing text : \t", end="");
print(*cputime_all, sep='\t');

print("");
print("Total CPU time : " + str(int((end - begin)/60)) + "min " + str(int((end - begin)%60)) + "sec");
