import string
import csv
import nltk
from nltk import word_tokenize
import treetaggerwrapper
import csv
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk import word_tokenize
from sklearn.neighbors import KNeighborsClassifier

def read_csv(csv_file_name):
    csv_file = open(csv_file_name, "r");
    opinions = csv_file.readlines();
    return opinions;

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

    to_save = ['love','hate','like','adore','deceive','dislike','enjoy', 'i like', 'i'];
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
                if strings[0] not in tags and strings[1] not in to_save:
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

def write_csv(csv_file_name, opinions):
    csv_file = open(csv_file_name, "w");
    writer = csv.writer(csv_file, delimiter='\n');
    writer.writerow(opinions);

def learn(opinions_m):
    # class_m = read_csv("data/labels.csv");
    # opinions_train, opinions_test, class_train, class_test = train_test_split(opinions_m, class_m, test_size=0.33);
    #
    # vectorizer = TfidfVectorizer();
    # opinions_train = vectorizer.fit_transform(opinions_train);
    # opinions_test = vectorizer.transform(opinions_test);
    #
    # neigh = KNeighborsClassifier(n_neighbors=100);
    # neigh.fit(opinions_train, class_train);
    #
    # pred = neigh.predict(opinions_test);
    #
    # return metrics.accuracy_score(class_test, pred);



    class_m = read_csv("data/labels.csv");

    opinions_train, opinions_test, class_train, class_test = train_test_split(opinions_m, class_m, test_size=0.33);

    vectorizer = TfidfVectorizer();
    opinions_train = vectorizer.fit_transform(opinions_train);
    opinions_test = vectorizer.transform(opinions_test);

    clf = MultinomialNB();
    clf.fit(opinions_train, class_train);
    pred = clf.predict(opinions_test);
    return metrics.accuracy_score(class_test, pred);

print("begin read\n");
opinions = read_csv("data/dataset.csv");
print("end read\n");
print(opinions[9999]);
print("");
print("begin tokenize\n");
opinions = tokenize(opinions);
print("end tokenize\n");
print(opinions[9999]);
print("");
print("begin lower\n");
opinions = to_lower(opinions);
print("end lower\n");
print(opinions[9999]);
print("");

print("begin remove tags\n");
opinions = remove_tags_from_list(opinions, ["(", ")", "``","CC","DT","EX","FW","IN","IN/that","LS","MD","PDT","POS","PP","PP$","SENT","SYM","TO","UH","WDT","WP","WP","WRB",":","$"]);
print("end remove tags\n");
print(opinions[9999]);
print("");

print("begin lemmatize\n");
opinions = lemmatize(opinions);
print("end lemmatize\n");
print(opinions[9999]);
print("");


print("begin stop words\n");
opinions = remove_stop_words(opinions);
print("end stop words\n");
print(opinions[9999]);
print("");
print("begin untokenize\n");
opinions = untokenize(opinions);
print("end untokenize\n");
print(opinions[9999]);
print("");
print("begin write\n");
write_csv("data/dataset2.csv", opinions);
print("end write\n");
# ratio = [];
# for i in range(100):
    # print("begin learn " + str(i) + "\n");
# print("begin learn");
# print(learn(opinions));
# print("end learn");
    # print("end learn " + str(i) + "\n");

# average = 0;
#
# for i in range(100):
#     average = average + ratio[i];
#
# print(average / 100);
