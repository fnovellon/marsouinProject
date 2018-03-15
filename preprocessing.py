import csv
import nltk
from nltk import word_tokenize
import treetaggerwrapper

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
        stop_words.append(word_tokenize(element)[0]);

    data = [];

    for opinion in opinions:
        tmp = [];

        for word in opinion:
            if word not in stop_words:
                tmp.append(word);

        data.append(tmp);

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

opinions = read_csv("data/test.csv");
print(opinions[0]);
print("");
opinions = to_lower(opinions);
print(opinions[0]);
print("");
opinions = tokenize(opinions);
print(opinions[0]);
print("");
opinions = lemmatize(opinions);
print(opinions[0]);
print("");
opinions = remove_stop_words(opinions);
print(opinions[0]);
print("");
opinions = split_sentences(opinions);
# for element in opinions[0]:
#     print(element);
#     print("");
print(opinions[0]);
print("");
