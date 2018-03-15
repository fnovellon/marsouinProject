from __future__ import division  # Python 2 users only
import csv
import re
import nltk, re, pprint, operator
from nltk import word_tokenize

# def m_lemmatize(array):
#
#
#
#
#
# def main():

csvfile = open('data/dataset.csv', "r");
# outputfile = open('data/result.txt', 'w');
# outputfile2 = open('data/result2.txt', 'w');

datas = csvfile.readlines();

print("tokenning... ", end='')
dict = {}
for data in datas :
    newTokens = []

    tokens = word_tokenize(data)
    for token in tokens :
        if "." in token : 
            m = re.search('[a-z]\.[A-Z]', token)
            if m :
                buffer = token.split(".")
                newTokens.append(buffer[0])
                newTokens.append(".")
                newTokens.append(buffer[1])
            else :
                newTokens.append(token)
        else :
            newTokens.append(token)

    for token in newTokens :
        
        if token in dict:
            dict[token] += 1
        else:
            dict[token] = 1


print("end")

print("sorting... ", end='')
sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
print("end")

print("writing... ", end='')
file = open("new", "w")
for key in sorted_x :
    s = key[0] + " : " + str(key[1]) + "\n"
    file.write(s)
print("end")
        
#
"""
wnl = nltk.WordNetLemmatizer()
lemmatize = [wnl.lemmatize(t, pos='v') for t in data]
lemmatize = [wnl.lemmatize(t) for t in lemmatize]
print(lemmatize);
"""