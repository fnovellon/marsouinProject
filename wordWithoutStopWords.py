import csv
import re
import nltk, re, pprint, operator
from nltk import word_tokenize
from nltk.corpus import stopwords

dictionnary={}

with open('new') as f: 
    for line in f: 
        (key, value) = line.split(' : ')
        dictionnary[key] = int(value)


stop_words = []
stop_words_capitalized = []
    

with open('rsc/stopwords2') as s: 
    for line in s: 
            if line == '\n':
                continue
            else:
                stop_words.append(line)



    for i in range (len(stop_words)):
        stop_words[i] = stop_words[i].rstrip('\n')

    for word in stop_words:
        stop_words_capitalized.append(word.capitalize())

    print(stop_words_capitalized)

    for word in stop_words: 
        try:
            del dictionnary[word]
        except:
            print("Dictionnary has no key ", word)

    for word in stop_words_capitalized: 
        try:
            del dictionnary[word]
        except:
            print("Dictionnary has no key ", word)



print("sorting... ", end='')
sorted_x = sorted(dictionnary.items(), key=operator.itemgetter(1))
print("end")

print("writing... ", end='')
file = open("new2", "w")
for key in sorted_x :
    s = key[0] + " : " + str(key[1]) + "\n"
    file.write(s)
print("end")
