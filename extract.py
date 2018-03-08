import csv, sys


''' prends les données du csv et les mets dans une liste '''

def put_texts_to_list( csvfile1 , taille):
	
	texts = []

	with open( csvfile1 ) as csvfile: 
		for i in range(taille) : 
			texts += [csvfile.readline()]

		return texts;

''' Fonction load_from_csv prend en paramètre les nombes des fichiers csv à concaténer ligne par ligne ainsi que le nombre de ligne à concaténer et renvoie une liste 2 dimensions TEXT et POID à partir de ces deux fichier '''

def load_from_csv(csvfile1, csvfile2, taille):

	Text=[]
	Poid=[]
	TexTPoid=[]

	Text = put_texts_to_list( csvfile1 , taille )

	Poid = put_texts_to_list( csvfile2 , taille )

	TexTPoid = list(zip(Text, Poid))

	return TexTPoid;

''' prends une liste deux dimensions reçue par load_from_csv et un indice, retour les mots présents à cet indice '''

def get_words ( list_words , indicator ): 

	items = list_words[indicator][0]

	words = []

	words = items.split()

	return words;

	
	
taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))

while(taille<1 or taille>8000):
	taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))


result = []
result = load_from_csv("dataset.csv", "labels.csv", taille)

offset = taille - 1

print("Entrez l'indice du commentaire que vous voulez : 0 à ", offset)
indice=int(input())

while(indice < 0 or indice > taille):
	print("Entrez l'indice du commentaire que vous voulez : 0 à ", offset)
	indice=int(input())
	
words = []

words = get_words(result, indice)

for i in words: 
	print(i)





