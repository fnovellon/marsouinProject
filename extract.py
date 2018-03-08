import csv, sys
Text=[]
Poid=[]
TexTPoid=[]

taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))

while(taille<1 or taille>8000):
	taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))

with open("dataset.csv") as csvfile:
	#Text=csvfile.read().split(' \n')
	for i in range(taille):
		Text+=[csvfile.readline()]

with open("labels.csv") as csvfile:
	for i in range(taille):
		Poid+=[csvfile.readline()]

TexTPoid= zip(Text,Poid)


for i in range(taille):
	print(Text[i],Poid[i])




	'''for line in csvfile:
		if(i==10):
			sys.exit(0)
		i=i+1
		Text[i]=csvfile.readline()
		print(format(csvfile.readline()))
'''