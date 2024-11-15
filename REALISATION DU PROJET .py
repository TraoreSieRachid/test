# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:03:58 2024

@author: hp
"""
############################## Partie I: ########################################################################

#************************* EXERCICE 1: ***************************************
print("#*** EXERCICE 1: ***\n")
def RESULTAT(T):# fonction retournant le reultat voulu
    n=len(T)
    nb_zeros=0
    # Calculons le nombre de zeros dabs le tableau
    for i in range(0,n):
        if T[i]==0: # verification si zero
            nb_zeros+=1 # augmente d'un ,la variable nb_zeros       
    # deplacement des zeros 0 dans les dernieres positions 
    i=0
    while i <n-nb_zeros:
        while T[i]==0: 
            del T[i]
            T.append(0)
        i+=1
    return T # Retourne le resultat

# Entrée des éléments du tableau
T=[1,2,0,6,9,0,0,8,0,3] 
# AFFICHAGE DES RESULTATS
print("EXEMPLE:",T)
print("RESULTAT:",RESULTAT(T))

#*********************** EXERCICE 2: ******************************************
print("#**** EXERCICE 2: ***\n")
def traitement(T):
    n=len(T)
    # Définition des constantes au debut du tableau 
    nb_souseq=1
    dlong=1
    flong=1
    d=1
    f=1
    info=[nb_souseq,dlong,flong]
    # Calcul des nombres de sous sequences dans le tableau
    for i in range(1,n):
        # Calcul des nombres de sous sequences dans le tableau
        if T[i-1]<T[i]: # element suivant superieur au precedent
            f+=1 # f,la position du dernier élément de la sous-séquence
            
        else: # sinon, une nouvelle sous sequence
            d=i+1 # un nouveau début et fin
            f=i+1
            nb_souseq+=1 # augmente le nombre de sous-sequences

        # verification si la nouvelle sous-sequence plus longue que l'ancienne
        if flong-dlong<f-d: 
            dlong=d # definition du debut et de la fin de la plus long sous sequence à l'étape i
            flong=f
        info=[nb_souseq,dlong,flong]
    return  info# retourne le nombre de sous-séquences , du début et de la fin de la plus long sous-séquence

# Entrée des éléments du tableau
T=[1,2,5,3,12,25,13,8,4,7,24,28,32,11,14]
# AFFICHAGE DES RESULTATS
print("EXEMPLE T=:",T)
print("RESULTAT: Le nombre de sous-séquences de:",T,"est ",traitement(T)[0],"sous-séquences"," et la position de début de la plus longue sous-séquence est",traitement(T)[1],"et la position de fin est",traitement(T)[2],".\n")


#***************************** Exercice 3: ****************************************
print("#**** EXERCICE 3: ***\n")

def calculate_sinus(x,N):
    result = x
    for n in range (1,N+1):
        sign=-1
        term_numerator=1
        term_denominator=1
        for i in (1,2*n+1):
            term_numerator*=x
            term_denominator*=i
        current_term = sign * term_numerator / term_denominator
        result+=current_term
        current_term=0
    return result


 # Test
 
x =float(input("Entrez la valeur dont le sinus est à calculer:"))  # Par exemple, pour x = 30 degrés
nombre_iteration =int(input("Entrez le nombre d'itérations:"))    # Précision souhaitée
result = calculate_sinus(x,nombre_iteration)

print("le sin(",x,") est  approximativement: ",result)


def calculate_pi(precision):
    result = 0
    sign = 1
    n = 0

    while True:
        term = sign / (2 * n + 1)
        result += term
        sign *= -1
        n += 1

        if abs(term) < precision:
            break
        
    return result * 4

# Test avec une précision de 1e-6 
precision = 1e-6
pi_approximation = calculate_pi(precision)
print("Approximation de π: ",pi_approximation)

#****************************** Exercice 4: **********************************
print("#**** EXERCICE 4: ***\n")

#1- le reste de la division entière par 10
def reste_division_par_10(dividende):
    while dividende >= 10:
        dividende -= 10
    return dividende

# Exemple d'utilisation
nombre = int(input("Entrez un nombre : "))
reste = reste_division_par_10(nombre)
print("Le reste de la division de ",nombre," par 10 est : ",reste)


#2-calcule le nombre de chiffres que contient un entier n:

def somme_chiffres(n):
    somme = 0
    
    # Utilisation d'une boucle while pour extraire chaque chiffre
    while n > 0:
        chiffre = n % 10  #  le dernier chiffre
        somme += chiffre  # Ajout à la somme
        n //= 10         # Suppression du dernier chiffre

    return somme

# Exemple d'utilisation
entier = 12345
resultat = somme_chiffres(entier)

print(f"La somme des chiffres de {entier} est {resultat}")
   
#tester si n est divisible par 9 en utilisant la somme des chiffres

def divisible_par_neuf(n):
    # la somme des chiffres sans convertir en chaîne de caractères
    somme = 0
    
    #  une boucle while pour extraire chaque chiffre
    while n > 0:
        chiffre = n % 10  # le dernier chiffre
        somme += chiffre  # Ajout à la somme
        n //= 10         # Suppression le dernier chiffre

    #  si la somme est divisible par 9
    return somme % 9 == 0

# Exemple d'utilisation
nombre = 123456
if divisible_par_neuf(nombre):
    print(f"{nombre} est divisible par 9.")
else:
    print(f"{nombre} n'est pas divisible par 9.")


#*********************** Exercice 5 *******************************************
print("#**** EXERCICE 5: ***\n")


def moyenne_eleve(notes):
    moyennes = []
    
    for ligne in notes:
        somme_notes = sum([note * (i + 1) for i, note in enumerate(ligne)])
        coefficient_total = sum(range(1, len(ligne) + 1))
        moyenne_eleve = somme_notes / coefficient_total
        moyennes.append(moyenne_eleve)
    
    return moyennes

def eleves_inf_moyenne_classe(notes, moyenne_classe):
    nombre_eleves_inf = sum(1 for moyenne in moyenne_eleve(notes) if moyenne < moyenne_classe)
    return nombre_eleves_inf

# Exemple d'utilisation
notes_classe = [
    [18, 15, 14],
    [15, 15, 15],
    [10, 17, 18],
    
]

moyenne_classe = sum(sum(note * (i + 1) for i, note in enumerate(ligne)) / sum(range(1, len(ligne) + 1)) for ligne in notes_classe) / len(notes_classe)

moyennes_eleves = moyenne_eleve(notes_classe)
nombre_eleves_inf = eleves_inf_moyenne_classe(notes_classe, moyenne_classe)

print("Moyennes des élèves:", moyennes_eleves)
print("Moyenne de la classe:", moyenne_classe)
print("Nombre d'élèves avec une moyenne inférieure à celle de la classe:", nombre_eleves_inf)

############################## Partie II:########################################################################

#****************************** NUMPY *************************************#


import numpy as np

# Étape 1
tableau_pairs = [i for i in range(2, 201, 2)]
print("Étape 1:\n", tableau_pairs)

# Étape 2
tableau_multiplication = np.array([[i * j for j in range(1, 11)] for i in range(1, 11)])
print("\nÉtape 2:\n", tableau_multiplication)

# Étape 3
masque_diagonale = np.eye(10, dtype=bool)  # Création d'un masque booléen pour la diagonale
diagonale = tableau_multiplication[masque_diagonale]
print("\nÉtape 3 - Diagonale:", diagonale)

# Étape 4
tableau_random_impairs = np.random.choice(np.arange(1, 101, 2), size=10)
print("\nÉtape 4 - Nombres impairs:", tableau_random_impairs)

# Étape 5
tableau_random = np.random.choice(np.arange(51, 101), size=10)
plus_petit = min(tableau_random)
print("\nÉtape 5 - tableau:", tableau_random)
print("\nÉtape 5 - Plus petit nombre > 50:", plus_petit)

# Étape 6
tableau_random_grands = np.random.choice(np.arange(1, 101), size=10)
cinq_plus_grands = np.partition(tableau_random_grands, -5)[-5:] # np.partition est utilisé pour trouver les cinq plus grands nombres du tableau en faisant une partition partielle
print("\nÉtape 6 - le tableau:", tableau_random_grands)
print("\nÉtape 6 - Cinq plus grands nombres:", cinq_plus_grands)


#**************************** PANDAS *****************************************#
#A)#
# Étape 1
import pandas as pd

# Étape 2 et 3
df = pd.read_csv('C:\\Users\\hp\\Documents\\bibliothèque de travail\\ELEVE INGENIEUR\\ISE\\Cours\\informatique\\tp_python-3\\student.csv')

# Étape 4
# Affichez les variables (colonnes) du DataFrame
variables = df.columns
print(variables)

data = df.loc[:, 'school':'guardian']

# Étape 5
majuscule_lambda = lambda x: x.upper()

# Étape 6
data['Mjob'] = data['Mjob'].apply(majuscule_lambda)
data['Fjob'] = data['Fjob'].apply(majuscule_lambda)

# Étape 7
derniers_elements = df.tail()
print("Étape 7 - Derniers éléments de l'ensemble de données:\n", derniers_elements)

# Étape 8
df
df['Mjob'] = df['Mjob'].apply(majuscule_lambda)
df['Fjob'] = df['Fjob'].apply(majuscule_lambda)
# Affichage du DataFrame après les modifications
df

# Étape 9
def est_majeur(age):
    return age > 17

df['legal_drinker'] = df['age'].apply(est_majeur)
df


#B)#
# Étape 1. les bibliothèques nécessaires
import pandas as pd

# Étape 2.  les 3 DataFrames 
raw_data_1 = {
    'subject_id': ['1', '2', '3', '4', '5'],
    'prénom': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'nom de famille': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}

raw_data_2 = {
    'subject_id': ['4', '5', '6', '7', '8'],
    'prénom': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
    'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}

raw_data_3 = {
    'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
    'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}

# Étape 3.  data1, data2, data3
data1 = pd.DataFrame(raw_data_1)
data2 = pd.DataFrame(raw_data_2)
data3 = pd.DataFrame(raw_data_3)

# Étape 4.  les deux dataframes le long de lignes 
all_data = pd.concat([data1, data2])
all_data

# Étape 5.  les deux dataframes le long des colonnes 
all_data_col = pd.concat([data1, data2], axis=1)
all_data_col

# Étape 6.  les données data3
print("Données data3:")
print(data3)

# Étape 7.  all_data et data3 
data = pd.merge(all_data, data3, on='subject_id')
print("Données fusionnées avec data3:")
print(data)

# Étape 8. 
merged_data = pd.merge(data1, data2, on='subject_id')
print("Données fusionnées sur subject_id:")
print(merged_data)

# Étape 9. 
merged_data_1 = pd.merge(data1, data2, on='subject_id', how='outer')
print("Données fusionnées avec toutes les valeurs:")
print(merged_data_1)


#C) FILTRAGE ET TRI DE DONNEES

#Etape 1
import pandas as pd

#Etape 2 #Etape 3

euro12=pd.read_csv("C:\\Users\\hp\\Documents\\bibliothèque de travail\\ELEVE INGENIEUR\\ISE\\Cours\\informatique\\tp_python-3\\euro_12.csv")

# #Etape 4
Objectif=euro12["Goals"]
print(Objectif)

#Etape 5

nb=len(euro12["Team"])
print("Le nombre d'équipes d'equipes ont participé à l'euro2012 est:",nb,"équipes")

#Etape 6

ncolumns=euro12.shape[1]
print("le nombre de colonnes dans l'ensemble de données est:",ncolumns)

#Etape 7

print(euro12["Team"])
print(euro12["Yellow Cards"])
print(euro12["Red Cards"])
discipline=euro12[["Team","Yellow Cards","Red Cards"]]
print(discipline)

#Etape 8

discipline=discipline.sort_values(by="Red Cards",ascending=True)
print(discipline)

discipline=discipline.sort_values(by="Yellow Cards",ascending=True)
print(discipline)

#Etape 9

moyenne=discipline["Yellow Cards"].mean()
print(moyenne)

#Etape 10

equip_pl_6but=euro12.query("Goals>6")[["Team","Goals"]]
print(equip_pl_6but)

equip_pl_6but=euro12.loc[euro12["Goals"]>6,["Team","Goals"]]
print(equip_pl_6but)

#Etape 11

equip_cm_G=euro12.loc[euro12["Team"].str.startswith("G"),"Team"]
print(equip_cm_G)

#Etape 12

colon_7_pre=euro12.iloc[:,:7]
print(colon_7_pre.to_string())

#Etape 13

colon_sauf_3d=euro12.iloc[:,:33]
print(colon_sauf_3d.to_string())

#Etape 14


prec_tir=euro12.loc[euro12["Team"].isin(["England","Italy","Russia"]),["Team","Shooting Accuracy"]]
print(prec_tir.to_string())


#E) VISUALISATION

#Etape 1

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Etape 2 #Etape 3

online_rt=pd.read_csv("C:\\Users\\hp\\Documents\\bibliothèque de travail\\ELEVE INGENIEUR\\ISE\\Cours\\informatique\\tp_python-3\\vente_detail.csv")

#Etape 4

pays_qut=online_rt.groupby("Country")["Quantity"].sum().sort_values(ascending=False).iloc[0:11]
pays_qut.drop("United Kingdom",axis=0,inplace=True)
pays_qut.plot.bar(rot=45)
plt.xlabel('Countries')
plt.ylabel('Quantity')
plt.title('The 10 Countries with most orders')

plt.show()

#Etape 5

online_rt=online_rt[online_rt["Quantity"]>=0]

#Etape 6,7
top_countries = ['Netherlands', 'EIRE', 'Germany']
for country in top_countries:
    group=online_rt[online_rt["Country"]==country]
    plt.scatter(x=group.UnitPrice,y=group.Quantity)
    plt.xlabel("LE PRIX MOYEN PAR LE CLIENT")
    plt.ylabel("LA QUANTITE COMMANDEE PAR UN CLIENT")
    plt.title(country)
    plt.show() 
#Etape 7 1 1

print(online_rt["UnitPrice"].head())

#Etape 7 1 2

print(online_rt["UnitPrice"].dtypes)

#Etape 7 1 3

client_12346=online_rt[online_rt["CustomerID"]==12346]
client_12347=online_rt[online_rt["CustomerID"]==12347]
print(client_12346)
print(client_12347)

#Etape 7 2 1

Top3_pays=online_rt.groupby("Country")["Quantity"].sum().sort_values(ascending=False).head(3)
print(Top3_pays)


#Etape 7 3 
#Etape 7 3 1

online_rt["Revenue"]=online_rt["Quantity"]*online_rt["UnitPrice"]
print(online_rt)

#Etape 7 3 2 et 7 3 3
for country in top_countries:
    Avg_price=online_rt[online_rt["Country"]==country].groupby("CustomerID")["Revenue"].sum()/sum(online_rt[online_rt["Country"]==country].groupby("CustomerID")["Quantity"].sum())
    Quant=online_rt[online_rt["Country"]==country].groupby("CustomerID")["Quantity"].sum()
    plt.scatter(x=Avg_price,y=Quant)
    plt.xlabel("LE PRIX MOYEN PAR LE CLIENT")
    plt.ylabel("LA QUANTITE COMMANDEE PAR UN CLIENT")
    plt.title(country)
    plt.show()


# #Etape 7.4.1

grouped = online_rt.groupby(['CustomerID'])
plottable = grouped['Quantity','Revenue'].agg('sum')
plottable['AvgPrice'] = plottable.Revenue / plottable.Quantity
plt.scatter(plottable.Quantity,plottable.AvgPrice)
plt.xlabel("LE PRIX MOYEN PAR LE CLIENT")
plt.ylabel("LA QUANTITE COMMANDEE PAR UN CLIENT")
plt.show()

# #Etape 7.4.2

grouped = online_rt.groupby(['CustomerID','Country'])
plottable = grouped.agg({'Quantity':'sum', 'Revenue':'sum'})
plottable['AvgPrice'] = plottable.Revenue / plottable.Quantity
plt.scatter(plottable.Quantity, plottable.AvgPrice)
plt.xlabel("LE PRIX MOYEN PAR LE CLIENT")
plt.ylabel("LA QUANTITE COMMANDEE PAR UN CLIENT")
plt.xlim(0,2500)
plt.ylim(0,10)
plt.show()

#Etape 8.1
price_start = 0
price_end = 50
price_interval = 1
buckets = np.arange(price_start,price_end,price_interval)
revenue_per_price = online_rt.groupby(pd.cut(online_rt.UnitPrice, buckets)).Revenue.sum()
revenue_per_price.head()

#Etape 8.3
revenue_per_price.plot()
plt.xlabel('Unit Price (in intervals of '+str(price_interval)+')')
plt.ylabel('Revenue')
plt.show()

# #Etape 8.4
revenue_per_price.plot()
plt.xlabel('Unit Price (in buckets of '+str(price_interval)+')')
plt.ylabel('Revenue') 
plt.xticks(np.arange(price_start,price_end,3),np.arange(price_start,price_end,3))
plt.yticks([0, 500000, 1000000, 1500000, 2000000, 2500000],['0', '$0.5M', '$1M', '$1.5M', '$2M', '$2.5M'])
plt.show()

# BONUS : Créez votre propre question et répondez-y.
# Par exemple : "Quels sont les articles les plus vendus (en termes de quantité) dans chaque pays ?"
top_items_by_country = online_rt.groupby(['Country', 'Description']).agg({'Quantity': 'sum'}).reset_index()
top_items_by_country = top_items_by_country.groupby('Country').apply(lambda x: x.loc[x['Quantity'].idxmax()])
print(top_items_by_country)