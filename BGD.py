# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:11:58 2024

@author: hucor
"""
import random 
import matplotlib.pyplot as plt



import numpy as np
def main():
    donnee = read_data()
    a = 10**-2
    itera = 0
    
    seuil=10**-2
    theta0 = random.uniform(0, 1)
    theta1 = random.uniform(0, 1)

    error = 10**5
    derror = 10**5
    
    while (derror > seuil):
        itera+=1
        for n in range(2):
            i = random.randint(1, 69)
            theta0 =(theta0 - a*((theta0 * donnee[i][0]) - donnee[i][1]) * donnee[i][0])
            theta1 =(theta1 - a*((theta1 * donnee[i][0])- donnee[i][1]) * donnee[i][0])

            erreur =  1/2*sum([(theta0 + theta1 * x1 - y1)**2 for x1, y1 in donnee])
        derror = derror - erreur
  
    print("Teta0: ", theta0)
    print("Teta1: ", theta1)
    print("Ereur: ", derror)
    
    donnee = np.array(donnee)

    # Extraire X et Y
    X = donnee[:, 0]
    Y = donnee[:, 1]

    # Créer les valeurs prédites par la fonction
    Y_pred = theta0 + theta1 * X

    # Tracer les points de données
    plt.scatter(X, Y, color='blue', label='Données réelles')

    # Tracer la fonction linéaire
    plt.plot(X, Y_pred, color='red', label='Ligne ajustée')

    # Ajouter des labels et une légende
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Afficher le graphique
    plt.show()



def read_data():
    # Ouvre le fichier text
    with open("data_lab1.txt", "r") as file:
        # Lit le fichier text
        data = file.read()
        # Split les données
        data = data.split("\n")
        # Crée une liste vide pour les données
        data_list = []
        # Pour chaque ligne dans les données
        i = 0
        for line in data:
            i += 1
            # Split les données
            line = line.split(" ")
            # Transforme la liste en liste de float
            line = list(map(float, line))
            # Ajoute la ligne à la liste de données
            data_list.append(line)
            if i == 70:
                break
    return data_list

main()