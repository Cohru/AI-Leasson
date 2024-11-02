# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:58:01 2024

@author: hucor
"""

import random
import matplotlib.pyplot as plt
import numpy as np

def main():
    donnees = read_data()

    # Paramètres d'entrainement
    alpha = 10**-2
    threshold = 10**-2
    deltaErreur = 10**5
    itera = 0
    erreur = 10**5

    # Initialisation des poids
    teta0 = random.uniform(0, 1)
    teta1 = random.uniform(0, 1)

    while deltaErreur > threshold:
        itera = itera + 1

        grad0 = sum([(teta0 + teta1 * x1 - y1) for x1, y1 in donnees])
        grad1 = sum([(teta0 + teta1 * x1 - y1) * x1 for x1, y1 in donnees])

        teta0 = teta0 - alpha * grad0
        teta1 = teta1 - alpha * grad1

        erreur = 1/2*sum([(teta0 + teta1 * x1 - y1)**2 for x1, y1 in donnees])

        deltaErreur = deltaErreur - erreur

    print("Teta0: ", teta0)
    print("Teta1: ", teta1)
    print("Erreur: ", erreur)

    # Convertir en tableau NumPy
    donnees = np.array(donnees)

    # Extraire X et Y
    X = donnees[:, 0]
    Y = donnees[:, 1]

    # Créer les valeurs prédites par la fonction
    Y_pred = teta0 + teta1 * X

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