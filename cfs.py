# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:39:06 2024

@author: hucor
"""

import numpy as np
import random


#theta=np.pow(np.tranpose(X)*X,-1)*np.tranpose(X)*Y


def test():
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


data=test()
print(data)