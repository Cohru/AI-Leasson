# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:33:20 2024

@author: hucor
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:25:29 2024

@author: hucor
"""

import math
import random
import numpy as np

def main():
    itera=0
    alpha = 1e-2
    alpha1 = 1e-2
    alpha2 = 1e-2
    threshold = 1e-2
    deltaerreur = 1e5
    term1 = 0
    term2 = 0
    e = 1e5
    # Lecture et conversion des données d'entrée en un tableau NumPy
    xbar = np.array(readData(), dtype=float)
    v = []
    w = []

    for j in range(3):
        v.append([random.uniform(0, 1) for i in range(3)])
    for k in range(4):
        w.append([random.uniform(0, 1) for i in range(3)])

    v = np.array(v, dtype=float)
    w = np.array(w, dtype=float)

    # Matrix multiplication of xbar and v
    xbarbar = np.dot(xbar, v)

    # Création de la matrice F avec fonction d'activation
    f = 1 / (1 + np.exp(-xbarbar))

    # Ajouter une colonne de biais à F
    fbar = np.insert(f, 0, 1, axis=1)

    # Calcul de la matrice Fbarbar (sortie de la couche cachée)
    fbarbar = np.dot(fbar, w)
    g = 1 / (1 + np.exp(-fbarbar))  # Appliquer la fonction d'activation sigmoïde

    y = readY()
    print(y)
    # Calcul de l'erreur initiale
    # for i in range(len(g)):
    #     for j in range(len(g[i])):
    #         e += (g[i][j] - tab_y[i][j]) ** 2
    # e /= 2
    
    tab_y=[]
    for i in range(len(y)):
        if (y[i] == 0):
            tab_y.append([1,0,0])
        elif (y[i] == 1):
            tab_y.append([0,1,0])
        else:
            tab_y.append([0,0,1])



    while abs(deltaerreur) > threshold:
        itera+=1
        for k in range(len(w)): 
            for j in range(len(w[k])):
                gradient_sum = 0
                for i in range(len(g)):  
                    term1 = (g[i][j] - tab_y[i][j]) * g[i][j] * (1 - g[i][j]) * fbar[i][k]
                    gradient_sum += term1
            
                w[k][j] -= alpha1 * gradient_sum

        for n in range(len(v)):  
            for k in range(len(v[0])):
                gradient_sum = 0
                for i in range(len(g)):
                    for j in range(len(g[i])): 
                        term2 = (g[i][j] - tab_y[i][j]) * g[i][j] * (1 - g[i][j]) * w[k+1][j] * f[i][k] * (1 - f[i][k]) * xbar[i][n]
                        gradient_sum += term2
                
                v[n][k] -= alpha2 * gradient_sum



        xbarbar = np.dot(xbar, v)
        f = 1 / (1 + np.exp(-xbarbar))
        fbar = np.insert(f, 0, 1, axis=1)
        fbarbar = np.dot(fbar, w)
        g = 1 / (1 + np.exp(-fbarbar))
        
        
        
        e_avant= e
        for i in range(len(g)):
            for j in range(len(g[i])):
                e += (g[i][j] - tab_y[i][j]) ** 2
        e /= 2
        deltaerreur = e - e_avant
   
    
    ypred = []
    for i in range(len(g)):
        max_val = g[i][0] 
        max_index = 0 
        for j in range(1, len(g[i])):
            if g[i][j] > max_val:
                max_val = g[i][j]
                max_index = j
    
        ypred.append(max_index)
            
    print(ypred)
def readY():
    with open("data_ffnn_3classes.txt", "r") as file:
        file.readline()  # Skip the first line (header)
        y = [float(line.split()[2]) for line in file]  # Extraire la valeur cible sous forme de float
    return y

def readData():
    with open("data_ffnn_3classes.txt", "r") as file:
        file.readline()  # Skip the first line (header)
        data = [line.split() for line in file]
    
    xbar = [[1, float(row[0]), float(row[1])] for row in data]  # Conversion explicite en float
    return xbar

main()
