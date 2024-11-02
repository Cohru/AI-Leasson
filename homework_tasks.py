import numpy as np
import random
import environment as env
import matplotlib.pyplot as plt

# actual state space 
# (1,1),(2,1),(3,1),(4,1),(1,2),(2,2),(3,2),(4,2),(1,3),(2,3),(3,3),(4,3)
#   0     1     2     3     4     5     6     7     8     9    10    11

# action space
# N,S,E,W
# 0,1,2,3

np.random.seed(seed=1)

# Variables/hyperparamètres 
size_state_space = 12  # Utilisation de 12 états
size_action_space = 4  # 4 actions possibles (N, S, E, W)
n_episodes = 1000
factor_discount = 0.9  # Facteur de discount
eps = 0.1  # Epsilon-greedy
alpha = 0.1  # Taux d'apprentissage

# Initialisation de la fonction de valeur d'action Q
Q = np.zeros((size_state_space, size_action_space))

# Boucle d'apprentissage sur les épisodes
for i in range(n_episodes):
    t = 0
    s = random.randint(0, 11)  # Choisir un état initial aléatoire entre 0 et 11

    for j in range(1000):  # Limite à 1000 étapes par épisode
        # Choisir une action en suivant la stratégie epsilon-greedy
        if random.uniform(0, 1) < eps:
            a = random.randint(0, size_action_space - 1)  # Prendre une action aléatoire
        else:
            a = np.argmax(Q[s])  # Exploitation: Prendre l'action avec la valeur Q maximale

        # Appliquer l'action et obtenir le nouvel état et la récompense
        st1, rt = env.step(s, a)

        # Mise à jour de la valeur Q
        Q[s][a] = (1 - alpha) * Q[s][a] + alpha * (rt + factor_discount * np.max(Q[st1][a]))

        # Passer à l'état suivant
        s = st1
        t += 1


# Affichage des valeurs Q après apprentissage
print("Table des valeurs Q après apprentissage :")
print(Q)


V = np.max(Q, axis=1)

# Obtenir la politique optimale (pi*) en choisissant l'action avec la meilleure valeur Q pour chaque état
pi_star = np.argmax(Q, axis=1)

# Création de la carte pour les valeurs Q*
carte_Q = np.zeros((3, 4))
for i in range(0, 3):
    for j in range(0, 4):
        carte_Q[i][j] = V[i * 4 + j]

# Affichage de la carte des valeurs Q*
fig, ax = plt.subplots()
im = ax.imshow(carte_Q, cmap="viridis")

# Affichage des valeurs dans chaque case (arrondi à trois chiffres)
for i in range(0, 3):
    for j in range(0, 4):
        text = ax.text(j, i, f"{carte_Q[i][j]:.3f}", ha="center", va="center", color="w")

plt.colorbar(im)
plt.title("Carte des valeurs Q*")
plt.show()

# Création de la carte pour les politiques optimales (pi*) en utilisant les indices des actions optimales
carte_pi = np.zeros((3, 4), dtype=int)
for i in range(0, 3):
    for j in range(0, 4):
        carte_pi[i][j] = pi_star[i * 4 + j]

# Affichage de la carte des politiques optimales pi* (indices des actions)
fig, ax = plt.subplots()
im = ax.imshow(carte_pi, cmap="tab20b")

# Affichage des indices des actions dans chaque case
actions = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}  # Mapping des actions
for i in range(0, 3):
    for j in range(0, 4):
        text = ax.text(j, i, f"{actions[carte_pi[i][j]]}", ha="center", va="center", color="w")
for i in range(0, 3):
    for j in range(0, 4):
        text = ax.text(j, i, f"{carte_pi[i][j]}", ha="center", va="center", color="w")

plt.colorbar(im)
plt.title("Carte des indices de politiques optimales (pi*)")
plt.show()
