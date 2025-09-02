"""
RÉSEAU DE NEURONES FROM SCRATCH
===============================

Ce script implémente un réseau de neurones simple à partir de zéro 
pour vous aider à comprendre les mécanismes fondamentaux de l'IA.

Auteur: Script d'apprentissage IA
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ReseauNeurones:
    """
    Classe principale pour créer et entraîner un réseau de neurones simple

    Architecture supportée:
    - Couche d'entrée (nombre d'inputs variable)
    - Une couche cachée (taille configurable)
    - Couche de sortie (1 neurone pour classification binaire)
    """

    def __init__(self, nb_entrees: int, nb_neurones_caches: int, taux_apprentissage: float = 0.1):
        """
        Initialise le réseau de neurones

        Args:
            nb_entrees: Nombre d'entrées du réseau
            nb_neurones_caches: Nombre de neurones dans la couche cachée
            taux_apprentissage: Vitesse d'apprentissage (learning rate)
        """
        self.nb_entrees = nb_entrees
        self.nb_neurones_caches = nb_neurones_caches
        self.taux_apprentissage = taux_apprentissage

        # Initialisation des poids de manière aléatoire (valeurs petites)
        # Poids entre couche d'entrée et couche cachée
        self.poids_entree_cache = np.random.uniform(-1, 1, (self.nb_entrees, self.nb_neurones_caches))

        # Poids entre couche cachée et couche de sortie  
        self.poids_cache_sortie = np.random.uniform(-1, 1, (self.nb_neurones_caches, 1))

        # Biais pour chaque couche
        self.biais_cache = np.zeros((1, self.nb_neurones_caches))
        self.biais_sortie = np.zeros((1, 1))

        # Variables pour stocker les valeurs pendant la propagation
        self.entrees = None
        self.sortie_cache = None
        self.sortie_finale = None

        # Historique pour le graphique
        self.historique_erreur = []

    def fonction_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation sigmoïde

        Formule: f(x) = 1 / (1 + e^(-x))
        Sortie: valeurs entre 0 et 1
        """
        # Clip pour éviter les overflow
        x = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x))

    def derivee_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Dérivée de la fonction sigmoïde

        Formule: f'(x) = f(x) * (1 - f(x))
        Nécessaire pour la rétropropagation
        """
        return x * (1 - x)

    def fonction_relu(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation ReLU (Rectified Linear Unit)

        Formule: f(x) = max(0, x)
        Plus simple et souvent plus efficace que sigmoid
        """
        return np.maximum(0, x)

    def derivee_relu(self, x: np.ndarray) -> np.ndarray:
        """
        Dérivée de la fonction ReLU

        Formule: f'(x) = 1 si x > 0, sinon 0
        """
        return (x > 0).astype(float)

    def propagation_avant(self, entrees: np.ndarray) -> np.ndarray:
        """
        PROPAGATION AVANT (Forward Pass)

        Calcule la sortie du réseau étape par étape:
        1. Entrées → Couche cachée
        2. Couche cachée → Sortie finale

        Args:
            entrees: Données d'entrée (shape: [nb_exemples, nb_entrees])

        Returns:
            Sortie du réseau (shape: [nb_exemples, 1])
        """
        self.entrees = entrees
        # Couche cachée avec ReLU
        z_cache = np.dot(entrees, self.poids_entree_cache) + self.biais_cache
        self.sortie_cache = self.fonction_relu(z_cache)
        # Couche de sortie avec sigmoid
        z_sortie = np.dot(self.sortie_cache, self.poids_cache_sortie) + self.biais_sortie
        self.sortie_finale = self.fonction_sigmoid(z_sortie)

        return self.sortie_finale

    def retropropagation(self, entrees: np.ndarray, sorties_attendues: np.ndarray):
        """
        RÉTROPROPAGATION (Backward Pass)

        Calcule les erreurs et met à jour les poids:
        1. Calcule l'erreur en sortie
        2. Propage l'erreur vers les couches précédentes
        3. Met à jour tous les poids et biais

        Args:
            entrees: Données d'entrée
            sorties_attendues: Résultats attendus
        """
        nb_exemples = entrees.shape[0]

        # ÉTAPE 1: Calcul de l'erreur en sortie
        erreur_sortie = sorties_attendues - self.sortie_finale

        # Gradient pour la couche de sortie
        delta_sortie = erreur_sortie * self.derivee_sigmoid(self.sortie_finale)

        # ÉTAPE 2: Rétropropagation vers la couche cachée
        erreur_cache = delta_sortie.dot(self.poids_cache_sortie.T)
        # Dérivée de ReLU pour la couche cachée
        delta_cache = erreur_cache * self.derivee_relu(self.sortie_cache)

        # ÉTAPE 3: Mise à jour des poids et biais

        # Mise à jour poids couche cachée → sortie
        self.poids_cache_sortie += self.sortie_cache.T.dot(delta_sortie) * self.taux_apprentissage / nb_exemples
        self.biais_sortie += np.sum(delta_sortie, axis=0, keepdims=True) * self.taux_apprentissage / nb_exemples

        # Mise à jour poids entrée → couche cachée  
        self.poids_entree_cache += entrees.T.dot(delta_cache) * self.taux_apprentissage / nb_exemples
        self.biais_cache += np.sum(delta_cache, axis=0, keepdims=True) * self.taux_apprentissage / nb_exemples

    def calculer_erreur(self, sorties_predites: np.ndarray, sorties_attendues: np.ndarray) -> float:
        """
        Calcule l'erreur moyenne quadratique (MSE)

        Formule: MSE = (1/n) * Σ(y_réel - y_prédit)²
        """
        return np.mean((sorties_attendues - sorties_predites) ** 2)

    def entrainer(self, entrees: np.ndarray, sorties: np.ndarray, nb_epochs: int = 1000, 
                 afficher_progres: bool = True) -> List[float]:
        """
        ENTRAÎNEMENT DU RÉSEAU

        Répète le processus d'apprentissage sur plusieurs époques:
        1. Propagation avant
        2. Calcul de l'erreur
        3. Rétropropagation
        4. Mise à jour des poids

        Args:
            entrees: Données d'entraînement
            sorties: Résultats attendus
            nb_epochs: Nombre d'itérations d'entraînement
            afficher_progres: Afficher le progrès pendant l'entraînement

        Returns:
            Historique des erreurs
        """
        self.historique_erreur = []

        for epoch in range(nb_epochs):
            # Propagation avant
            predictions = self.propagation_avant(entrees)

            # Calcul de l'erreur
            erreur = self.calculer_erreur(predictions, sorties)
            self.historique_erreur.append(erreur)

            # Rétropropagation et mise à jour des poids
            self.retropropagation(entrees, sorties)

            # Affichage du progrès
            if afficher_progres and epoch % (nb_epochs // 10) == 0:
                print(f"Époque {epoch}/{nb_epochs} - Erreur: {erreur:.6f}")

        if afficher_progres:
            print(f"Entraînement terminé! Erreur finale: {self.historique_erreur[-1]:.6f}")

        return self.historique_erreur

    def predire(self, entrees: np.ndarray) -> np.ndarray:
        """
        Fait des prédictions sur de nouvelles données

        Args:
            entrees: Nouvelles données à classifier

        Returns:
            Prédictions (valeurs entre 0 et 1)
        """
        return self.propagation_avant(entrees)

    def classifier(self, entrees: np.ndarray, seuil: float = 0.5) -> np.ndarray:
        """
        Classifie les données en utilisant un seuil

        Args:
            entrees: Données à classifier
            seuil: Seuil de décision (par défaut 0.5)

        Returns:
            Classifications binaires (0 ou 1)
        """
        predictions = self.predire(entrees)
        return (predictions > seuil).astype(int)

    def afficher_architecture(self):
        """
        Affiche l'architecture du réseau
        """
        print("\n" + "="*50)
        print("ARCHITECTURE DU RÉSEAU DE NEURONES")
        print("="*50)
        print(f"Couche d'entrée:    {self.nb_entrees} neurones")
        print(f"Couche cachée:      {self.nb_neurones_caches} neurones (activation: sigmoid)")
        print(f"Couche de sortie:   1 neurone (activation: sigmoid)")
        print(f"Taux d'apprentissage: {self.taux_apprentissage}")
        print("="*50)

    def visualiser_apprentissage(self):
        """
        Affiche un graphique de l'évolution de l'erreur pendant l'entraînement
        """
        if not self.historique_erreur:
            print("Aucun entraînement effectué. Lancez d'abord la méthode entrainer().")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.historique_erreur, 'b-', linewidth=2, label='Erreur d\'entraînement')
        plt.title('Évolution de l\'erreur pendant l\'entraînement', fontsize=14, fontweight='bold')
        plt.xlabel('Époque', fontsize=12)
        plt.ylabel('Erreur (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.yscale('log')  # Échelle logarithmique pour mieux voir l'évolution
        plt.tight_layout()
        plt.show()


# =============================================================================
# EXEMPLES D'UTILISATION
# =============================================================================

def exemple_xor():
    """
    Exemple classique: Apprendre la fonction XOR

    XOR n'est pas linéairement séparable, donc impossible 
    à résoudre avec un perceptron simple. Nécessite une couche cachée.
    """
    print("\n" + "="*60)
    print("EXEMPLE 1: FONCTION XOR")
    print("="*60)

    # Données XOR
    # Entrées: [A, B] → Sortie: A XOR B
    entrees_xor = np.array([
        [0, 0],  # 0 XOR 0 = 0
        [0, 1],  # 0 XOR 1 = 1  
        [1, 0],  # 1 XOR 0 = 1
        [1, 1]   # 1 XOR 1 = 0
    ])

    sorties_xor = np.array([[0], [1], [1], [0]])

    print("Table de vérité XOR:")
    print("A | B | A XOR B")
    print("-" * 15)
    for i in range(len(entrees_xor)):
        a, b = entrees_xor[i]
        resultat = sorties_xor[i][0]
        print(f"{int(a)} | {int(b)} | {int(resultat)}")

    # Création et entraînement du réseau
    reseau = ReseauNeurones(nb_entrees=2, nb_neurones_caches=4, taux_apprentissage=1.0)
    reseau.afficher_architecture()

    print("\nEntraînement en cours...")
    reseau.entrainer(entrees_xor, sorties_xor, nb_epochs=5000)

    # Test des prédictions
    print("\nRésultats après entraînement:")
    print("Entrée -> Prédiction (Classification)")
    print("-" * 35)

    for i in range(len(entrees_xor)):
        entree = entrees_xor[i:i+1]  # Une ligne à la fois
        prediction = reseau.predire(entree)[0][0]
        classification = reseau.classifier(entree)[0][0]
        a, b = entrees_xor[i]
        print(f"[{int(a)}, {int(b)}] -> {prediction:.4f} ({int(classification)})")

    # Visualisation
    reseau.visualiser_apprentissage()

    return reseau

def exemple_et_ou():
    """
    Exemple simple: Apprendre les fonctions ET et OU

    Ces fonctions sont linéairement séparables, 
    plus faciles à apprendre.
    """
    print("\n" + "="*60)
    print("EXEMPLE 2: FONCTIONS ET & OU")
    print("="*60)

    # Données pour fonction ET
    entrees = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ])

    sorties_et = np.array([[0], [0], [0], [1]])  # Fonction ET
    sorties_ou = np.array([[0], [1], [1], [1]])  # Fonction OU

    print("\nFonction ET:")
    reseau_et = ReseauNeurones(nb_entrees=2, nb_neurones_caches=2, taux_apprentissage=0.5)
    reseau_et.entrainer(entrees, sorties_et, nb_epochs=1000)

    print("Résultats fonction ET:")
    for i in range(len(entrees)):
        entree = entrees[i:i+1]
        pred = reseau_et.predire(entree)[0][0]
        classif = reseau_et.classifier(entree)[0][0]
        a, b = entrees[i]
        print(f"[{int(a)}, {int(b)}] ET -> {pred:.4f} ({int(classif)})")

    print("\nFonction OU:")
    reseau_ou = ReseauNeurones(nb_entrees=2, nb_neurones_caches=2, taux_apprentissage=0.5)
    reseau_ou.entrainer(entrees, sorties_ou, nb_epochs=1000)

    print("Résultats fonction OU:")
    for i in range(len(entrees)):
        entree = entrees[i:i+1]
        pred = reseau_ou.predire(entree)[0][0]
        classif = reseau_ou.classifier(entree)[0][0]
        a, b = entrees[i]
        print(f"[{int(a)}, {int(b)}] OU -> {pred:.4f} ({int(classif)})")
        
    # Visualisation
    reseau_et.visualiser_apprentissage()
    reseau_ou.visualiser_apprentissage()

def exemple_donnees_aleatoires():
    """
    Exemple avec des données générées aléatoirement

    Montre comment le réseau peut apprendre des patterns 
    plus complexes.
    """
    print("\n" + "="*60)
    print("EXEMPLE 3: DONNÉES ALÉATOIRES")
    print("="*60)

    # Génération de données aléatoires
    np.random.seed(42)  # Pour reproductibilité
    nb_echantillons = 100

    # Données d'entrée: 2 features aléatoires
    entrees = np.random.randn(nb_echantillons, 2)

    # Fonction cible: classification basée sur une règle simple
    # Si x1 + x2 > 0, alors classe 1, sinon classe 0
    sorties = ((entrees[:, 0] + entrees[:, 1]) > 0).astype(int).reshape(-1, 1)

    print(f"Généré {nb_echantillons} échantillons avec 2 features")
    print(f"Règle de classification: classe 1 si (x1 + x2) > 0")

    # Division en ensembles d'entraînement et de test
    indices = np.random.permutation(nb_echantillons)
    split = int(0.8 * nb_echantillons)

    entrees_train = entrees[indices[:split]]
    sorties_train = sorties[indices[:split]]
    entrees_test = entrees[indices[split:]]
    sorties_test = sorties[indices[split:]]

    print(f"Entraînement: {len(entrees_train)} échantillons")
    print(f"Test: {len(entrees_test)} échantillons")

    # Entraînement
    reseau = ReseauNeurones(nb_entrees=2, nb_neurones_caches=5, taux_apprentissage=0.1)
    reseau.entrainer(entrees_train, sorties_train, nb_epochs=5000)

    # Évaluation
    predictions_test = reseau.classifier(entrees_test)
    precision = np.mean(predictions_test == sorties_test)

    print(f"\nPrécision sur l'ensemble de test: {precision:.2%}")

    # Quelques exemples de prédictions
    print("\nQuelques prédictions:")
    print("Entrée [x1, x2] -> Prédiction (Attendu)")
    print("-" * 40)
    for i in range(min(5, len(entrees_test))):
        x1, x2 = entrees_test[i]
        pred = predictions_test[i][0]
        attendu = sorties_test[i][0]
        print(f"[{x1:.2f}, {x2:.2f}] -> {pred} ({attendu})")
    
    reseau.visualiser_apprentissage()

def guide_utilisation():
    """
    Guide d'utilisation du script
    """
    print("\n" + "="*80)
    print("GUIDE D'UTILISATION - RÉSEAU DE NEURONES FROM SCRATCH")
    print("="*80)

    print("""
Ce script vous permet de créer et entraîner vos propres réseaux de neurones !

ÉTAPES POUR CRÉER VOTRE RÉSEAU:

1. CRÉER LE RÉSEAU
   reseau = ReseauNeurones(nb_entrees=2, nb_neurones_caches=4, taux_apprentissage=0.1)

2. PRÉPARER VOS DONNÉES
   entrees = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])     # Données d'entrée
   sorties = np.array([[0], [1], [1], [0]])                  # Résultats attendus

3. ENTRAÎNER LE RÉSEAU
   reseau.entrainer(entrees, sorties, nb_epochs=1000)

4. FAIRE DES PRÉDICTIONS
   nouvelles_donnees = np.array([[0.5, 0.3]])
   prediction = reseau.predire(nouvelles_donnees)
   classification = reseau.classifier(nouvelles_donnees)

PARAMÈTRES IMPORTANTS:

• nb_entrees: Nombre de caractéristiques de vos données
• nb_neurones_caches: Complexité du réseau (plus = plus puissant mais plus lent)
• taux_apprentissage: Vitesse d'apprentissage (0.01 à 1.0)
• nb_epochs: Nombre d'itérations d'entraînement

CONSEILS:
• Commencez avec des données simples (XOR, ET, OU)
• Augmentez progressivement la complexité
• Visualisez l'apprentissage avec reseau.visualiser_apprentissage()
• Expérimentez avec différents paramètres

EXEMPLES DISPONIBLES:
• exemple_xor() - Fonction XOR classique
• exemple_et_ou() - Fonctions logiques simples  
• exemple_donnees_aleatoires() - Données plus complexes
    """)

def main():
    """
    Fonction principale pour exécuter les exemples
    """
    print("BIENVENUE DANS VOTRE PREMIER RÉSEAU DE NEURONES!")
    print("Ce script vous apprend à créer une IA from scratch \n")

    # Affichage du guide
    guide_utilisation()

    # Exécution des exemples
    print("\n" + "="*80)
    print("EXÉCUTION DES EXEMPLES D'APPRENTISSAGE")
    print("="*80)

    # Exemple 1: XOR (le plus important)
    reseau_xor = exemple_xor()

    # Exemple 2: ET et OU
    exemple_et_ou()

    # Exemple 3: Données aléatoires
    exemple_donnees_aleatoires()

    print("\n" + "="*80)
    print("FÉLICITATIONS! 🎉")
    print("="*80)
    print("Vous avez créé et entraîné vos premiers réseaux de neurones!")
    print("Vous pouvez maintenant:")
    print("• Modifier les paramètres pour expérimenter")
    print("• Créer vos propres datasets")
    print("• Tester d'autres fonctions")
    print("\nBonne exploration de l'IA! 🚀")

if __name__ == "__main__":
    main()
