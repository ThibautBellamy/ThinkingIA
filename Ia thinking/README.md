# 🤖 IA de Raisonnement Autonome

Une intelligence artificielle capable d'apprentissage auto-supervisé et de raisonnement autonome avec complexité croissante.

## 🏗️ Architecture du Projet

```
autonomous_reasoning_ai/
├── main.py                          # Point d'entrée principal
├── config.py                        # Configuration globale
├── requirements.txt                 # Dépendances Python
├── models/
│   ├── __init__.py
│   └── reasoning_core.py            # Cœur du raisonnement autonome
├── problem_generators/
│   ├── __init__.py
│   ├── base_generator.py            # Classe de base pour générateurs
│   ├── math_generator.py            # Problèmes mathématiques
│   ├── logic_generator.py           # Problèmes de logique
│   └── pattern_generator.py         # Reconnaissance de patterns
├── training/
│   ├── __init__.py
│   └── autonomous_trainer.py        # Entraînement auto-supervisé
├── utils/
│   ├── __init__.py
│   └── validation.py                # Utilitaires de validation
└── experiments/
    ├── __init__.py
    └── experiment_runner.py          # Gestion des expérimentations
```

## 🚀 Installation et Utilisation

### 1. Activation de l'environnement virtuel
```bash
venv_ia\Scripts\activate  # Windows
```

### 2. Test du système
```bash
python -m autonomous_reasoning_ai.main test
```

### 3. Entraînement complet
```bash
python -m autonomous_reasoning_ai.main train
```

### 4. Test rapide avec script
```bash
python run_ai.py test
python run_ai.py train
```

## 🧠 Fonctionnalités Principales

### 🔄 Raisonnement Itératif
- L'IA réfléchit étape par étape jusqu'à convergence
- Auto-contrôle de la profondeur de raisonnement
- Mécanisme d'attention avec mémoire de travail

### 📚 Apprentissage Auto-Supervisé
- Pas besoin de réponses externes
- Apprentissage par consistance des réponses
- Auto-évaluation de la confiance

### 📈 Curriculum Learning
- Progression automatique de la complexité
- Domaines multiples : math, logique, patterns
- Adaptation basée sur les performances

### 🎯 Génération de Problèmes
- **Mathématiques** : Arithmétique → Équations → Polynômes
- **Logique** : Syllogismes → Logique propositionnelle → Puzzles
- **Patterns** : Séquences → Matrices → Fractales

## 📊 Métriques de Performance

- **Accuracy** : Pourcentage de bonnes réponses
- **Consistency** : Cohérence entre plusieurs tentatives
- **Confidence** : Auto-évaluation de la certitude
- **Reasoning Depth** : Nombre d'étapes de raisonnement
- **Convergence Time** : Vitesse de résolution

## 🛠️ Configuration

Voir `autonomous_reasoning_ai/config.py` pour personnaliser :
- Dimensions du modèle
- Paramètres d'entraînement
- Complexité maximale
- Métriques à suivre

## 📁 Fichiers Utilitaires

- `test_new_architecture.py` : Tests de validation de l'architecture
- `run_ai.py` : Script de lancement simplifié
- `checkpoints/` : Sauvegardes des modèles
- `logs/` : Logs TensorBoard
- `results/` : Résultats d'expérimentations

## 🎯 Innovation Clé

**L'IA développe sa propre logique de raisonnement sans supervision externe**, en s'appuyant sur :
1. **Consistance** de ses réponses multiples
2. **Confiance** dans ses prédictions  
3. **Validation** automatique pour les domaines formels
4. **Progression** adaptative de la complexité

---

*Développé avec PyTorch 2.8.0 et Python 3.13*
