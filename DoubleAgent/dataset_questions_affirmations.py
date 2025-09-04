import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys

class DatasetCreator:
    """
    Créateur de dataset pour fine-tuner CamemBERT sur questions/affirmations
    """
    def __init__(self):
        # Dataset de base questions/affirmations français
        self.data_questions = [
            "Qu'est-ce que tu penses de cela ?",
            "Comment ça fonctionne ?",
            "Pourquoi est-ce important ?",
            "Où se trouve cette information ?",
            "Quand aura lieu cet événement ?",
            "Qui est responsable de ce projet ?",
            "Est-ce que c'est possible ?",
            "Peux-tu m'expliquer ?",
            "Comment puis-je t'aider ?",
            "Qu'est-ce qui se passe ?",
            "Y a-t-il des solutions ?",
            "Est-ce que tu comprends ?",
            "Comment résoudre ce problème ?",
            "Quelle est la différence ?",
            "Combien ça coûte ?",
            "Quel est le meilleur choix ?",
            "Comment prendre une décision ?",
            "Quels sont les critères à considérer ?",
            "Puis je te poser encore une question",
            "Est-ce que tu comprends",
            "Comment ça fonctionne",
            "Qui est responsable de ce projet",
            "Est-ce que tu me comprends",
            "Quand aura lieu cet événement",
        ]
        
        self.data_affirmations = [
            "Je pense que c'est une bonne idée.",
            "Cette solution fonctionne très bien.",
            "C'est vraiment important à retenir.",
            "Cette information se trouve dans le document.",
            "L'événement aura lieu demain matin.",
            "Marie est responsable de ce projet.",
            "C'est tout à fait possible à réaliser.",
            "Je vais t'expliquer le processus.",
            "Tu peux m'aider en faisant ceci.",
            "Il se passe quelque chose d'intéressant.",
            "Il y a plusieurs solutions disponibles.",
            "Je comprends parfaitement la situation.",
            "Voici comment résoudre ce problème.",
            "La différence est très claire.",
            "Le prix est de cinquante euros.",
            "Cet événement est très attendu.",
            "'Bonjour', n'est pas une phrase",
            "5 + 5 = 10",
        ]
    
    def generer_dataset_augmente(self, taille=1000):
        """
        Génère un dataset augmenté avec variations
        """
        import random
        
        # Modèles de questions français
        modeles_questions = [
            "Qu'est-ce que {} ?",
            "Comment {} ?", 
            "Pourquoi {} ?",
            "Est-ce que {} ?",
            "Peux-tu {} ?",
            "Y a-t-il {} ?",
            "Où {} ?",
            "Quand {} ?",
            "Qui {} ?",
            "Combien {} ?",
            "Qu'est-ce que {}",
            "Comment {}", 
            "Pourquoi {}",
            "Est-ce que {}",
            "Peux-tu {}",
            "Y a-t-il {}",
            "Où {}",
            "Quand {}",
            "Qui {}",
            "Combien {}",
        ]
        
        # Modèles d'affirmations
        modeles_affirmations = [
            "Je pense que {}.",
            "Il est évident que {}.",
            "On peut dire que {}.",
            "C'est certain que {}.",
            "Il faut savoir que {}.",
            "La réalité est que {}.",
            "Il s'avère que {}.",
            "On constate que {}.",
            "Il est clair que {}.",
            "Le fait est que {}.",
        ]
        
        fragments = [
            "c'est important", "ça marche bien", "tu comprends",
            "c'est possible", "il y a une solution", "ça coûte cher",
            "c'est difficile", "on peut réussir", "ça prend du temps",
            "c'est logique", "on doit essayer", "ça vaut le coup"
        ]
        
        dataset = []
        
        # Ajouter les données de base
        for q in self.data_questions:
            dataset.append({"text": q, "label": 1})  # 1 = question
            
        for a in self.data_affirmations:
            dataset.append({"text": a, "label": 0})  # 0 = affirmation
        
        # Générer des données augmentées
        for _ in range(taille - len(dataset)):
            fragment = random.choice(fragments)
            
            if random.random() < 0.5:  # 50% questions
                modele = random.choice(modeles_questions)
                text = modele.format(fragment)
                label = 1
            else:  # 50% affirmations
                modele = random.choice(modeles_affirmations)
                text = modele.format(fragment)
                label = 0
            
            dataset.append({"text": text, "label": label})
        
        # Mélanger
        random.shuffle(dataset)
        return dataset
    
    def sauvegarder_dataset(self, dataset, filename="./datasets/dataset_questions_affirmations.json"):
        """
        Sauvegarde le dataset au format JSON
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset sauvegardé : {len(dataset)} exemples dans {filename}")
        
        # Statistiques
        questions = sum(1 for item in dataset if item['label'] == 1)
        affirmations = len(dataset) - questions
        print(f"Questions: {questions}, Affirmations: {affirmations}")

# Créer le dataset
if __name__ == "__main__":
    creator = DatasetCreator()
    taille = 3000
    if len(sys.argv) > 1:
        try:
            taille = int(sys.argv[1])
        except ValueError:
            print("Usage: python dataset_creator.py [taille]")
            sys.exit(1)    
    dataset = creator.generer_dataset_augmente(taille)
    creator.sauvegarder_dataset(dataset)
