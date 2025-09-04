# Installation : pip install transformers torch sentence-transformers
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import sys

class BaseModeleFrancais:
    """
    Modèle de base français - Fondation pour nos hémisphères
    """
    def __init__(self):
        # CamemBERT : le BERT français léger et performant
        self.model_name = "camembert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        print("🇫🇷 Modèle français chargé avec succès !")
        print(f"📊 Vocabulaire : {self.tokenizer.vocab_size} mots")
        print(f"🔧 Paramètres : ~110M")
    
    def encoder_texte(self, texte):
        """
        Encode un texte français en représentation vectorielle
        C'est la base de la compréhension linguistique
        """
        inputs = self.tokenizer(
            texte, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Moyenne des tokens pour avoir une représentation du texte complet
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
    
    def similarite_semantique(self, texte1, texte2):
        """
        Calcule la similarité sémantique entre deux textes
        Utile pour la compréhension contextuelle
        """
        emb1 = self.encoder_texte(texte1)
        emb2 = self.encoder_texte(texte2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity[0][0])
    
    def analyser_phrase(self, phrase):
        """
        Analyse basique d'une phrase française
        Premier pas vers la compréhension logique
        """
        # Tokenisation pour voir comment le modèle décompose
        tokens = self.tokenizer.tokenize(phrase)
        
        # Encoding pour obtenir la représentation vectorielle
        encoding = self.encoder_texte(phrase)
        
        return {
            'tokens': tokens,
            'nombre_tokens': len(tokens),
            'representation_vectorielle': encoding,
            'taille_representation': encoding.shape
        }
    
    def test_comprehension(self):
        """
        Test rapide pour vérifier que le modèle comprend le français
        """
        print("\n🧪 Test de compréhension française...")
        
        phrases_test = [
            "Le chat mange une souris",
            "Einstein était un génie",
            "2 + 2 = 4",
            "Je pense donc je suis"
        ]
        
        for phrase in phrases_test:
            analyse = self.analyser_phrase(phrase)
            print(f"📝 '{phrase}' -> {analyse['nombre_tokens']} tokens")
        
        # Test de similarité sémantique
        sim = self.similarite_semantique(
            "Le chat mange", 
            "L'animal se nourrit"
        )
        print(f"🔍 Similarité sémantique : {sim:.3f}")
        print("✅ Modèle français opérationnel !")

    def tester_comprehension_interactive(self):
        """
        Mode interactif pour tester la compréhension du modèle
        """
        print("\n🧠 Mode test de compréhension - Tapez 'quitter' pour arrêter")
        print("💡 Le modèle va analyser ce que vous dites mais ne peut pas répondre directement")
        
        while True:
            user_input = input("\nVous: ").strip()
            
            if user_input.lower() in ['quitter', 'exit', 'stop']:
                print("Test terminé !")
                break
            
            # Analyse de la phrase
            analyse = self.analyser_phrase(user_input)
            print(f"🔍 Analyse: {analyse['nombre_tokens']} tokens")
            print(f"📝 Tokens: {analyse['tokens'][:10]}...")  # Affiche les 10 premiers
            
            # Test de compréhension avec des phrases de référence
            phrases_reference = [
                "C'est une question",
                "C'est une affirmation", 
                "C'est une salutation",
                "C'est du contenu mathématique",
                "C'est de la philosophie",
                "C'est de la science"
            ]
            
            print("📊 Le modèle pense que c'est closest à :")
            scores = []
            for ref in phrases_reference:
                score = self.similarite_semantique(user_input, ref)
                scores.append((ref, score))
            
            # Trier par score descendant
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (ref, score) in enumerate(scores[:3]):
                print(f"  {i+1}. {ref} (score: {score:.3f})")

    def chat_simule(self):
        """
        Chat simulé : le modèle choisit la réponse la plus appropriée
        basée sur la compréhension sémantique
        """
        print("\n🤖 Mode chat simulé - Le modèle va 'répondre' par similarité")
        
        # Base de réponses avec contextes
        base_reponses = {
            "salutations": [
                "Bonjour ! Comment allez-vous ?",
                "Salut ! Que puis-je analyser pour vous ?"
            ],
            "questions_logique": [
                "C'est un problème logique intéressant.",
                "Laissez-moi analyser cette proposition."
            ],
            "mathematiques": [
                "Je vois des éléments mathématiques ici.",
                "C'est du calcul ou de la logique numérique."
            ],
            "philosophie": [
                "Voilà une réflexion philosophique profonde.",
                "Cette question touche aux concepts abstraits."
            ],
            "incomprehension": [
                "Je n'ai pas assez de contexte pour analyser cela.",
                "Pouvez-vous reformuler différemment ?"
            ]
        }
        
        while True:
            user_input = input("\nVous: ").strip()
            
            if user_input.lower() in ['quitter', 'exit']:
                print("Chat terminé !")
                break
            
            # Analyser le type de contenu
            meilleur_score = 0
            meilleure_categorie = "incomprehension"
            
            for categorie, reponses in base_reponses.items():
                if categorie != "incomprehension":
                    # Créer une phrase représentative de la catégorie
                    phrase_type = f"Ceci est lié à {categorie}"
                    score = self.similarite_semantique(user_input, phrase_type)
                    
                    if score > meilleur_score:
                        meilleur_score = score
                        meilleure_categorie = categorie
            
            # Seuil minimum pour éviter les réponses aléatoires
            if meilleur_score < 0.3:
                meilleure_categorie = "incomprehension"
            
            # Choisir une réponse aléatoire dans la catégorie
            import random
            reponse = random.choice(base_reponses[meilleure_categorie])
            
            print(f"Modèle: {reponse}")
            print(f"         (catégorie: {meilleure_categorie}, confiance: {meilleur_score:.3f})")

    def analyser_semantique_detaillee(self, phrase):
        """
        Analyse sémantique poussée d'une phrase
        """
        concepts_references = {
            "logique": "raisonnement logique déduction",
            "émotion": "sentiment émotion ressenti",
            "action": "faire agir mouvement",
            "description": "être état caractéristique",
            "question": "quoi comment pourquoi où",
            "science": "physique chimie biologie",
            "philosophie": "existence pensée conscience",
            "mathématiques": "nombre calcul équation"
        }
        
        print(f"\n🔬 Analyse sémantique de: '{phrase}'")
        
        scores_concepts = {}
        for concept, mots_cles in concepts_references.items():
            score = self.similarite_semantique(phrase, mots_cles)
            scores_concepts[concept] = score
        
        # Trier par score
        concepts_tries = sorted(scores_concepts.items(), key=lambda x: x[1], reverse=True)
        
        print("📈 Concepts détectés :")
        for concept, score in concepts_tries[:5]:
            barre = "█" * int(score * 20)  # Barre de progression visuelle
            print(f"  {concept:12} : {score:.3f} {barre}")
        
        return concepts_tries[0]  # Retourner le concept principal

    # Méthode pour utiliser l'analyse
    def mode_analyse(self):
        print("\n🔬 Mode analyse sémantique - Tapez 'quitter' pour arrêter")
        
        while True:
            user_input = input("\nPhrase à analyser: ").strip()
            
            if user_input.lower() in ['quitter', 'exit']:
                break
            
            concept_principal = self.analyser_semantique_detaillee(user_input)
            print(f"🎯 Concept principal détecté: {concept_principal[0]} ({concept_principal[1]:.3f})")



def main():
    print("🤖 IA de Raisonnement Autonome")
    
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Mode d'exécution (test/interac/simule/semantique ): ").strip().lower()
    
    if mode == "test":
        modele = BaseModeleFrancais()
        modele.test_comprehension()
    elif mode == "interac":
        modele = BaseModeleFrancais()
        modele.tester_comprehension_interactive()
    elif mode == "simule":
        modele = BaseModeleFrancais()
        modele.chat_simule()
    elif mode == "semantique":
        modele = BaseModeleFrancais()
        modele.mode_analyse()
    else:
        print("Mode non reconnu. Utilisez 'test ou interac ou simule ou semantique'")
        sys.exit(1)

if __name__ == "__main__":
    main()
