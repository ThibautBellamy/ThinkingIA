# main_finetuned.py
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AutoModel, AutoTokenizer
import torch
import numpy as np
import sys

class BaseModeleFrancaisFineTune:
    """
    Version améliorée qui utilise le modèle fine-tuné pour questions/affirmations
    """
    def __init__(self, model_path="./camembert-finetuned-questions"):
        # Charger le modèle fine-tuné pour classification
        print(f"🎯 Chargement du modèle fine-tuné depuis {model_path}")
        
        try:
            self.finetuned_tokenizer = CamembertTokenizer.from_pretrained(model_path)
            self.finetuned_model = CamembertForSequenceClassification.from_pretrained(model_path)
            print("✅ Modèle fine-tuné chargé avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle fine-tuné: {e}")
            print("📝 Utilisation du modèle de base à la place...")
            model_path = "camembert-base"
        
        # Charger aussi le modèle de base pour les embeddings
        self.base_tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        self.base_model = AutoModel.from_pretrained("camembert-base")
        
        print("🇫🇷 Système hybride initialisé !")
        print("   🎯 Classification questions/affirmations : Modèle fine-tuné")
        print("   🔍 Analyse sémantique générale : Modèle de base")
    
    def classifier_question_affirmation(self, phrase):
        """
        Classification précise avec le modèle fine-tuné
        """
        inputs = self.finetuned_tokenizer(
            phrase,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.finetuned_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # 0 = affirmation, 1 = question
        type_phrase = "question" if predicted_class == 1 else "affirmation"
        
        return {
            'type': type_phrase,
            'confiance': confidence,
            'score_question': probabilities[0][1].item(),
            'score_affirmation': probabilities[0][0].item(),
            'classe_predite': predicted_class
        }
    
    def encoder_texte(self, texte):
        """
        Encodage sémantique avec le modèle de base (pour autres analyses)
        """
        inputs = self.base_tokenizer(
            texte, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.base_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
    
    def similarite_semantique(self, texte1, texte2):
        """
        Calcule la similarité sémantique (version conservée)
        """
        emb1 = self.encoder_texte(texte1)
        emb2 = self.encoder_texte(texte2)
        
        similarity = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity[0][0])
    
    def analyser_phrase_complete(self, phrase):
        """
        Analyse complète : classification fine-tunée + analyse sémantique
        """
        # Classification question/affirmation avec fine-tuning
        classification = self.classifier_question_affirmation(phrase)
        
        # Tokenisation
        tokens = self.base_tokenizer.tokenize(phrase)
        
        # Concepts plus orientés "raisonnement" pour votre projet
        concepts_references = {
            "raisonnement_logique": "raisonnement logique déduction inférence",
            "questionnement": "question interrogation pourquoi comment",
            "affirmation_certitude": "affirmation certitude évident sûr",
            "analyse_critique": "analyser critiquer examiner évaluer", 
            "résolution_problème": "résoudre solution problème méthode",
            "mathématiques": "nombre calcul équation mathématique",
            "philosophie": "existence pensée conscience être",
            "action_concrète": "faire agir réaliser exécuter",
        }
        
        scores_concepts = {}
        for concept, mots_cles in concepts_references.items():
            score = self.similarite_semantique(phrase, mots_cles)
            scores_concepts[concept] = score
        
        # Trier par score
        concepts_tries = sorted(scores_concepts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'phrase': phrase,
            'classification': classification,
            'tokens': tokens,
            'nombre_tokens': len(tokens),
            'concepts': concepts_tries[:3],  # Top 3 concepts
            'concept_principal': concepts_tries[0]
        }
    
    def test_comparaison_avant_apres(self):
        """
        Compare les résultats avant/après fine-tuning
        """
        phrases_test = [
            "Qu'est-ce que tu penses de cette solution ?",
            "Comment résoudre ce problème complexe ?", 
            "Es-tu capable de raisonner logiquement ?",
            "Je pense que cette approche est correcte.",
            "Cette méthode fonctionne très bien.",
            "Il est évident que 2+2 fait 4.",
            "Peux-tu m'expliquer le raisonnement ?",
            "La logique dicte cette conclusion."
        ]
        
        print("\n🧪 TEST COMPARATIF - Modèle fine-tuné vs méthodes classiques")
        print("=" * 80)
        
        for phrase in phrases_test:
            print(f"\n📝 Phrase: '{phrase}'")
            
            # Analyse avec modèle fine-tuné
            analyse_complete = self.analyser_phrase_complete(phrase)
            classif = analyse_complete['classification']
            
            print(f"🎯 Fine-tuné : {classif['type']} (confiance: {classif['confiance']:.3f})")
            print(f"   Scores: Q={classif['score_question']:.3f} | A={classif['score_affirmation']:.3f}")
            
            # Analyse sémantique classique (pour comparaison)
            sim_question = self.similarite_semantique(phrase, "C'est une question")
            sim_affirmation = self.similarite_semantique(phrase, "C'est une affirmation")
            
            type_classique = "question" if sim_question > sim_affirmation else "affirmation"
            print(f"🔍 Classique: {type_classique} (Q={sim_question:.3f} | A={sim_affirmation:.3f})")
            
            # Concept principal détecté
            concept_principal = analyse_complete['concept_principal']
            print(f"🏷️  Concept: {concept_principal[0]} ({concept_principal[1]:.3f})")
    
    def mode_chat_ameliore(self):
        """
        Mode chat avec classification fine-tunée
        """
        print("\n🤖 Mode chat amélioré - Powered by Fine-Tuning!")
        print("💡 Le modèle utilise maintenant la classification fine-tunée")
        print("Tapez 'quitter' pour sortir\n")
        
        while True:
            user_input = input("Vous: ").strip()
            
            if user_input.lower() in ['quitter', 'exit']:
                print("Chat terminé !")
                break
            
            # Analyse complète
            analyse = self.analyser_phrase_complete(user_input)
            classif = analyse['classification']
            
            print(f"🎯 Type détecté: {classif['type']} (confiance: {classif['confiance']:.3f})")
            
            # Réponse adaptée au type
            if classif['type'] == 'question':
                if classif['confiance'] > 0.8:
                    print("Modèle: Je détecte une question claire. Laissez-moi analyser...")
                else:
                    print("Modèle: Cela semble être une question. Précisez si besoin.")
            else:  # affirmation
                if classif['confiance'] > 0.8:
                    print("Modèle: J'enregistre cette affirmation. Intéressant point de vue.")
                else:
                    print("Modèle: Je note cette information. Continuez...")
            
            # Concept principal
            concept = analyse['concept_principal']
            if concept[1] > 0.4:  # Seuil de pertinence
                print(f"💭 Concept détecté: {concept[0]} (score: {concept[1]:.3f})")

def main():
    print("🤖 IA de Raisonnement Autonome - Version Fine-Tunée")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Mode (test/comparaison/chat): ").strip().lower()
    
    # Initialisation du modèle fine-tuné
    try:
        modele = BaseModeleFrancaisFineTune()
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        sys.exit(1)
    
    if mode == "test":
        phrases_test = [
            "Qu'est-ce que tu penses ?",
            "Comment ça marche ?", 
            "Je pense que c'est bien.",
            "C'est vraiment intéressant.",
            "Es-tu d'accord ?",
            "Il faut que tu comprennes."
        ]
        
        print("\n🧪 Test rapide du modèle fine-tuné:")
        for phrase in phrases_test:
            resultat = modele.classifier_question_affirmation(phrase)
            print(f"'{phrase}' → {resultat['type']} ({resultat['confiance']:.3f})")
            
    elif mode == "comparaison":
        modele.test_comparaison_avant_apres()
        
    elif mode == "chat":
        modele.mode_chat_ameliore()
        
    else:
        print("Mode non reconnu. Utilisez 'test', 'comparaison' ou 'chat'")

if __name__ == "__main__":
    main()
