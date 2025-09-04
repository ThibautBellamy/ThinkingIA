# main_double_finetuning.py
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AutoModel, AutoTokenizer
import torch
import numpy as np
import sys

class BaseModeleFrancaisDoubleTuning:
    """
    Modèle avec DOUBLE fine-tuning : questions/affirmations + concepts
    """
    def __init__(self, 
                 model_questions_path="./finetuned/camembert-finetuned-questions",
                 model_concepts_path="./finetuned/camembert-finetuned-concepts"):
        
        print("🧠 Initialisation du système à DOUBLE fine-tuning")
        
        # Modèle 1: Classification questions/affirmations
        self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_questions_path)
        self.questions_model = CamembertForSequenceClassification.from_pretrained(model_questions_path)
        
        # Modèle 2: Classification conceptuelle
        try:
            self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
            self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
            self.concepts_disponibles = True
            print("✅ Double fine-tuning chargé avec succès !")
        except:
            print("⚠️  Modèle concepts non trouvé, utilisation de la méthode classique")
            self.concepts_disponibles = False
        
        # Mapping des concepts (doit correspondre à l'ordre du dataset)
        self.concepts_labels = {
            0: "salutation_presentation",
            1: "questionnement_interrogation", 
            2: "raisonnement_logique",
            3: "affirmation_factuelle",
            4: "demande_action",
            5: "expression_emotion",
            6: "analyse_critique"
        }
    
    def classifier_question_affirmation(self, phrase):
        """Classification questions/affirmations (inchangée)"""
        inputs = self.questions_tokenizer(phrase, return_tensors="pt", padding=True, 
                                         truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.questions_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        type_phrase = "question" if predicted_class == 1 else "affirmation"
        return {
            'type': type_phrase,
            'confiance': confidence,
            'score_question': probabilities[0][1].item(),
            'score_affirmation': probabilities[0][0].item()
        }
    
    def classifier_concept_finetune(self, phrase):
        """
        Classification conceptuelle avec modèle fine-tuné
        """
        if not self.concepts_disponibles:
            return {"concept": "non_classifie", "confiance": 0.0}
        
        inputs = self.concepts_tokenizer(phrase, return_tensors="pt", padding=True,
                                        truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.concepts_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        concept_name = self.concepts_labels.get(predicted_class, "inconnu")
        
        return {
            'concept': concept_name,
            'confiance': confidence,
            'classe_predite': predicted_class,
            'scores_tous_concepts': probabilities[0].tolist()
        }
    
    def analyser_phrase_double_tuning(self, phrase):
        """
        Analyse complète avec double fine-tuning
        """
        # Classification type (question/affirmation) 
        type_result = self.classifier_question_affirmation(phrase)
        
        # Classification conceptuelle
        concept_result = self.classifier_concept_finetune(phrase)
        
        return {
            'phrase': phrase,
            'type': type_result,
            'concept': concept_result,
            'tokens': len(self.questions_tokenizer.tokenize(phrase))
        }
    
    def mode_chat_double_tuning(self):
        """
        Chat avec double classification fine-tunée
        """
        print("\n🧠 Mode chat DOUBLE FINE-TUNING !")
        print("🎯 Classification type: Modèle spécialisé questions/affirmations") 
        print("🏷️  Classification concept: Modèle spécialisé conceptuel")
        print("Tapez 'quitter' pour sortir\n")
        
        while True:
            user_input = input("Vous: ").strip()
            
            if user_input.lower() in ['quitter', 'exit']:
                print("Chat terminé !")
                break
            
            # Analyse avec double fine-tuning
            analyse = self.analyser_phrase_double_tuning(user_input)
            
            type_info = analyse['type']
            concept_info = analyse['concept']
            
            print(f"🎯 Type: {type_info['type']} (confiance: {type_info['confiance']:.3f})")
            
            if concept_info['confiance'] > 0.3:  # Seuil de confiance
                print(f"🏷️  Concept: {concept_info['concept']} (confiance: {concept_info['confiance']:.3f})")
            else:
                print("🏷️  Concept: non déterminé (confiance trop faible)")

# Usage complet
def main():
    print("🤖 IA Double Fine-Tuning : Type + Concepts")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Mode (test/chat): ").strip().lower()
    
    modele = BaseModeleFrancaisDoubleTuning()
    
    if mode == "test":
        phrases_test = [
            "Qui es-tu ?",
            "Bonjour, je suis Tibo",
            "Comment résoudre ce problème ?", 
            "Je pense que cette solution fonctionne",
            "Peux-tu m'aider ?",
            "Je suis vraiment content du résultat"
        ]
        
        print("\n🧪 Test double fine-tuning:")
        for phrase in phrases_test:
            analyse = modele.analyser_phrase_double_tuning(phrase)
            print(f"\n'{phrase}'")
            print(f"  Type: {analyse['type']['type']} ({analyse['type']['confiance']:.3f})")
            print(f"  Concept: {analyse['concept']['concept']} ({analyse['concept']['confiance']:.3f})")
    
    elif mode == "chat":
        modele.mode_chat_double_tuning()

if __name__ == "__main__":
    main()
