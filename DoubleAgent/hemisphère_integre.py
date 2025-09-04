# hemisphère_integre.py
from transformers import (
    CamembertForSequenceClassification, 
    CamembertTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch

class HemisphereIntegre:
    """
    Architecture complète utilisant les 3 modèles fine-tunés
    """
    def __init__(self, 
                 model_questions_path="./finetuned/camembert-finetuned-questions",
                 model_concepts_path="./finetuned/camembert-finetuned-concepts", 
                 model_reasoning_path="./finetuned/camembert-finetuned-raisonnement"):
        
        print("🧠 Initialisation de l'architecture bi-hémisphérique complète...")
        
        # HÉMISPHÈRE 1: Classification questions/affirmations
        self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_questions_path)
        self.questions_model = CamembertForSequenceClassification.from_pretrained(model_questions_path)
        
        # HÉMISPHÈRE 2: Classification conceptuelle  
        self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
        self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
        
        # SYNTHÈSE: Génération de raisonnement
        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(model_reasoning_path)
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(model_reasoning_path)
        
        if self.reasoning_tokenizer.pad_token is None:
            self.reasoning_tokenizer.pad_token = self.reasoning_tokenizer.eos_token
        
        # Mapping des concepts
        self.concepts_labels = {
            0: "salutation_presentation", 1: "questionnement_interrogation", 
            2: "raisonnement_logique", 3: "affirmation_factuelle",
            4: "demande_action", 5: "expression_emotion", 6: "analyse_critique"
        }
        
        print("✅ Architecture complète chargée !")
        print("   🎯 Hémisphère 1: Classification type (questions/affirmations)")
        print("   🏷️  Hémisphère 2: Classification conceptuelle") 
        print("   🧠 Synthèse: Génération de raisonnement contextualisé")
    
    def analyser_avec_hemispheres(self, enigme):
        """
        Analyse complète utilisant les 2 hémisphères
        """
        # HÉMISPHÈRE 1: Type de phrase
        inputs_q = self.questions_tokenizer(enigme, return_tensors="pt", 
                                          padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs_q = self.questions_model(**inputs_q)
            probs_q = torch.nn.functional.softmax(outputs_q.logits, dim=-1)
            type_predicted = torch.argmax(probs_q, dim=1).item()
            type_confidence = torch.max(probs_q).item()
        
        type_phrase = "question" if type_predicted == 1 else "affirmation"
        
        # HÉMISPHÈRE 2: Concept
        inputs_c = self.concepts_tokenizer(enigme, return_tensors="pt",
                                         padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs_c = self.concepts_model(**inputs_c)
            probs_c = torch.nn.functional.softmax(outputs_c.logits, dim=-1)
            concept_predicted = torch.argmax(probs_c, dim=1).item()
            concept_confidence = torch.max(probs_c).item()
        
        concept_name = self.concepts_labels.get(concept_predicted, "inconnu")
        
        return {
            'type': type_phrase,
            'type_confidence': type_confidence,
            'concept': concept_name,
            'concept_confidence': concept_confidence,
            'enigme': enigme
        }
    
    def generer_raisonnement_contextualise(self, enigme):
        """
        Génération de raisonnement guidée par l'analyse des hémisphères
        """
        print(f"🔍 Analyse bi-hémisphérique de: '{enigme}'")
        
        # Étape 1: Analyse avec les 2 hémisphères
        analyse = self.analyser_avec_hemispheres(enigme)
        print(f"   🎯 Type: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})")
        print(f"   🏷️  Concept: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})")
        
        # Étape 2: Contexte pour le raisonnement basé sur l'analyse
        if analyse['concept'] == 'raisonnement_logique':
            contexte = "Cette énigme nécessite un raisonnement logique rigoureux."
        elif analyse['concept'] == 'questionnement_interrogation':
            contexte = "Cette question demande une analyse méthodique."
        elif analyse['concept'] == 'analyse_critique':
            contexte = "Cette situation nécessite une évaluation critique."
        else:
            contexte = f"Cette {analyse['type']} concerne {analyse['concept']}."
        
        # Étape 3: Prompt enrichi avec contexte des hémisphères
        prompt_enrichi = f"""CONTEXTE: {contexte}
TYPE: {analyse['type']} 
CONCEPT: {analyse['concept']}
ÉNIGME: {enigme}
RAISONNEMENT:"""
        
        print(f"\n🧠 Génération contextuelle du raisonnement...")
        
        # Génération avec contexte
        inputs = self.reasoning_tokenizer(prompt_enrichi, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=300,
                min_length=len(inputs['input_ids'][0]) + 30,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.reasoning_tokenizer.pad_token_id,
            )
        
        generated_text = self.reasoning_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raisonnement = generated_text[len(prompt_enrichi):].strip()
        
        return {
            'analyse_hemispheres': analyse,
            'contexte': contexte,
            'raisonnement': raisonnement,
            'prompt_utilise': prompt_enrichi
        }
    
    def mode_test_integre(self):
        """
        Test de l'architecture complète
        """
        enigmes_test = [
            "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?",
            "Bonjour, je suis Pierre et j'aimerais votre aide.",
            "Cette méthode présente des failles logiques importantes.",
            "Comment résoudre cette équation : 2x + 5 = 13 ?"
        ]
        
        print("🧪 Test de l'architecture bi-hémisphérique intégrée")
        print("=" * 80)
        
        for i, enigme in enumerate(enigmes_test, 1):
            print(f"\n🔢 Test {i}/{len(enigmes_test)}")
            print("-" * 60)
            
            resultat = self.generer_raisonnement_contextualise(enigme)
            
            print(f"\n📋 Raisonnement généré:")
            print(f"{resultat['raisonnement']}")
            print("-" * 60)

# Test de l'architecture intégrée
if __name__ == "__main__":
    try:
        hemisphere_integre = HemisphereIntegre()
        hemisphere_integre.mode_test_integre()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Assurez-vous que tous les modèles sont disponibles")
