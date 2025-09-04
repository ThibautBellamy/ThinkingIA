# cerveau_bi_hemisphere_llama_windows.py
import ollama
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import time

class CerveauBiHemispheriqueLlama:
    """
    Architecture bi-hémisphérique utilisant Llama 3.1 8B sur Windows
    """
    def __init__(self, 
                 model_questions_path="./finetuned/camembert-finetuned-questions",
                 model_concepts_path="./finetuned/camembert-finetuned-concepts"):
        
        print("🧠 Initialisation du cerveau bi-hémisphérique Windows + Llama 3.1...")
        
        # Vérifier qu'Ollama fonctionne
        try:
            ollama.list()
            print("✅ Connexion Ollama réussie")
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            print("💡 Assurez-vous qu'Ollama est lancé : ollama serve")
            return
        
        # VOS MODÈLES EXISTANTS (hémisphères d'analyse)
        print("📚 Chargement de vos modèles fine-tunés...")
        try:
            self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_questions_path)
            self.questions_model = CamembertForSequenceClassification.from_pretrained(model_questions_path)
            print("   ✅ Modèle questions/affirmations chargé")
        except Exception as e:
            print(f"   ❌ Erreur modèle questions: {e}")
            return
        
        try:
            self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
            self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
            print("   ✅ Modèle concepts chargé")
        except Exception as e:
            print(f"   ❌ Erreur modèle concepts: {e}")
            return
        
        # Mapping des concepts
        self.concepts_labels = {
            0: "salutation_presentation", 1: "questionnement_interrogation", 
            2: "raisonnement_logique", 3: "affirmation_factuelle",
            4: "demande_action", 5: "expression_emotion", 6: "analyse_critique"
        }
        
        print("\n✅ Cerveau bi-hémisphérique initialisé avec succès !")
        print("   🎯 Hémisphère Logique : Analyse + Llama 3.1 (mode vulcain)")
        print("   🎨 Hémisphère Créatif : Analyse + Llama 3.1 (mode créatif)")
        print("   💬 Débat interne : Confrontation des 2 approches")
    
    def analyser_avec_vos_modeles(self, enigme):
        """Utilise VOS modèles fine-tunés pour l'analyse préliminaire"""
        # Classification type (question/affirmation)
        inputs_q = self.questions_tokenizer(enigme, return_tensors="pt", 
                                          padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs_q = self.questions_model(**inputs_q)
            probs_q = torch.nn.functional.softmax(outputs_q.logits, dim=-1)
            type_predicted = torch.argmax(probs_q, dim=1).item()
            type_confidence = torch.max(probs_q).item()
        
        type_phrase = "question" if type_predicted == 1 else "affirmation"
        
        # Classification conceptuelle
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
            'concept_confidence': concept_confidence
        }
    
    def interroger_llama(self, prompt, temperature=0.7):
        """Interroge Llama 3.1 8B via Ollama sur Windows"""
        try:
            response = ollama.generate(
                model='llama3.1',
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 500,
                    'batch_size': 8,        # ✅ AJOUTÉ : Batch plus gros
                    'num_thread': 4,        # ✅ AJOUTÉ : Plus de threads
                    'top_k': 50,
                    'top_p': 0.9,
                }
            )
            return response['response']
        except Exception as e:
            return f"Erreur Llama: {str(e)}"
    
    def hemisphere_logique(self, enigme, analyse):
        """🧠 HÉMISPHÈRE LOGIQUE : Mode Vulcain"""
        prompt_logique = f"""Tu es l'hémisphère logique d'un cerveau artificiel. Tu raisonnes comme un Vulcain : logique pure, précision absolue, pas d'émotion.

ANALYSE PRÉLIMINAIRE:
- Type: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})
- Concept: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})

ÉNIGME À RÉSOUDRE: {enigme}

CONSIGNES LOGIQUES:
1. Identifie les prémisses avec précision
2. Applique les règles de logique formelle
3. Déduis la conclusion par étapes rigoureuses
4. Critique ton propre raisonnement pour détecter les failles
5. Formule une réponse logique irréfutable

RAISONNEMENT LOGIQUE VULCAIN:"""

        return self.interroger_llama(prompt_logique, temperature=0.3)
    
    def hemisphere_creatif(self, enigme, analyse):
        """🎨 HÉMISPHÈRE CRÉATIF : Mode intuitif"""
        prompt_creatif = f"""Tu es l'hémisphère créatif d'un cerveau artificiel. Tu explores avec intuition, imagination et ouverture d'esprit.

ANALYSE PRÉLIMINAIRE:
- Type: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})
- Concept: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})

ÉNIGME À EXPLORER: {enigme}

CONSIGNES CRÉATIVES:
1. Explore des angles inattendus et originaux
2. Utilise ton intuition pour voir au-delà de l'évidence
3. Propose des interprétations multiples
4. Connecte des idées apparemment non-liées
5. Sois ouvert aux possibilités paradoxales

APPROCHE CRÉATIVE INTUITIVE:"""

        return self.interroger_llama(prompt_creatif, temperature=0.9)
    
    def debat_interne(self, solution_logique, solution_creative, enigme):
        """💬 DÉBAT ENTRE LES DEUX HÉMISPHÈRES"""
        prompt_debat = f"""Tu es un modérateur impartial qui organise un débat entre deux hémisphères cérébraux.

ÉNIGME ORIGINALE: {enigme}

SOLUTION HÉMISPHÈRE LOGIQUE:
{solution_logique}

SOLUTION HÉMISPHÈRE CRÉATIF:
{solution_creative}

TON RÔLE DE MODÉRATEUR:
1. Compare les deux approches objectivement
2. Identifie les points forts et faibles de chaque solution
3. Détermine laquelle est la plus pertinente ou si une synthèse est possible
4. Explique ton raisonnement pour cette conclusion
5. Formule la réponse finale optimale

SYNTHÈSE ET DÉCISION FINALE:"""

        return self.interroger_llama(prompt_debat, temperature=0.5)
    
    def resoudre_enigme_complete(self, enigme):
        """🧩 PROCESSUS COMPLET Windows"""
        print(f"🧩 Résolution complète de: '{enigme}'")
        print("=" * 80)
        
        # ÉTAPE 1: Analyse avec vos modèles fine-tunés
        print("📊 Phase 1: Analyse préliminaire avec vos modèles...")
        analyse = self.analyser_avec_vos_modeles(enigme)
        print(f"   🎯 Type détecté: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})")
        print(f"   🏷️  Concept détecté: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})")
        
        # ÉTAPE 2: Hémisphère logique
        print("\n🧠 Phase 2: Raisonnement hémisphère logique avec Llama...")
        solution_logique = self.hemisphere_logique(enigme, analyse)
        print("   ✅ Raisonnement logique généré")
        
        # ÉTAPE 3: Hémisphère créatif  
        print("🎨 Phase 3: Exploration hémisphère créatif avec Llama...")
        solution_creative = self.hemisphere_creatif(enigme, analyse)
        print("   ✅ Approche créative générée")
        
        # ÉTAPE 4: Débat et synthèse
        print("💬 Phase 4: Débat interne et synthèse finale...")
        synthese_finale = self.debat_interne(solution_logique, solution_creative, enigme)
        print("   ✅ Synthèse finale générée")
        
        # RÉSULTATS
        print("\n" + "=" * 80)
        print("📋 RÉSULTATS DU CERVEAU BI-HÉMISPHÉRIQUE")
        print("=" * 80)
        
        print(f"\n🧠 HÉMISPHÈRE LOGIQUE:")
        print("-" * 40)
        print(solution_logique)
        
        print(f"\n🎨 HÉMISPHÈRE CRÉATIF:")
        print("-" * 40)
        print(solution_creative)
        
        print(f"\n💬 SYNTHÈSE FINALE:")
        print("-" * 40)
        print(synthese_finale)
        
        return {
            'enigme': enigme,
            'analyse_preliminaire': analyse,
            'solution_logique': solution_logique,
            'solution_creative': solution_creative,
            'synthese_finale': synthese_finale
        }
    
    def mode_interactif_windows(self):
        """🎮 Mode interactif Windows"""
        print("\n🎮 MODE INTERACTIF WINDOWS - CERVEAU BI-HÉMISPHÉRIQUE")
        print("Soumettez vos énigmes au cerveau complet !")
        print("Tapez 'quitter' pour arrêter, 'test' pour une énigme de démonstration\n")
        
        while True:
            enigme = input("🧩 Votre énigme: ").strip()
            
            if enigme.lower() in ['quitter', 'exit', 'stop']:
                print("🛑 Cerveau bi-hémisphérique en veille. À bientôt !")
                break
            
            if enigme.lower() == 'test':
                enigme = "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?"
                print(f"🧪 Énigme de test: {enigme}")
            
            if not enigme:
                print("⚠️ Veuillez saisir une énigme...")
                continue
            
            # Résolution complète
            start_time = time.time()
            self.resoudre_enigme_complete(enigme)
            end_time = time.time()
            
            print(f"\n⏱️  Temps de traitement: {end_time - start_time:.2f} secondes")
            print("\n" + "🔄" * 40 + "\n")

# Test du cerveau bi-hémisphérique Windows
if __name__ == "__main__":
    print("🪟 CERVEAU BI-HÉMISPHÉRIQUE WINDOWS + LLAMA 3.1")
    print("=" * 50)
    
    try:
        cerveau = CerveauBiHemispheriqueLlama()
        cerveau.mode_interactif_windows()
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        print("\n💡 Vérifications à faire:")
        print("   1. Ollama est installé : ollama --version")
        print("   2. Llama 3.1 est téléchargé : ollama list")
        print("   3. Vos modèles fine-tunés sont disponibles")
