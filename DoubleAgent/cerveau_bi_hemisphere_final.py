# cerveau_bi_hemisphere_final.py
import ollama
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import concurrent.futures
import threading
import time

class CerveauBiHemispheriqueFinal:
    """
    Version FINALE : Parallélisée + Prompts courts + Sans béquilles + Vos modèles
    """
    def __init__(self, 
                 model_questions_path="./finetuned/camembert-finetuned-questions",
                 model_concepts_path="./finetuned/camembert-finetuned-concepts"):
        
        print("🧠 Cerveau bi-hémisphérique FINAL (parallélisé, sans béquilles)...")
        
        # Lock pour protéger l'accès GPU
        self.gpu_lock = threading.Lock()
        
        # Vérifier Ollama
        try:
            ollama.list()
            print("✅ Ollama connecté")
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            return
        
        # Charger VOS modèles fine-tunés
        try:
            with self.gpu_lock:
                self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_questions_path)
                self.questions_model = CamembertForSequenceClassification.from_pretrained(model_questions_path)
                self.questions_model.eval()
                
                self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
                self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
                self.concepts_model.eval()
                
                print("✅ Vos modèles fine-tunés chargés")
        except Exception as e:
            print(f"❌ Erreur modèles: {e}")
            return
        
        self.concepts_labels = {
            0: "salutation_presentation", 1: "questionnement_interrogation", 
            2: "raisonnement_logique", 3: "affirmation_factuelle",
            4: "demande_action", 5: "expression_emotion", 6: "analyse_critique"
        }
        
        print("✅ Architecture bi-hémisphérique PARALLÉLISÉE prête !")
        print("   🎯 Hémisphère Logique : Vos modèles + Llama (mode précis)")
        print("   🎨 Hémisphère Créatif : Vos modèles + Llama (mode intuitif)")
        print("   ⚡ Traitement : PARALLÈLE thread-safe")
        print("   🚫 Sans béquilles de code")
    
    def analyser_thread_safe(self, enigme):
        """Analyse avec VOS modèles fine-tunés (thread-safe)"""
        with self.gpu_lock:
            # Classification type
            inputs_q = self.questions_tokenizer(enigme, return_tensors="pt", 
                                              padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs_q = self.questions_model(**inputs_q)
                probs_q = torch.nn.functional.softmax(outputs_q.logits, dim=-1)
                type_predicted = torch.argmax(probs_q, dim=1).item()
                type_confidence = torch.max(probs_q).item()
            
            type_phrase = "question" if type_predicted == 1 else "affirmation"
            
            # Classification concept
            inputs_c = self.concepts_tokenizer(enigme, return_tensors="pt",
                                             padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs_c = self.concepts_model(**inputs_c)
                probs_c = torch.nn.functional.softmax(outputs_c.logits, dim=-1)
                concept_predicted = torch.argmax(probs_c, dim=1).item()
                concept_confidence = torch.max(probs_c).item()
            
            concept_name = self.concepts_labels.get(concept_predicted, "inconnu")
            
            return {
                'type': type_phrase, 'type_confidence': type_confidence,
                'concept': concept_name, 'concept_confidence': concept_confidence
            }
    
    def interroger_llama_court(self, prompt, temperature=0.7, thread_id=""):
        """Interrogation Llama avec limites strictes"""
        try:
            response = ollama.generate(
                model='llama3.1',
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 100,  # LIMITE STRICTE
                    'top_k': 40,
                    'top_p': 0.8,
                    'stop': ['\n\n', 'Conclusion:', 'En résumé:', 'Donc:'],  # Arrêts
                }
            )
            print(f"   ✅ Thread {thread_id} terminé")
            return response['response']
        except Exception as e:
            print(f"   ❌ Erreur Thread {thread_id}: {str(e)}")
            return f"Erreur {thread_id}: {str(e)}"
    
    def hemisphere_logique_thread(self, enigme, analyse):
        """🧠 Hémisphère logique (thread parallèle)"""
        thread_id = "LOGIQUE"
        print(f"   🧠 Démarrage {thread_id}...")
        
        prompt = f"""Tu es l'hémisphère logique, précis.

ÉNIGME: {enigme}
TYPE: {analyse['type']} | CONCEPT: {analyse['concept']}

CONSIGNE: Résous en 3 lignes maximum, étape par étape, conclusion directe.

SOLUTION LOGIQUE:"""
        
        return self.interroger_llama_court(prompt, temperature=0.2, thread_id=thread_id)
    
    def hemisphere_creatif_thread(self, enigme, analyse):
        """🎨 Hémisphère créatif (thread parallèle)"""
        thread_id = "CRÉATIF"
        print(f"   🎨 Démarrage {thread_id}...")
        
        prompt = f"""Tu es l'hémisphère créatif, intuitif et original.

ÉNIGME: {enigme}
TYPE: {analyse['type']} | CONCEPT: {analyse['concept']}

CONSIGNE: Approche différente en 3 lignes max, solution créative.

APPROCHE CRÉATIVE:"""
        
        return self.interroger_llama_court(prompt, temperature=0.9, thread_id=thread_id)
    
    def debat_rapide(self, solution_logique, solution_creative):
        """💬 Débat synthétique rapide"""
        print("   💬 Débat en cours...")
        
        prompt = f"""Compare ces 2 solutions, choisis la meilleure :

LOGIQUE: {solution_logique}
CRÉATIF: {solution_creative}

Réponds en 2 lignes : quelle solution est correcte et pourquoi.

VERDICT:"""
        
        return self.interroger_llama_court(prompt, temperature=0.5, thread_id="DÉBAT")
    
    def resoudre_enigme_parallele(self, enigme):
        """🧩 Résolution PARALLÉLISÉE complète"""
        start_time = time.time()
        print(f"🧩 RÉSOLUTION PARALLÈLE: '{enigme}'")
        print("=" * 60)
        
        # ÉTAPE 1: Analyse avec VOS modèles fine-tunés
        print("📊 Analyse avec vos modèles CamemBERT...")
        analyse_start = time.time()
        analyse = self.analyser_thread_safe(enigme)
        analyse_time = time.time() - analyse_start
        
        print(f"   🎯 Type: {analyse['type']} (conf: {analyse['type_confidence']:.3f})")
        print(f"   🏷️  Concept: {analyse['concept']} (conf: {analyse['concept_confidence']:.3f})")
        print(f"   ⏱️  Temps analyse: {analyse_time:.2f}s")
        
        # ÉTAPE 2: TRAITEMENT PARALLÈLE des hémisphères
        print("\n🚀 Lancement PARALLÈLE des hémisphères...")
        parallel_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # ⚡ PARALLÉLISATION VRAIE
            future_logique = executor.submit(self.hemisphere_logique_thread, enigme, analyse)
            future_creatif = executor.submit(self.hemisphere_creatif_thread, enigme, analyse)
            
            # Attendre les résultats
            solution_logique = future_logique.result()
            solution_creative = future_creatif.result()
        
        parallel_time = time.time() - parallel_start
        print(f"   ✅ Hémisphères terminés en PARALLÈLE !")
        print(f"   ⏱️  Temps parallèle: {parallel_time:.2f}s")
        
        # ÉTAPE 3: Débat rapide
        print("\n💬 Débat et synthèse...")
        debat_start = time.time()
        verdict = self.debat_rapide(solution_logique, solution_creative)
        debat_time = time.time() - debat_start
        
        # RÉSULTATS COMPACTS
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("📋 RÉSULTATS BI-HÉMISPHÉRIQUE PARALLÈLE")
        print("=" * 60)
        
        print(f"\n📊 ANALYSE (vos modèles): {analyse['type']} | {analyse['concept']}")
        print(f"\n🧠 LOGIQUE: {solution_logique}")
        print(f"\n🎨 CRÉATIF: {solution_creative}")
        print(f"\n💬 VERDICT: {verdict}")
        
        print(f"\n⏱️  PERFORMANCES:")
        print(f"   Analyse: {analyse_time:.2f}s | Parallèle: {parallel_time:.2f}s | Débat: {debat_time:.2f}s")
        print(f"   TOTAL: {total_time:.2f}s")
        
        return {
            'enigme': enigme, 'analyse': analyse,
            'solution_logique': solution_logique, 'solution_creative': solution_creative,
            'verdict': verdict, 'temps_total': total_time
        }
    
    def mode_interactif_final(self):
        """🎮 Mode interactif final optimisé"""
        print("\n🎮 CERVEAU BI-HÉMISPHÉRIQUE FINAL")
        print("✅ Parallélisé | ✅ Prompts courts | ✅ Vos modèles | ❌ Sans béquilles")
        print("Commandes: 'quitter' pour sortir\n")
        
        while True:
            enigme = input("🧩 Votre énigme: ").strip()
            
            if enigme.lower() in ['quitter', 'exit', 'stop']:
                print("🛑 Cerveau en veille. À bientôt !")
                break
            
            if not enigme:
                print("⚠️ Veuillez saisir une énigme...")
                continue
            
            # RÉSOLUTION PARALLÉLISÉE
            self.resoudre_enigme_parallele(enigme)
            print("\n" + "🔄" * 30 + "\n")

# LANCEMENT
if __name__ == "__main__":
    print("🚀 CERVEAU BI-HÉMISPHÉRIQUE FINAL")
    print("Architecture complète : Parallélisation + Vos modèles + Sans triche")
    print("=" * 65)
    
    try:
        cerveau = CerveauBiHemispheriqueFinal()
        cerveau.mode_interactif_final()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
