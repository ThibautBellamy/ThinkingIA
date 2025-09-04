# cerveau_bi_hemisphere_parallele.py
import ollama
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import concurrent.futures
import threading
import time

class CerveauBiHemispheriqueParallele:
    """
    Architecture bi-hémisphérique COMPLÈTE avec traitement parallèle optimisé
    Utilise VOS modèles fine-tunés + Llama 3.1 8B en parallèle
    """
    def __init__(self, 
                 model_questions_path="./camembert-finetuned-questions",
                 model_concepts_path="./camembert-finetuned-concepts"):
        
        print("🧠 Initialisation du cerveau bi-hémisphérique PARALLÈLE...")
        
        # Lock pour éviter les conflits GPU avec les modèles CamemBERT
        self.gpu_lock = threading.Lock()
        
        # Vérifier qu'Ollama fonctionne
        try:
            ollama.list()
            print("✅ Connexion Ollama réussie")
        except Exception as e:
            print(f"❌ Erreur Ollama: {e}")
            print("💡 Assurez-vous qu'Ollama est lancé")
            return
        
        # VOS MODÈLES FINE-TUNÉS (chargement sécurisé)
        print("📚 Chargement de vos modèles fine-tunés...")
        try:
            with self.gpu_lock:
                self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_questions_path)
                self.questions_model = CamembertForSequenceClassification.from_pretrained(model_questions_path)
                self.questions_model.eval()  # Mode évaluation pour performance
                print("   ✅ Modèle questions/affirmations chargé")
                
                self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
                self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
                self.concepts_model.eval()  # Mode évaluation pour performance
                print("   ✅ Modèle concepts chargé")
        except Exception as e:
            print(f"   ❌ Erreur chargement modèles: {e}")
            return
        
        # Mapping des concepts
        self.concepts_labels = {
            0: "salutation_presentation", 1: "questionnement_interrogation", 
            2: "raisonnement_logique", 3: "affirmation_factuelle",
            4: "demande_action", 5: "expression_emotion", 6: "analyse_critique"
        }
        
        print("\n✅ Cerveau bi-hémisphérique PARALLÈLE initialisé !")
        print("   🎯 Hémisphère Logique : Vos modèles + Llama 3.1 (mode vulcain)")
        print("   🎨 Hémisphère Créatif : Vos modèles + Llama 3.1 (mode créatif)")
        print("   ⚡ Traitement : PARALLÈLE optimisé GPU + CPU")
        print("   💬 Débat interne : Confrontation des 2 approches")
    
    def analyser_avec_vos_modeles_thread_safe(self, enigme):
        """
        Analyse thread-safe avec VOS modèles fine-tunés
        Utilise un lock pour éviter les conflits GPU
        """
        with self.gpu_lock:  # Protection contre accès concurrent GPU
            # Classification type (question/affirmation)
            inputs_q = self.questions_tokenizer(
                enigme, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            with torch.no_grad():
                outputs_q = self.questions_model(**inputs_q)
                probs_q = torch.nn.functional.softmax(outputs_q.logits, dim=-1)
                type_predicted = torch.argmax(probs_q, dim=1).item()
                type_confidence = torch.max(probs_q).item()
            
            type_phrase = "question" if type_predicted == 1 else "affirmation"
            
            # Classification conceptuelle
            inputs_c = self.concepts_tokenizer(
                enigme, 
                return_tensors="pt",
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
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
    
    def interroger_llama_optimise(self, prompt, temperature=0.7, thread_id=""):
        """
        Version optimisée pour interroger Llama avec identification thread
        """
        try:
            response = ollama.generate(
                model='llama3.1',
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 500,
                    'batch_size': 4,      # Optimisation batch
                    'num_thread': 2,      # Plus de threads CPU
                    'top_k': 50,
                    'top_p': 0.9,
                }
            )
            print(f"   ✅ Thread {thread_id} terminé")
            return response['response']
        except Exception as e:
            print(f"   ❌ Erreur Thread {thread_id}: {str(e)}")
            return f"Erreur Llama Thread {thread_id}: {str(e)}"
    
    def hemisphere_logique_thread(self, enigme, analyse):
        """
        🧠 HÉMISPHÈRE LOGIQUE - Version thread optimisée
        """
        thread_id = "LOGIQUE"
        print(f"   🧠 Démarrage hémisphère {thread_id}...")
        
        prompt_logique = f"""Tu es l'hémisphère logique d'un cerveau artificiel. Tu raisonnes comme un Vulcain : logique pure, précision absolue.

ANALYSE PRÉLIMINAIRE par modèles spécialisés:
- Type détecté: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})
- Concept détecté: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})

ÉNIGME À RÉSOUDRE LOGIQUEMENT: {enigme}

CONSIGNES LOGIQUES STRICTES:
1. Identifie les prémisses avec précision mathématique
2. Applique les règles de logique formelle (syllogisme, modus ponens, etc.)
3. Déduis la conclusion par étapes rigoureuses et vérifiables
4. Critique ton propre raisonnement pour détecter les failles logiques
5. Formule une réponse irréfutable basée sur la logique pure

RAISONNEMENT LOGIQUE VULCAIN:"""

        result = self.interroger_llama_optimise(prompt_logique, temperature=0.2, thread_id=thread_id)
        return result
    
    def hemisphere_creatif_thread(self, enigme, analyse):
        """
        🎨 HÉMISPHÈRE CRÉATIF - Version thread optimisée
        """
        thread_id = "CRÉATIF"
        print(f"   🎨 Démarrage hémisphère {thread_id}...")
        
        prompt_creatif = f"""Tu es l'hémisphère créatif d'un cerveau artificiel. Tu explores avec intuition, imagination et ouverture d'esprit.

ANALYSE PRÉLIMINAIRE par modèles spécialisés:
- Type détecté: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})
- Concept détecté: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})

ÉNIGME À EXPLORER CRÉATIVEMENT: {enigme}

CONSIGNES CRÉATIVES LIBRES:
1. Explore des angles inattendus et des perspectives originales
2. Utilise ton intuition pour voir au-delà de l'évidence logique
3. Propose des interprétations multiples et des solutions alternatives
4. Connecte des idées apparemment non-liées de manière innovante
5. Sois ouvert aux possibilités paradoxales et aux solutions non-conventionnelles
6. Offre une perspective humaine et émotionnelle à l'énigme

APPROCHE CRÉATIVE INTUITIVE:"""

        result = self.interroger_llama_optimise(prompt_creatif, temperature=0.9, thread_id=thread_id)
        return result
    
    def debat_interne_optimise(self, solution_logique, solution_creative, enigme, analyse):
        """
        💬 DÉBAT OPTIMISÉ entre les deux hémisphères
        """
        print("   💬 Démarrage du débat interne...")
        
        prompt_debat = f"""Tu es un modérateur expert qui organise un débat constructif entre deux hémisphères cérébraux.

ÉNIGME ORIGINALE: {enigme}
ANALYSE PRÉLIMINAIRE: Type={analyse['type']}, Concept={analyse['concept']}

SOLUTION HÉMISPHÈRE LOGIQUE (Vulcain):
{solution_logique}

SOLUTION HÉMISPHÈRE CRÉATIF (Intuitif):
{solution_creative}

TON RÔLE DE MODÉRATEUR EXPERT:
1. Compare objectivement les deux approches en analysant leurs mérites respectifs
2. Identifie les points forts et les limites de chaque solution
3. Évalue la pertinence de chaque approche selon le type d'énigme
4. Détermine si une approche est supérieure ou si une synthèse est optimale
5. Explique ton raisonnement de manière claire et structurée
6. Formule la réponse finale la plus complète et précise possible

SYNTHÈSE FINALE DU DÉBAT ET DÉCISION:"""

        result = self.interroger_llama_optimise(prompt_debat, temperature=0.5, thread_id="DÉBAT")
        return result
    
    def resoudre_enigme_parallele_complete(self, enigme):
        """
        🧩 PROCESSUS COMPLET PARALLÉLISÉ avec vos modèles fine-tunés
        """
        start_time = time.time()
        print(f"🧩 Résolution PARALLÈLE de: '{enigme}'")
        print("=" * 80)
        
        # ÉTAPE 1: Analyse préliminaire avec VOS modèles fine-tunés
        print("📊 Phase 1: Analyse préliminaire avec vos modèles CamemBERT...")
        analyse_start = time.time()
        
        analyse = self.analyser_avec_vos_modeles_thread_safe(enigme)
        
        analyse_time = time.time() - analyse_start
        print(f"   🎯 Type détecté: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})")
        print(f"   🏷️  Concept détecté: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})")
        print(f"   ⏱️  Temps analyse: {analyse_time:.2f}s")
        
        # ÉTAPE 2: TRAITEMENT PARALLÈLE des deux hémisphères
        print("\n🚀 Phase 2: Lancement PARALLÈLE des hémisphères avec Llama...")
        parallel_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Soumettre les 2 tâches en parallèle
            future_logique = executor.submit(self.hemisphere_logique_thread, enigme, analyse)
            future_creatif = executor.submit(self.hemisphere_creatif_thread, enigme, analyse)
            
            # Attendre les résultats
            solution_logique = future_logique.result()
            solution_creative = future_creatif.result()
        
        parallel_time = time.time() - parallel_start
        print(f"   ✅ Les 2 hémisphères terminés en parallèle !")
        print(f"   ⏱️  Temps parallèle: {parallel_time:.2f}s")
        
        # ÉTAPE 3: Débat et synthèse finale
        print("\n💬 Phase 3: Débat interne et synthèse finale...")
        debat_start = time.time()
        
        synthese_finale = self.debat_interne_optimise(solution_logique, solution_creative, enigme, analyse)
        
        debat_time = time.time() - debat_start
        print(f"   ✅ Synthèse finale générée")
        print(f"   ⏱️  Temps débat: {debat_time:.2f}s")
        
        # AFFICHAGE DES RÉSULTATS COMPLETS
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("📋 RÉSULTATS DU CERVEAU BI-HÉMISPHÉRIQUE PARALLÈLE")
        print("=" * 80)
        
        print(f"\n📊 ANALYSE PRÉLIMINAIRE (Vos modèles fine-tunés):")
        print("-" * 50)
        print(f"Type: {analyse['type']} (confiance: {analyse['type_confidence']:.3f})")
        print(f"Concept: {analyse['concept']} (confiance: {analyse['concept_confidence']:.3f})")
        
        print(f"\n🧠 HÉMISPHÈRE LOGIQUE (Llama 3.1 mode Vulcain):")
        print("-" * 50)
        print(solution_logique)
        
        print(f"\n🎨 HÉMISPHÈRE CRÉATIF (Llama 3.1 mode intuitif):")
        print("-" * 50)
        print(solution_creative)
        
        print(f"\n💬 SYNTHÈSE FINALE (Débat modéré):")
        print("-" * 50)
        print(synthese_finale)
        
        print(f"\n⏱️  PERFORMANCES:")
        print(f"   Analyse: {analyse_time:.2f}s | Parallèle: {parallel_time:.2f}s | Débat: {debat_time:.2f}s")
        print(f"   TOTAL: {total_time:.2f}s")
        
        return {
            'enigme': enigme,
            'analyse_preliminaire': analyse,
            'solution_logique': solution_logique,
            'solution_creative': solution_creative,
            'synthese_finale': synthese_finale,
            'temps_total': total_time,
            'temps_analyse': analyse_time,
            'temps_parallele': parallel_time,
            'temps_debat': debat_time
        }
    
    def mode_test_performance(self):
        """
        🧪 Mode de test des performances parallèles
        """
        enigmes_test = [
            "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?",
            "Alice est plus âgée que Bob. Bob est plus âgé que Charlie. Qui est le plus âgé ?",
            "Comment résoudre l'équation : 2x + 5 = 13 ?",
            "Un barbier rase tous les hommes qui ne se rasent pas eux-mêmes. Qui rase le barbier ?"
        ]
        
        print("🧪 TEST DE PERFORMANCE - Architecture bi-hémisphérique parallèle")
        print("=" * 80)
        
        temps_total = 0
        for i, enigme in enumerate(enigmes_test, 1):
            print(f"\n🔢 Test {i}/{len(enigmes_test)}")
            print("-" * 60)
            
            resultat = self.resoudre_enigme_parallele_complete(enigme)
            temps_total += resultat['temps_total']
            
            print("\n" + "🔄" * 40 + "\n")
        
        print(f"📊 BILAN PERFORMANCE GLOBALE:")
        print(f"   Temps total: {temps_total:.2f}s")
        print(f"   Temps moyen par énigme: {temps_total/len(enigmes_test):.2f}s")
        print(f"   Énigmes traitées: {len(enigmes_test)}")
    
    def mode_interactif_optimise(self):
        """
        🎮 Mode interactif avec architecture parallèle optimisée
        """
        print("\n🎮 MODE INTERACTIF - CERVEAU BI-HÉMISPHÉRIQUE PARALLÈLE")
        print("Architecture complète : Vos modèles fine-tunés + Llama 3.1 en parallèle")
        print("Commandes : 'test' = énigme démo | 'perf' = test performance | 'quitter' = sortir\n")
        
        while True:
            enigme = input("🧩 Votre énigme: ").strip()
            
            if enigme.lower() in ['quitter', 'exit', 'stop']:
                print("🛑 Cerveau bi-hémisphérique en veille. À bientôt !")
                break
            
            if enigme.lower() == 'test':
                enigme = "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?"
                print(f"🧪 Énigme de test: {enigme}")
            
            if enigme.lower() == 'perf':
                self.mode_test_performance()
                continue
            
            if not enigme:
                print("⚠️ Veuillez saisir une énigme...")
                continue
            
            # Résolution complète parallèle
            self.resoudre_enigme_parallele_complete(enigme)
            print("\n" + "🔄" * 40 + "\n")

# LANCEMENT DU CERVEAU BI-HÉMISPHÉRIQUE PARALLÈLE
if __name__ == "__main__":
    print("🪟 CERVEAU BI-HÉMISPHÉRIQUE PARALLÈLE - Windows + Llama 3.1")
    print("Vos modèles CamemBERT fine-tunés + Traitement parallèle optimisé")
    print("=" * 70)
    
    try:
        # Variables d'environnement pour optimiser Ollama (optionnel)
        import os
        os.environ['OLLAMA_NUM_PARALLEL'] = '4'
        os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
        
        # Initialisation du cerveau
        cerveau = CerveauBiHemispheriqueParallele()
        
        # Mode interactif optimisé
        cerveau.mode_interactif_optimise()
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        print("\n💡 Vérifications à faire:")
        print("   1. Ollama fonctionne : ollama --version")
        print("   2. Llama 3.1 disponible : ollama list")
        print("   3. Vos modèles fine-tunés sont accessibles")
        print("   4. GPU NVIDIA drivers à jour")
