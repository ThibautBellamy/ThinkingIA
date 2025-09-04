# hemisphère_logique_corrige.py
from transformers import (
    CamembertForSequenceClassification, 
    CamembertTokenizer
)
import torch
import re

class HemisphereLogique:
    """
    Hémisphère logique capable de raisonnement étape par étape
    """
    def __init__(self, 
                 model_base_path="./finetuned/camembert-finetuned-questions",
                 model_concepts_path="./finetuned/camembert-finetuned-concepts"):
        
        print("🧠 Initialisation de l'hémisphère logique...")
        
        # Modèles de compréhension (vos modèles fine-tunés)
        self.questions_tokenizer = CamembertTokenizer.from_pretrained(model_base_path)
        self.questions_model = CamembertForSequenceClassification.from_pretrained(model_base_path)
        
        self.concepts_tokenizer = CamembertTokenizer.from_pretrained(model_concepts_path)
        self.concepts_model = CamembertForSequenceClassification.from_pretrained(model_concepts_path)
        
        # Concepts mapping
        self.concepts_labels = {
            0: "salutation_presentation",
            1: "questionnement_interrogation", 
            2: "raisonnement_logique",
            3: "affirmation_factuelle",
            4: "demande_action",
            5: "expression_emotion",
            6: "analyse_critique"
        }
        
        print("✅ Hémisphère logique initialisé")
    
    def analyser_enigme(self, enigme):
        """
        Analyse une énigme avec les modèles de compréhension
        """
        # Classification type (question/affirmation)
        inputs = self.questions_tokenizer(enigme, return_tensors="pt", padding=True, 
                                         truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.questions_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            type_predicted = torch.argmax(probs, dim=1).item()
            type_confidence = torch.max(probs).item()
        
        type_phrase = "question" if type_predicted == 1 else "affirmation"
        
        # Classification conceptuelle
        inputs_concepts = self.concepts_tokenizer(enigme, return_tensors="pt", padding=True,
                                                 truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs_concepts = self.concepts_model(**inputs_concepts)
            probs_concepts = torch.nn.functional.softmax(outputs_concepts.logits, dim=-1)
            concept_predicted = torch.argmax(probs_concepts, dim=1).item()
            concept_confidence = torch.max(probs_concepts).item()
        
        concept_name = self.concepts_labels.get(concept_predicted, "inconnu")
        
        return {
            "type": type_phrase,
            "type_confidence": type_confidence,
            "concept": concept_name,
            "concept_confidence": concept_confidence,
            "enigme": enigme
        }
    
    def detecter_patterns_logiques(self, enigme):
        """
        Détecte des patterns logiques spécifiques dans l'énigme
        """
        patterns = {
            "syllogisme": r"(tous les|toutes les).+(sont|est).+et.+(est|sont)",
            "implication": r"si.+(alors|donc|implique)",
            "negation": r"(ne .+ pas|non|aucun|jamais)",
            "quantificateurs": r"(tous|toutes|quelques|certains|aucun)",
            "comparaisons": r"(plus|moins|autant|égal|supérieur|inférieur)",
            "contraintes": r"(différent|même|unique|seul|tous différents)",
            "equations": r"\d+[x]\s*[\+\-\*\/]\s*\d+\s*=\s*\d+",
        }
        
        patterns_detectes = []
        for pattern_name, pattern_regex in patterns.items():
            if re.search(pattern_regex, enigme.lower()):
                patterns_detectes.append(pattern_name)
        
        return patterns_detectes
    
    def raisonner_etape_par_etape(self, analyse):
        """
        Génère un raisonnement étape par étape selon les patterns détectés
        """
        enigme = analyse["enigme"]
        concept = analyse["concept"]
        patterns = self.detecter_patterns_logiques(enigme)
        
        # Sélection du type de raisonnement selon les patterns
        if "syllogisme" in patterns:
            return self._raisonnement_syllogisme(enigme)
        elif "implication" in patterns:
            return self._raisonnement_implication(enigme)
        elif "equations" in patterns:
            return self._raisonnement_equation(enigme)
        elif "contraintes" in patterns or "comparaisons" in patterns:
            return self._raisonnement_contraintes(enigme)
        else:
            return self._raisonnement_general(enigme)
    
    def _raisonnement_syllogisme(self, enigme):
        """
        Template de raisonnement pour les syllogismes
        """
        etapes = [
            "Étape 1: Identifier les prémisses",
            "- Recherche de la règle générale (Tous les A sont B)",
            "- Identification du cas particulier (C est un A)",
            "Étape 2: Appliquer le syllogisme",
            "- Si tous les A sont B, et C est un A",
            "- Alors C est un B",
            "Étape 3: Formuler la conclusion logique",
            "- La conclusion découle directement des prémisses"
        ]
        
        return {
            "etapes": etapes,
            "type_raisonnement": "syllogisme",
            "confiance": 0.8
        }
    
    def _raisonnement_implication(self, enigme):
        """
        Template pour les implications logiques
        """
        etapes = [
            "Étape 1: Identifier la structure Si...Alors",
            "- Condition (prémisse): Si P",
            "- Conséquence: Alors Q", 
            "Étape 2: Vérifier si la condition est satisfaite",
            "- La prémisse P est-elle vraie ?",
            "Étape 3: Appliquer modus ponens si applicable",
            "- Si P est vrai et P → Q, alors Q est vrai",
            "Étape 4: Formuler la conclusion",
            "- Énoncer clairement ce qui peut être déduit"
        ]
        
        return {
            "etapes": etapes,
            "type_raisonnement": "implication",
            "confiance": 0.85
        }
    
    def _raisonnement_equation(self, enigme):
        """
        ✅ AJOUTÉ: Template pour les équations mathématiques
        """
        etapes = [
            "Étape 1: Identifier l'équation à résoudre",
            "- Repérer la variable inconnue (généralement x)",
            "- Noter la structure de l'équation",
            "Étape 2: Isoler la variable",
            "- Utiliser les opérations inverses",
            "- Maintenir l'équilibre des deux côtés",
            "Étape 3: Calculer la solution",
            "- Effectuer les calculs arithmétiques",
            "Étape 4: Vérifier la solution",
            "- Substituer dans l'équation originale"
        ]
        
        return {
            "etapes": etapes,
            "type_raisonnement": "equation",
            "confiance": 0.9
        }
    
    def _raisonnement_contraintes(self, enigme):
        """
        ✅ AJOUTÉ: Template pour les problèmes de contraintes et comparaisons
        """
        etapes = [
            "Étape 1: Lister les contraintes données",
            "- Identifier chaque relation (>, <, =, ≠)",
            "- Noter les entités impliquées",
            "Étape 2: Organiser les relations",
            "- Ordonner les contraintes de manière cohérente",
            "- Détecter d'éventuelles contradictions",
            "Étape 3: Déduire l'ordre ou la solution",
            "- Appliquer la transitivité des relations",
            "- Établir l'ordre complet si possible",
            "Étape 4: Formuler la réponse",
            "- Identifier l'élément recherché (plus grand, plus petit, etc.)"
        ]
        
        return {
            "etapes": etapes,
            "type_raisonnement": "contraintes",
            "confiance": 0.75
        }
    
    def _raisonnement_general(self, enigme):
        """
        ✅ AJOUTÉ: Raisonnement par défaut pour les cas non spécifiés
        """
        etapes = [
            "Étape 1: Analyser l'énigme donnée",
            "- Identifier les éléments clés du problème",
            "- Repérer les informations disponibles",
            "Étape 2: Déterminer l'approche de résolution",
            "- Quelle méthode logique appliquer ?",
            "- Quelles règles peuvent s'appliquer ?",
            "Étape 3: Formuler une hypothèse de solution",
            "- Proposer une réponse basée sur l'analyse",
            "Étape 4: Vérifier la cohérence",
            "- La solution respecte-t-elle toutes les contraintes ?",
            "Étape 5: Formuler la conclusion finale"
        ]
        
        return {
            "etapes": etapes,
            "type_raisonnement": "general",
            "confiance": 0.6
        }
    
    def auto_critiquer(self, raisonnement, enigme):
        """
        Fonction d'auto-critique du raisonnement
        """
        critiques = []
        
        # Vérifications logiques de base
        if raisonnement["type_raisonnement"] == "syllogisme":
            if "prémisse" not in " ".join(raisonnement["etapes"]).lower():
                critiques.append("❌ Prémisses non clairement identifiées")
            else:
                critiques.append("✅ Prémisses correctement identifiées")
        
        # Cohérence du raisonnement
        if len(raisonnement["etapes"]) < 3:
            critiques.append("❌ Raisonnement trop superficiel")
        else:
            critiques.append("✅ Raisonnement structuré en étapes")
        
        # Vérification de la conclusion
        if "conclusion" not in " ".join(raisonnement["etapes"]).lower():
            critiques.append("⚠️ Conclusion pas explicitement formulée")
        
        # Vérifications spécifiques par type
        if raisonnement["type_raisonnement"] == "equation":
            if "vérifier" in " ".join(raisonnement["etapes"]).lower():
                critiques.append("✅ Vérification de la solution incluse")
            else:
                critiques.append("⚠️ Vérification de la solution recommandée")
        
        return {
            "critiques": critiques,
            "score_coherence": len([c for c in critiques if c.startswith("✅")]) / len(critiques),
            "recommandations": self._generer_recommandations(critiques)
        }
    
    def _generer_recommandations(self, critiques):
        """
        Génère des recommandations d'amélioration
        """
        recommandations = []
        
        if any("Prémisses non clairement" in c for c in critiques):
            recommandations.append("Identifier plus clairement les prémisses de départ")
        
        if any("superficiel" in c for c in critiques):
            recommandations.append("Développer davantage les étapes intermédiaires")
        
        if any("Conclusion pas explicitement" in c for c in critiques):
            recommandations.append("Formuler une conclusion claire et explicite")
        
        if any("Vérification de la solution recommandée" in c for c in critiques):
            recommandations.append("Ajouter une étape de vérification de la solution")
        
        return recommandations
    
    def resoudre_enigme_complete(self, enigme):
        """
        Processus complet : Analyse → Raisonnement → Auto-critique
        """
        print(f"🔍 Analyse de l'énigme: '{enigme}'")
        
        # Étape 1: Analyse de l'énigme
        analyse = self.analyser_enigme(enigme)
        print(f"   Type détecté: {analyse['type']} (conf: {analyse['type_confidence']:.3f})")
        print(f"   Concept: {analyse['concept']} (conf: {analyse['concept_confidence']:.3f})")
        
        # Étape 2: Détection des patterns et raisonnement
        patterns = self.detecter_patterns_logiques(enigme)
        print(f"   Patterns logiques: {patterns}")
        
        raisonnement = self.raisonner_etape_par_etape(analyse)
        print(f"\n🧠 Raisonnement ({raisonnement['type_raisonnement']}):")
        for etape in raisonnement["etapes"]:
            print(f"   {etape}")
        
        # Étape 3: Auto-critique
        critique = self.auto_critiquer(raisonnement, enigme)
        print(f"\n🔍 Auto-critique (score: {critique['score_coherence']:.2f}):")
        for c in critique["critiques"]:
            print(f"   {c}")
        
        if critique["recommandations"]:
            print(f"\n💡 Recommandations:")
            for rec in critique["recommandations"]:
                print(f"   • {rec}")
        
        return {
            "analyse": analyse,
            "raisonnement": raisonnement,
            "critique": critique,
            "patterns": patterns
        }

# Test de l'hémisphère logique
if __name__ == "__main__":
    # Initialisation
    hemisphere = HemisphereLogique()
    
    # Tests avec différents types d'énigmes
    enigmes_test = [
        "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?",
        "Si A implique B, et A est vrai, que peut-on dire de B ?",
        "Alice est plus âgée que Bob. Bob est plus âgé que Charlie. Qui est le plus âgé ?",
        "Si 2x + 3 = 11, quelle est la valeur de x ?"
    ]
    
    for enigme in enigmes_test:
        print("="*80)
        resultat = hemisphere.resoudre_enigme_complete(enigme)
        print()
