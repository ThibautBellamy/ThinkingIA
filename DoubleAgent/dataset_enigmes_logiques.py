# dataset_enigmes_logiques.py
import json
import random

class DatasetEnigmesLogiques:
    """
    Créateur de dataset d'énigmes avec traces de raisonnement
    """
    def __init__(self):
        self.enigmes_base = {
            "deduction_simple": [
                {
                    "enigme": "Si tous les chats sont des mammifères et Félix est un chat, que peut-on dire de Félix ?",
                    "raisonnement_etapes": [
                        "Étape 1: Identifier les prémisses",
                        "- Prémisse 1: Tous les chats sont des mammifères",
                        "- Prémisse 2: Félix est un chat",
                        "Étape 2: Appliquer la règle de déduction (syllogisme)",
                        "- Si A → B et C = A, alors C → B",
                        "- Si (chat → mammifère) et (Félix = chat), alors (Félix → mammifère)",
                        "Étape 3: Formuler la conclusion",
                        "- Félix est un mammifère"
                    ],
                    "reponse": "Félix est un mammifère",
                    "critique": "Cette déduction est valide car elle suit correctement la règle du syllogisme. Les prémisses sont cohérentes et la conclusion découle logiquement.",
                    "type": "syllogisme_simple"
                },
                
                {
                    "enigme": "Dans une classe, tous les élèves qui réussissent font leurs devoirs. Marie réussit. Que peut-on dire de Marie ?",
                    "raisonnement_etapes": [
                        "Étape 1: Formaliser les prémisses",
                        "- Si élève réussit → élève fait ses devoirs",
                        "- Marie réussit",
                        "Étape 2: Appliquer modus ponens",
                        "- Si P → Q et P, alors Q",
                        "- Marie réussit → Marie fait ses devoirs",
                        "Étape 3: Conclure",
                        "- Marie fait ses devoirs"
                    ],
                    "reponse": "Marie fait ses devoirs",
                    "critique": "Raisonnement valide par modus ponens. La règle logique est correctement appliquée.",
                    "type": "modus_ponens"
                }
            ],
            
            "logique_propositionnelle": [
                {
                    "enigme": "Si A implique B, et B implique C, et A est vrai, que peut-on dire de C ?",
                    "raisonnement_etapes": [
                        "Étape 1: Identifier la chaîne d'implications",
                        "- A → B (si A alors B)",
                        "- B → C (si B alors C)",
                        "- A est vrai",
                        "Étape 2: Appliquer la transitivité",
                        "- A → B et B → C impliquent A → C",
                        "- Donc A → C",
                        "Étape 3: Appliquer modus ponens",
                        "- A est vrai et A → C",
                        "- Donc C est vrai"
                    ],
                    "reponse": "C est vrai",
                    "critique": "Raisonnement correct utilisant la transitivité des implications puis modus ponens.",
                    "type": "chaine_implications"
                }
            ],
            
            "enigmes_contraintes": [
                {
                    "enigme": "Trois amis (Alice, Bob, Charlie) ont des âges différents. Alice est plus âgée que Bob. Charlie est plus jeune que Bob. Qui est le plus âgé ?",
                    "raisonnement_etapes": [
                        "Étape 1: Formaliser les contraintes",
                        "- Alice > Bob (en âge)",
                        "- Charlie < Bob (en âge)",
                        "Étape 2: Ordonner les relations",
                        "- De Alice > Bob et Charlie < Bob",
                        "- On obtient : Charlie < Bob < Alice",
                        "Étape 3: Identifier l'ordre complet",
                        "- Alice est la plus âgée",
                        "- Bob est au milieu",
                        "- Charlie est le plus jeune"
                    ],
                    "reponse": "Alice est la plus âgée",
                    "critique": "Raisonnement correct par ordonnancement des relations. La transitivité des inégalités est bien appliquée.",
                    "type": "contraintes_ordre"
                }
            ],
            
            "paradoxes_logiques": [
                {
                    "enigme": "Un barbier rase tous les hommes qui ne se rasent pas eux-mêmes. Qui rase le barbier ?",
                    "raisonnement_etapes": [
                        "Étape 1: Analyser les possibilités",
                        "- Cas 1: Le barbier se rase lui-même",
                        "- Cas 2: Le barbier ne se rase pas lui-même",
                        "Étape 2: Tester le Cas 1",
                        "- Si le barbier se rase lui-même, alors il fait partie des hommes qui se rasent eux-mêmes",
                        "- Mais la règle dit qu'il ne rase que ceux qui ne se rasent PAS eux-mêmes",
                        "- Contradiction : il ne devrait pas se raser",
                        "Étape 3: Tester le Cas 2",
                        "- Si le barbier ne se rase pas lui-même, alors il fait partie des hommes qui ne se rasent pas eux-mêmes",
                        "- Donc selon la règle, le barbier doit le raser",
                        "- Contradiction : cela signifie qu'il se rase lui-même",
                        "Étape 4: Conclusion",
                        "- Cette situation est paradoxale (paradoxe du barbier de Russell)"
                    ],
                    "reponse": "Cette situation est un paradoxe logique sans solution cohérente",
                    "critique": "L'analyse révèle correctement le paradoxe. Les deux cas mènent à des contradictions, démontrant l'impossibilité logique de la situation.",
                    "type": "paradoxe"
                }
            ],
            
            "enigmes_numeriques": [
                {
                    "enigme": "Si 2x + 3 = 11, quelle est la valeur de x ?",
                    "raisonnement_etapes": [
                        "Étape 1: Isoler le terme avec x",
                        "- 2x + 3 = 11",
                        "- Soustraire 3 des deux côtés : 2x = 11 - 3",
                        "- 2x = 8",
                        "Étape 2: Résoudre pour x",
                        "- Diviser par 2 : x = 8/2",
                        "- x = 4",
                        "Étape 3: Vérification",
                        "- Remplacer x par 4 dans l'équation originale",
                        "- 2(4) + 3 = 8 + 3 = 11 ✓"
                    ],
                    "reponse": "x = 4",
                    "critique": "Résolution algébrique correcte avec vérification. Chaque étape respecte les règles mathématiques.",
                    "type": "equation_lineaire"
                }
            ]
        }
    
    def generer_variations_enigmes(self, nombre_par_type=50):
        """
        Génère des variations d'énigmes pour l'entraînement
        """
        dataset_complet = []
        
        for type_enigme, enigmes in self.enigmes_base.items():
            print(f"Génération de variations pour {type_enigme}...")
            
            for enigme_base in enigmes:
                # Ajouter l'énigme de base
                dataset_complet.append(self._formater_pour_entrainement(enigme_base))
                
                # Générer des variations
                for i in range(nombre_par_type):
                    enigme_variee = self._creer_variation(enigme_base, type_enigme)
                    dataset_complet.append(self._formater_pour_entrainement(enigme_variee))
        
        return dataset_complet
    
    def _creer_variation(self, enigme_base, type_enigme):
        """
        Crée une variation d'une énigme selon son type
        """
        noms = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hugo"]
        animaux = ["chat", "chien", "oiseau", "poisson", "lapin"]
        objets = ["livre", "stylo", "ordinateur", "téléphone", "clé"]
        
        enigme_variee = enigme_base.copy()
        
        if type_enigme == "deduction_simple":
            # Changer les noms et objets
            ancien_nom = "Félix"
            nouveau_nom = random.choice(noms)
            enigme_variee["enigme"] = enigme_base["enigme"].replace(ancien_nom, nouveau_nom)
            
            # Adapter le raisonnement
            nouveaux_etapes = []
            for etape in enigme_base["raisonnement_etapes"]:
                nouveaux_etapes.append(etape.replace(ancien_nom, nouveau_nom))
            enigme_variee["raisonnement_etapes"] = nouveaux_etapes
            enigme_variee["reponse"] = enigme_base["reponse"].replace(ancien_nom, nouveau_nom)
        
        elif type_enigme == "contraintes_ordre":
            # Mélanger les noms
            nouveaux_noms = random.sample(noms, 3)
            anciens_noms = ["Alice", "Bob", "Charlie"]
            
            enigme_text = enigme_base["enigme"]
            for ancien, nouveau in zip(anciens_noms, nouveaux_noms):
                enigme_text = enigme_text.replace(ancien, nouveau)
            enigme_variee["enigme"] = enigme_text
            
            # Adapter tout le raisonnement
            nouveaux_etapes = []
            for etape in enigme_base["raisonnement_etapes"]:
                etape_modifiee = etape
                for ancien, nouveau in zip(anciens_noms, nouveaux_noms):
                    etape_modifiee = etape_modifiee.replace(ancien, nouveau)
                nouveaux_etapes.append(etape_modifiee)
            enigme_variee["raisonnement_etapes"] = nouveaux_etapes
            
            enigme_variee["reponse"] = enigme_base["reponse"].replace("Alice", nouveaux_noms[0])
        
        elif type_enigme == "equation_lineaire":
            # Générer de nouveaux nombres
            a = random.randint(2, 5)  # coefficient
            b = random.randint(1, 10)  # constante
            x = random.randint(1, 10)  # solution
            resultat = a * x + b
            
            enigme_variee["enigme"] = f"Si {a}x + {b} = {resultat}, quelle est la valeur de x ?"
            
            enigme_variee["raisonnement_etapes"] = [
                "Étape 1: Isoler le terme avec x",
                f"- {a}x + {b} = {resultat}",
                f"- Soustraire {b} des deux côtés : {a}x = {resultat} - {b}",
                f"- {a}x = {resultat - b}",
                "Étape 2: Résoudre pour x",
                f"- Diviser par {a} : x = {resultat - b}/{a}",
                f"- x = {x}",
                "Étape 3: Vérification",
                f"- Remplacer x par {x} dans l'équation originale",
                f"- {a}({x}) + {b} = {a*x} + {b} = {resultat} ✓"
            ]
            
            enigme_variee["reponse"] = f"x = {x}"
        
        return enigme_variee
    
    def _formater_pour_entrainement(self, enigme):
        """
        Formate une énigme pour l'entraînement du modèle
        """
        # Concaténer toutes les étapes de raisonnement
        raisonnement_complet = "\n".join(enigme["raisonnement_etapes"])
        
        # Format d'entrée pour le modèle
        input_text = f"ÉNIGME: {enigme['enigme']}\nRAISONNEMENT:"
        
        # Format de sortie attendu
        output_text = f"{raisonnement_complet}\nRÉPONSE: {enigme['reponse']}\nCRITIQUE: {enigme['critique']}"
        
        return {
            "input": input_text,
            "output": output_text,
            "type": enigme["type"],
            "enigme_seule": enigme["enigme"],
            "reponse_seule": enigme["reponse"]
        }
    
    def sauvegarder_dataset_enigmes(self, dataset, filename="dataset_enigmes_logiques.json"):
        """
        Sauvegarde le dataset d'énigmes
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Dataset d'énigmes sauvegardé : {len(dataset)} exemples")
        
        # Statistiques par type
        types_stats = {}
        for item in dataset:
            type_enigme = item['type']
            types_stats[type_enigme] = types_stats.get(type_enigme, 0) + 1
        
        print("\n📈 Distribution par type d'énigme:")
        for type_enigme, count in sorted(types_stats.items()):
            print(f"   {type_enigme:20} : {count:4} exemples")

# Utilisation
if __name__ == "__main__":
    creator = DatasetEnigmesLogiques()
    dataset = creator.generer_variations_enigmes(nombre_par_type=30)
    creator.sauvegarder_dataset_enigmes(dataset)
