# dataset_concepts_enrichi.py
import json
import random
import sys

class DatasetConceptsEnrichi:
    """
    Version corrigée qui évite les boucles infinies
    """
    def __init__(self):
        self.data_concepts = {
            "salutation_presentation": [
                # Salutations de base
                "Bonjour, je m'appelle Marie",
                "Salut, je suis Paul", 
                "Bonsoir, moi c'est Sophie",
                "Hello, je me présente : Thomas",
                "Coucou, je suis ta nouvelle collègue",
                "Salut tout le monde, je suis le nouveau",
                "Bonjour à tous, je me nomme Pierre",
                
                # Présentations formelles
                "Bonjour Monsieur, permettez-moi de me présenter",
                "Bonsoir Madame, je suis ravi de faire votre connaissance",
                "Enchanté, je me nomme Jean-Claude Durand",
                "Ravi de vous rencontrer, je suis la directrice",
                "Permettez-moi de me présenter : Docteur Martin",
                "Je me permets de me présenter, je suis votre nouveau voisin",
                "Bonjour, je suis le responsable du projet",
                
                # Présentations informelles
                "Salut, moi c'est Alex et toi ?",
                "Hey ! Je m'appelle Emma, enchantée !",
                "Coucou tout le monde ! Moi c'est Léa",
                "Yo ! Je suis le pote de Marc",
                "Salut les amis, je me présente : Julien",
                "Hello ! Moi c'est Sarah, ravie de vous voir",
                "Coucou ! Je suis la copine de Lisa",
                
                # Contextes spécifiques
                "Bonjour, je suis votre nouveau professeur",
                "Salut, je viens d'emménager dans le quartier",
                "Bonsoir, je suis invité par Catherine",
                "Hello, je suis stagiaire ici pour deux mois",
                "Coucou, je suis la baby-sitter des enfants",
                "Bonjour, je représente l'entreprise Dupont",
                "Salut, on m'a dit de venir vous voir",
                
                # Avec contexte temporel
                "Bonne matinée ! Je me présente : Antoine",
                "Bon après-midi, moi c'est Valérie",
                "Rebonjour, je suis passé ce matin mais vous n'étiez pas là",
                "Re-salut ! C'est encore moi, David",
                
                # Présentations de groupe
                "Bonjour à tous, nous sommes l'équipe de marketing",
                "Salut la compagnie ! On est les nouveaux",
                "Hello everyone ! Nous représentons l'association",
                "Coucou les filles ! C'est nous, les cousines de Nice"
            ],
            
            "questionnement_interrogation": [
                # Questions simples
                "Qu'est-ce que tu penses de ça ?",
                "Comment résoudre ce problème ?",
                "Pourquoi cette méthode ne fonctionne pas ?",
                "Où trouver plus d'informations ?",
                "Quand aura lieu la réunion ?",
                "Qui est responsable de ce projet ?",
                
                # Questions avec inversion
                "Est-ce que tu comprends ?",
                "Penses-tu que c'est possible ?",
                "Sais-tu comment faire ?",
                "Veux-tu bien m'expliquer ?",
                "Pourrais-tu me dire pourquoi ?",
                "Serais-tu d'accord pour ?",
                
                # Questions rhétoriques
                "N'est-ce pas évident ?",
                "Que peut-on y faire ?",
                "Qui aurait pu prévoir ça ?",
                "Comment aurions-nous pu savoir ?",
                "N'est-ce pas formidable ?",
                "Que dire de plus ?",
                
                # Questions de clarification
                "Tu veux dire quoi exactement ?",
                "Dans quel sens l'entends-tu ?",
                "Peux-tu préciser ta pensée ?",
                "Que signifie cette expression ?",
                "Comment interpréter cette phrase ?",
                "Quelle est ta définition de ?",
                
                # Questions d'opinion
                "Quel est ton avis sur cette question ?",
                "Que penses-tu de cette approche ?",
                "Comment juges-tu cette situation ?",
                "Quelle est ta position là-dessus ?",
                "Comment vois-tu les choses ?",
                "Quel est ton sentiment sur ce point ?",
                
                # Questions pratiques
                "Comment procède-t-on concrètement ?",
                "Quelles sont les étapes à suivre ?",
                "Combien de temps cela prendra-t-il ?",
                "Quels outils faut-il utiliser ?",
                "Où doit-on commencer ?",
                "Qui peut nous aider dans cette tâche ?",
                
                # Questions existentielles/philosophiques
                "Quel est le sens de tout ça ?",
                "Pourquoi sommes-nous ici ?",
                "Qu'est-ce qui nous motive vraiment ?",
                "Comment définir le bonheur ?",
                "Que signifie être libre ?",
                "Quelle est notre raison d'être ?",
                
                # Questions techniques
                "Comment fonctionne ce mécanisme ?",
                "Quelle est la cause de cette panne ?",
                "Pourquoi ce code ne s'exécute pas ?",
                "Comment optimiser cette performance ?",
                "Quelle méthode est la plus efficace ?",
                "Comment déboguer ce problème ?"
            ],
            
            "raisonnement_logique": [
                # Logique formelle
                "Si A implique B, alors non-B implique non-A",
                "Cette conclusion découle logiquement des prémisses",
                "Analysons cette proposition étape par étape",
                "Le raisonnement par l'absurde montre que",
                "Cette déduction est cohérente avec les faits",
                
                # Raisonnement causal
                "La cause première de ce phénomène est",
                "Cet effet résulte directement de cette cause",
                "Il existe une relation de causalité entre",
                "Cette conséquence découle naturellement de",
                "On peut établir un lien logique entre",
                
                # Raisonnement inductif/déductif
                "À partir de ces observations, on peut déduire que",
                "Ces exemples particuliers nous mènent à la règle générale",
                "Cette loi générale s'applique à ce cas particulier",
                "L'induction nous permet de conclure que",
                "Par déduction, nous arrivons à",
                
                # Analyse logique
                "Cette argumentation présente une faille logique",
                "Le syllogisme se structure de la manière suivante",
                "Cette prémisse majeure combinée à cette prémisse mineure",
                "L'enchaînement logique suit cette progression",
                "Cette démonstration repose sur ces axiomes",
                
                # Raisonnement mathématique
                "Cette équation se résout par élimination",
                "En appliquant ce théorème, on obtient",
                "Cette propriété mathématique implique que",
                "Par récurrence, on peut démontrer que",
                "Cette fonction vérifie les conditions suivantes",
                
                # Pensée systémique
                "Dans ce système, chaque élément interagit avec",
                "Cette approche holistique révèle que",
                "L'analyse systémique montre l'interdépendance",
                "Cette vision globale met en évidence",
                "Les rétroactions dans ce système provoquent",
                
                # Raisonnement critique
                "Cette hypothèse mérite d'être testée car",
                "Les preuves empiriques soutiennent l'idée que",
                "Cette théorie explique de manière cohérente",
                "Les données convergent vers cette conclusion",
                "Cette corrélation suggère une relation causale"
            ],
            
            "affirmation_factuelle": [
                "La température est de 20 degrés",
                "Cette méthode fonctionne correctement",
                "Le projet sera terminé demain", 
                "Python est un langage de programmation",
                "Cette approche s'avère efficace",
                "Il pleut depuis ce matin",
                "La réunion commence à 14h30",
                "Le rapport contient 50 pages",
                "Cette entreprise existe depuis 1995",
                "Le train part de la voie 3",
                "Cette information figure page 25",
                "Le taux d'inflation atteint 3%",
                "Les résultats montrent une amélioration",
                "Cette technique donne des résultats probants",
                "L'expérience confirme notre hypothèse",
                "Les données indiquent une tendance positive",
                "Cette mesure produit l'effet escompté",
                "L'analyse révèle des points importants",
                "Cette loi est entrée en vigueur hier",
                "Le contrat expire le mois prochain",
                "Cette version corrige les bugs précédents",
                "Le nouveau système est opérationnel",
                "Cette procédure respecte les normes",
                "Le budget alloué s'élève à 50 000 euros"
            ],
            
            "demande_action": [
                "Peux-tu m'aider à résoudre ceci ?",
                "Veux-tu bien faire cette tâche ?",
                "Pourrais-tu m'expliquer la méthode ?",
                "Aide-moi à comprendre ce concept",
                "Montre-moi comment procéder",
                "Pourriez-vous bien vouloir m'assister ?",
                "Seriez-vous en mesure de m'orienter ?",
                "Auriez-vous l'amabilité de m'expliquer ?",
                "Fais-moi un résumé de ce document",
                "Envoie-moi le rapport demain matin",
                "Prépare la présentation pour jeudi",
                "Vérifie ces calculs s'il te plaît",
                "Corrige cette erreur rapidement",
                "Finalise ce projet avant vendredi",
                "Peux-tu déboguer ce code ?",
                "Lance l'analyse sur ce dataset",
                "Configure ce serveur pour la production",
                "Optimise cette requête SQL",
                "Teste cette nouvelle fonctionnalité",
                "Déploie cette version en staging"
            ],
            
            "expression_emotion": [
                "Je suis vraiment content de ce résultat",
                "Je suis fier de notre accomplissement",
                "Cela me remplit de bonheur",
                "Je déborde de joie en voyant ça",
                "Cette réussite me ravit énormément",
                "Cela me rend triste de voir ça", 
                "Je suis déçu par ces résultats",
                "Cette situation me chagrine profondément",
                "J'ai peur que ça ne marche pas",
                "Cette situation m'inquiète beaucoup",
                "Je suis angoissé par cette perspective",
                "Cette injustice me met en colère",
                "Je suis furieux de cette décision",
                "Cela m'exaspère au plus haut point",
                "Je suis stupéfait par cette nouvelle",
                "Cette révélation me surprend énormément",
                "J'adore cette nouvelle approche",
                "Je suis fou amoureux de cette idée",
                "J'espère sincèrement que ça marchera",
                "Je garde confiance en notre capacité"
            ],
            
            "analyse_critique": [
                "Cette approche présente des failles",
                "Il faut examiner cette méthode plus attentivement", 
                "Cette conclusion me semble discutable",
                "Analysons les points faibles de cette théorie",
                "Cette démonstration contient des erreurs",
                "Cette étude manque de rigueur scientifique",
                "L'échantillon utilisé n'est pas représentatif",
                "Cette méthodologie présente des biais évidents",
                "Cette hypothèse mérite d'être questionnée",
                "Il convient de remettre en cause ce postulat",
                "Cette solution est moins efficace que l'alternative",
                "Comparativement, cette méthode montre des limites",
                "Cette architecture logicielle présente des vulnérabilités",
                "Ce code manque d'optimisation et de clarté",
                "Cette stratégie néglige des aspects cruciaux",
                "Ce plan présente des risques sous-estimés"
            ]
        }
    
    def generer_variations_intelligentes(self, phrase_base, concept):
        """
        Génère des variations plus créatives pour éviter l'épuisement
        """
        # Noms variés pour les substitutions
        noms = ["Marie", "Paul", "Sophie", "Thomas", "Emma", "Lucas", "Léa", "Antoine", 
                "Valérie", "Pierre", "Julie", "Marc", "Sarah", "David", "Lisa", "Jean"]
        
        # Salutations variées
        salutations = ["Bonjour", "Salut", "Bonsoir", "Hello", "Coucou", "Hey"]
        
        # Verbes de présentation
        verbes_presentation = ["je m'appelle", "je me nomme", "moi c'est", "je suis"]
        
        variations = [phrase_base]  # Inclure la phrase de base
        
        if concept == "salutation_presentation":
            # Changer les noms
            for nom in random.sample(noms, 3):
                for ancien_nom in noms:
                    if ancien_nom in phrase_base:
                        variations.append(phrase_base.replace(ancien_nom, nom))
            
            # Changer les salutations
            for salut in salutations:
                for ancien_salut in salutations:
                    if ancien_salut in phrase_base:
                        variations.append(phrase_base.replace(ancien_salut, salut))
            
            # Changer les verbes de présentation
            for verbe in verbes_presentation:
                for ancien_verbe in verbes_presentation:
                    if ancien_verbe in phrase_base:
                        variations.append(phrase_base.replace(ancien_verbe, verbe))
        
        elif concept == "questionnement_interrogation":
            # Variations de politesse
            variations.extend([
                phrase_base.replace("tu", "vous"),
                phrase_base.replace("peux-tu", "pourriez-vous"),
                phrase_base.replace("Comment", "De quelle manière"),
                phrase_base.replace("Pourquoi", "Pour quelle raison"),
                phrase_base.replace("?", " exactement ?"),
                phrase_base.replace("Qu'est-ce que", "Que")
            ])
        
        elif concept == "affirmation_factuelle":
            # Variations de temps et quantités
            variations.extend([
                phrase_base.replace("20", str(random.randint(15, 25))),
                phrase_base.replace("50", str(random.randint(40, 60))),
                phrase_base.replace("demain", "bientôt"),
                phrase_base.replace("correctement", "parfaitement"),
                phrase_base.replace("efficace", "performante")
            ])
        
        # Retourner une variation aléatoire différente de l'originale
        variations_uniques = list(set(variations))  # Éliminer les doublons
        if len(variations_uniques) > 1:
            return random.choice([v for v in variations_uniques if v != phrase_base])
        else:
            # Si pas de variation possible, ajouter un suffixe aléatoire
            suffixes = [" vraiment", " effectivement", " certainement", " absolument"]
            return phrase_base + random.choice(suffixes)
    
    def generer_dataset_concepts_securise(self, taille_cible_par_concept=200):
        """
        Génération sécurisée qui évite les boucles infinies
        """
        dataset = []
        
        print("🔄 Génération du dataset sécurisée...")
        
        for i, (concept, phrases_base) in enumerate(self.data_concepts.items()):
            print(f"   Traitement concept: {concept}")
            
            # Ajouter toutes les phrases de base
            for phrase in phrases_base:
                dataset.append({"text": phrase, "label": i, "concept": concept})
            
            # Calculer combien de variations ajouter
            phrases_existantes = set(phrases_base)  # Utiliser un set pour rapidité
            variations_ajoutees = 0
            max_tentatives = taille_cible_par_concept * 10  # Limite de sécurité
            tentatives = 0
            
            while len(phrases_existantes) < taille_cible_par_concept and tentatives < max_tentatives:
                phrase_base = random.choice(phrases_base)  # Toujours partir des phrases originales
                nouvelle_variation = self.generer_variations_intelligentes(phrase_base, concept)
                
                if nouvelle_variation not in phrases_existantes:
                    dataset.append({"text": nouvelle_variation, "label": i, "concept": concept})
                    phrases_existantes.add(nouvelle_variation)
                    variations_ajoutees += 1
                
                tentatives += 1
            
            print(f"     Base: {len(phrases_base)}, Variations: {variations_ajoutees}, Total: {len(phrases_existantes)}")
        
        random.shuffle(dataset)
        return dataset
    
    def sauvegarder_dataset_securise(self, dataset, filename="datasets/dataset_concepts.json"):
        """
        Sauvegarde avec statistiques complètes
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Dataset SÉCURISÉ sauvegardé : {len(dataset)} exemples")
        
        # Statistiques détaillées
        concepts_stats = {}
        for item in dataset:
            concept = item['concept']
            concepts_stats[concept] = concepts_stats.get(concept, 0) + 1
        
        print("\n📈 Distribution finale par concept:")
        for concept, count in sorted(concepts_stats.items()):
            print(f"   {concept:25} : {count:4} exemples")
        
        total = sum(concepts_stats.values())
        moyenne = total // len(concepts_stats)
        print(f"\n🎯 Total: {total} | Moyenne: {moyenne} exemples/concept")
        
        return concepts_stats

# Utilisation
if __name__ == "__main__":
    creator = DatasetConceptsEnrichi()
    taille_par_concept = 150
    if len(sys.argv) > 1:
        try:
            taille_par_concept = int(sys.argv[1])
            if taille_par_concept > 300:
                print("⚠️ Taille limitée à 300 pour éviter les boucles infinies")
                taille_par_concept = 300
        except ValueError:
            print("Usage: python dataset_creator.py [taille_par_concept]")
            sys.exit(1)  
            
    print(f"🎯 Génération de {taille_par_concept} exemples par concept")
    dataset = creator.generer_dataset_concepts_securise(taille_par_concept)  # taille_par_concept exemples par concept !
    creator.sauvegarder_dataset_securise(dataset)
    
    print("\n✅ Dataset créé avec succès !")
