# dataset_concepts_enrichi.py
import json
import random
import sys

class DatasetConceptsEnrichi:
    """
    Dataset enrichi avec beaucoup plus d'exemples pour chaque concept
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
                # Faits scientifiques
                "La température est de 20 degrés",
                "Cette méthode fonctionne correctement",
                "Le projet sera terminé demain", 
                "Python est un langage de programmation",
                "Cette approche s'avère efficace",
                
                # Déclarations factuelles
                "Il pleut depuis ce matin",
                "La réunion commence à 14h30",
                "Le rapport contient 50 pages",
                "Cette entreprise existe depuis 1995",
                "Le train part de la voie 3",
                "Cette information figure page 25",
                "Le taux d'inflation atteint 3%",
                
                # Constatations objectives
                "Les résultats montrent une amélioration",
                "Cette technique donne des résultats probants",
                "L'expérience confirme notre hypothèse",
                "Les données indiquent une tendance positive",
                "Cette mesure produit l'effet escompté",
                "L'analyse révèle des points importants",
                
                # Faits établis
                "Cette loi est entrée en vigueur hier",
                "Le contrat expire le mois prochain",
                "Cette version corrige les bugs précédents",
                "Le nouveau système est opérationnel",
                "Cette procédure respecte les normes",
                "Le budget alloué s'élève à 50 000 euros",
                
                # Informations vérifiables
                "Cette étude porte sur 1000 participants",
                "Le taux de réussite atteint 95%",
                "Cette formation dure trois jours",
                "L'événement se déroule au Palais des Congrès",
                "Cette technologie existe depuis 2010",
                "Le délai de livraison est de deux semaines",
                
                # États de fait
                "Tous les documents sont signés",
                "La base de données contient 10 000 entrées",
                "Cette version inclut de nouvelles fonctionnalités",
                "Le serveur fonctionne correctement",
                "Cette machine produit 100 pièces par heure",
                "Le test d'intégration est réussi"
            ],
            
            "demande_action": [
                # Demandes polies
                "Peux-tu m'aider à résoudre ceci ?",
                "Veux-tu bien faire cette tâche ?",
                "Pourrais-tu m'expliquer la méthode ?",
                "Aide-moi à comprendre ce concept",
                "Montre-moi comment procéder",
                
                # Demandes formelles
                "Pourriez-vous bien vouloir m'assister ?",
                "Seriez-vous en mesure de m'orienter ?",
                "Auriez-vous l'amabilité de m'expliquer ?",
                "Pourriez-vous avoir la gentillesse de ?",
                "Voudriez-vous bien me renseigner sur ?",
                
                # Demandes directes
                "Fais-moi un résumé de ce document",
                "Envoie-moi le rapport demain matin",
                "Prépare la présentation pour jeudi",
                "Vérifie ces calculs s'il te plaît",
                "Corrige cette erreur rapidement",
                "Finalise ce projet avant vendredi",
                
                # Requêtes techniques
                "Peux-tu déboguer ce code ?",
                "Lance l'analyse sur ce dataset",
                "Configure ce serveur pour la production",
                "Optimise cette requête SQL",
                "Teste cette nouvelle fonctionnalité",
                "Déploie cette version en staging",
                
                # Demandes de collaboration
                "Travaillons ensemble sur ce problème",
                "Pouvons-nous planifier une session de brainstorming ?",
                "Organisons une réunion pour faire le point",
                "Coordonnons nos efforts sur ce projet",
                "Répartissons-nous les tâches équitablement",
                
                # Instructions spécifiques
                "Commence par analyser les données",
                "Prends en compte ces contraintes",
                "Respecte ces spécifications techniques",
                "Suis cette procédure étape par étape",
                "Applique cette méthode rigoureusement",
                "Intègre ces modifications au code",
                
                # Demandes urgentes
                "Il faut absolument terminer ça aujourd'hui",
                "Peux-tu traiter ça en priorité ?",
                "Cette tâche ne peut pas attendre",
                "Il est urgent de résoudre ce problème",
                "Dépêche-toi de finir cette partie"
            ],
            
            "expression_emotion": [
                # Joie et satisfaction
                "Je suis vraiment content de ce résultat",
                "Je suis fier de notre accomplissement",
                "Cela me remplit de bonheur",
                "Je déborde de joie en voyant ça",
                "Cette réussite me ravit énormément",
                "Je suis aux anges avec ce succès",
                "Quelle fierté de voir ce projet aboutir",
                
                # Tristesse et déception
                "Cela me rend triste de voir ça", 
                "Je suis déçu par ces résultats",
                "Cette situation me chagrine profondément",
                "J'ai le cœur lourd en pensant à ça",
                "Cette nouvelle m'attriste beaucoup",
                "Je ressens une grande mélancolie",
                "Cette défaite me décourage énormément",
                
                # Peur et anxiété
                "J'ai peur que ça ne marche pas",
                "Cette situation m'inquiète beaucoup",
                "Je suis angoissé par cette perspective",
                "Cette incertitude me stresse énormément",
                "J'appréhende la suite des événements",
                "Cette menace me terrorise",
                "Je tremble à l'idée de",
                
                # Colère et frustration
                "Cette injustice me met en colère",
                "Je suis furieux de cette décision",
                "Cela m'exaspère au plus haut point",
                "Cette situation me frustre énormément",
                "Je bouillonne de rage intérieurement",
                "Cette attitude m'agace profondément",
                "Je suis outré par ce comportement",
                
                # Surprise et étonnement
                "Je suis stupéfait par cette nouvelle",
                "Cette révélation me surprend énormément",
                "Je n'en reviens pas de ce retournement",
                "Cette découverte me sidère complètement",
                "Je reste bouche bée devant ce résultat",
                "Cette performance m'épate vraiment",
                
                # Amour et affection
                "J'adore cette nouvelle approche",
                "Je suis fou amoureux de cette idée",
                "Cette méthode me plaît énormément",
                "J'ai un faible pour cette solution",
                "Cette proposition me séduit vraiment",
                "Je craque complètement pour ce concept",
                
                # Espoir et optimisme
                "J'espère sincèrement que ça marchera",
                "Je garde confiance en notre capacité",
                "Cette possibilité m'encourage beaucoup",
                "Je vois l'avenir avec optimisme",
                "Cette perspective me donne de l'espoir",
                "Je crois fermement en notre succès"
            ],
            
            "analyse_critique": [
                # Critique constructive
                "Cette approche présente des failles",
                "Il faut examiner cette méthode plus attentivement", 
                "Cette conclusion me semble discutable",
                "Analysons les points faibles de cette théorie",
                "Cette démonstration contient des erreurs",
                
                # Évaluation méthodologique
                "Cette étude manque de rigueur scientifique",
                "L'échantillon utilisé n'est pas représentatif",
                "Cette méthodologie présente des biais évidents",
                "Les variables n'ont pas été correctement contrôlées",
                "Cette analyse statistique est insuffisante",
                "Les conclusions dépassent ce que montrent les données",
                
                # Remise en question
                "Cette hypothèse mérite d'être questionnée",
                "Il convient de remettre en cause ce postulat",
                "Cette affirmation demande à être vérifiée",
                "Ce raisonnement souffre d'incohérences",
                "Cette logique présente des contradictions",
                "Cette argumentation manque de solidité",
                
                # Analyse comparative
                "Cette solution est moins efficace que l'alternative",
                "Comparativement, cette méthode montre des limites",
                "Cette approche pâlit face à la concurrence",
                "En regard des autres options, celle-ci déçoit",
                "Cette performance reste en deçà des attentes",
                "Cette proposition ne rivalise pas avec",
                
                # Critique technique
                "Cette architecture logicielle présente des vulnérabilités",
                "Ce code manque d'optimisation et de clarté",
                "Cette conception ignore les bonnes pratiques",
                "Cette implémentation souffre de problèmes de performance",
                "Cette solution technique n'est pas scalable",
                "Ce design pattern n'est pas approprié ici",
                
                # Évaluation stratégique
                "Cette stratégie néglige des aspects cruciaux",
                "Ce plan présente des risques sous-estimés",
                "Cette décision manque de vision à long terme",
                "Cette politique ignore les effets secondaires",
                "Cette approche manque de cohérence globale",
                "Cette orientation stratégique est questionnaire"
            ]
        }
    
    def generer_dataset_concepts_enrichi(self, taille_par_concept=500):
        """
        Génère un dataset très enrichi avec beaucoup d'exemples
        """
        dataset = []
        
        # Ajouter toutes les données de base
        for i, (concept, phrases) in enumerate(self.data_concepts.items()):
            for phrase in phrases:
                dataset.append({"text": phrase, "label": i, "concept": concept})
        
        # Générer des variations supplémentaires
        for i, (concept, phrases_base) in enumerate(self.data_concepts.items()):
            phrases_existantes = [d['text'] for d in dataset if d['label'] == i]
            
            while len(phrases_existantes) < taille_par_concept:
                phrase_base = random.choice(self.data_concepts[concept])
                phrase_variee = self._creer_variations_avancees(phrase_base, concept)
                
                # Éviter les doublons
                if phrase_variee not in phrases_existantes:
                    dataset.append({"text": phrase_variee, "label": i, "concept": concept})
                    phrases_existantes.append(phrase_variee)
        
        random.shuffle(dataset)
        return dataset
    
    def _creer_variations_avancees(self, phrase, concept):
        """
        Crée des variations sophistiquées selon le concept
        """
        variations_par_concept = {
            "salutation_presentation": [
                phrase.replace("je m'appelle", "je me nomme"),
                phrase.replace("je suis", "moi c'est"),
                phrase.replace("Bonjour", "Salut"),
                phrase.replace("Bonsoir", "Hello"),
                phrase.replace("je me présente", "permettez-moi de me présenter"),
                phrase.replace("nouveau", "nouvel arrivant"),
                phrase.replace("collègue", "coéquipier"),
            ],
            
            "questionnement_interrogation": [
                phrase.replace("Qu'est-ce que", "Que"),
                phrase.replace("Comment", "De quelle manière"),
                phrase.replace("Pourquoi", "Pour quelle raison"),
                phrase.replace("tu penses", "vous pensez"),
                phrase.replace("peux-tu", "pourriez-vous"),
                phrase.replace("?", " exactement ?"),
                phrase.replace("Est-ce que", "Pensez-vous que"),
            ],
            
            "raisonnement_logique": [
                phrase.replace("cette", "cette présente"),
                phrase.replace("logiquement", "de manière logique"),
                phrase.replace("implique", "entraîne"),
                phrase.replace("conclusion", "déduction"),
                phrase.replace("prémisse", "postulat de base"),
                phrase.replace("raisonnement", "argumentation"),
            ],
            
            # Autres variations...
        }
        
        if concept in variations_par_concept:
            variations = variations_par_concept[concept]
            variation_choisie = random.choice(variations + [phrase])
            return variation_choisie
        
        return phrase
    
    def sauvegarder_dataset_enrichi(self, dataset, filename="./datasets/dataset_concepts.json"):
        """
        Sauvegarde le dataset enrichi
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Dataset ENRICHI sauvegardé : {len(dataset)} exemples au total")
        
        # Statistiques détaillées
        concepts_stats = {}
        for item in dataset:
            concept = item['concept']
            concepts_stats[concept] = concepts_stats.get(concept, 0) + 1
        
        print("\n📈 Distribution par concept:")
        for concept, count in sorted(concepts_stats.items()):
            print(f"   {concept:25} : {count:4} exemples")
        
        total_examples = sum(concepts_stats.values())
        print(f"\n🎯 Total général: {total_examples} exemples")
        print(f"⚖️  Équilibre: {total_examples // len(concepts_stats)} exemples/concept en moyenne")

# Utilisation
if __name__ == "__main__":
    creator = DatasetConceptsEnrichi()
    taille_par_concept = 600
    if len(sys.argv) > 1:
        try:
            taille_par_concept = int(sys.argv[1])
        except ValueError:
            print("Usage: python dataset_creator.py [taille_par_concept]")
            sys.exit(1)  
    dataset = creator.generer_dataset_concepts_enrichi(taille_par_concept)  # 600 exemples par concept !
    creator.sauvegarder_dataset_enrichi(dataset)
