#!/usr/bin/env python3
"""
Point d'entrée principal pour l'IA de raisonnement autonome - version YAML Config
"""

import torch
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import du système de configuration YAML
from autonomous_reasoning_ai.config import config

# Imports des modules du projet
from autonomous_reasoning_ai.models import AutonomousReasoningCore
from autonomous_reasoning_ai.problem_generators import (
    ProblemGenerator, MathProblemGenerator, 
    LogicProblemGenerator, PatternProblemGenerator
)
from autonomous_reasoning_ai.training import AutonomousLearningTrainer
from autonomous_reasoning_ai.experiments import ExperimentRunner

def setup_environment():
    """Configure l'environnement d'exécution"""
    # Seed pour la reproductibilité
    torch.manual_seed(config.get('project.random_seed'))
    
    # Création des répertoires nécessaires
    experiment_paths = config.get('experiment.paths', {})
    for key, directory in experiment_paths.items():
        os.makedirs(directory, exist_ok=True)
    
    # Configuration du device
    device = config.get('model.device')
    logger.info(f"Utilisation du device: {device}")
    
    return device

def initialize_components():
    """Initialise tous les composants du système"""
    logger.info("Initialisation des composants...")
    
    # Création du générateur de problèmes
    problem_generator = ProblemGenerator()
    
    # Chargement conditionnel des générateurs suivant config
    if config.get('problem_generators.math.enabled', True):
        problem_generator.register_generator("math", MathProblemGenerator())
        logger.info("✅ Générateur mathématique activé")
    
    if config.get('problem_generators.logic.enabled', False):
        try:
            problem_generator.register_generator("logic", LogicProblemGenerator())
            logger.info("✅ Générateur logique activé")
        except Exception as e:
            logger.warning(f"⚠️  Générateur logique désactivé: {e}")
    
    if config.get('problem_generators.patterns.enabled', False):
        try:
            problem_generator.register_generator("pattern", PatternProblemGenerator())
            logger.info("✅ Générateur patterns activé")
        except Exception as e:
            logger.warning(f"⚠️  Générateur patterns désactivé: {e}")
    
    # Création du modèle
    model = AutonomousReasoningCore()
    model = model.to(config.get('model.device'))
    
    # Compilation du modèle si supportée
    if config.get('project.compile_model', False):
        try:
            model = torch.compile(model)
            logger.info("Modèle compilé avec torch.compile")
        except Exception as e:
            logger.warning(f"Impossible de compiler le modèle: {e}")
    
    # Création du trainer
    trainer = AutonomousLearningTrainer(model, problem_generator)
    
    return model, problem_generator, trainer

def run_quick_test():
    """Lance un test rapide du système"""
    logger.info("🧪 Lancement du test rapide...")
    
    device = setup_environment()
    model, problem_generator, trainer = initialize_components()
    
    try:
        # Test de génération de problèmes
        available_generators = list(problem_generator.generators.keys())
        logger.info(f"Générateurs disponibles: {available_generators}")
        
        for gen_type in available_generators:
            problems = problem_generator.generate_batch(gen_type, complexity=1, batch_size=3)
            logger.info(f"✅ {gen_type}: {len(problems)} problèmes générés")
        
        # Test du modèle avec le premier générateur disponible
        if available_generators:
            logger.info("Test du modèle de raisonnement...")
            with torch.no_grad():
                input_data = problems[0].input_data.unsqueeze(0).to(model.device)
                result = model(input_data)
            
            logger.info(f"✅ Test du modèle réussi:")
            logger.info(f"   - Profondeur de raisonnement: {result['reasoning_depth']}")
            logger.info(f"   - Confiance finale: {result['final_confidence'].item():.3f}")
            logger.info(f"   - Score de consistance: {result['consistency_score'].item():.3f}")
            
            # Test d'entraînement
            logger.info("Test d'une étape d'entraînement...")
            training_results = trainer.self_supervised_training_step(problems[:3])
            logger.info(f"✅ Entraînement testé - Récompense: {training_results['average_reward']:.4f}")
        
        logger.info("🎉 Test rapide terminé avec succès!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_training():
    """Lance l'entraînement complet"""
    logger.info("🚀 Démarrage de l'entraînement complet...")
    
    device = setup_environment()
    model, problem_generator, trainer = initialize_components()
    
    # Configuration de l'expérimentation
    experiment = ExperimentRunner(
        model=model,
        trainer=trainer,
        problem_generator=problem_generator,
        config_obj=config
    )
    
    # Lancement de l'expérience
    experiment.run()

def main():
    """Fonction principale avec intégration config YAML"""
    print("🤖 IA de Raisonnement Autonome")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Projet: {config.get('project.name')} v{config.get('project.version')}")
    print(f"  - Device: {config.get('model.device')}")
    print(f"  - Dimensions: {config.get('model.input_dim')}x{config.get('model.hidden_dim')}")
    print(f"  - Learning rate: {config.get('training.learning_rate')}")
    print(f"  - Batch size: {config.get('training.batch_size')}")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Mode d'exécution (test/train): ").strip().lower()
    
    if mode == "test":
        success = run_quick_test()
        if not success:
            sys.exit(1)
    elif mode == "train":
        run_full_training()
    else:
        logger.error("Mode non reconnu. Utilisez 'test' ou 'train'")
        sys.exit(1)

if __name__ == "__main__":
    main()
