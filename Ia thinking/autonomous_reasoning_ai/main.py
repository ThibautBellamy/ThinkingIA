#!/usr/bin/env python3
"""
Point d'entrée principal pour l'IA de raisonnement autonome
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

# Imports des modules du projet
from autonomous_reasoning_ai.config import config
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
    torch.manual_seed(config.random_seed)
    
    # Création des répertoires nécessaires
    # Note: checkpoint_dir est maintenant géré par experiment_runner
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.results_dir, exist_ok=True)
    
    # Configuration du device
    device = torch.device(config.model.device)
    logger.info(f"Utilisation du device: {device}")
    
    return device

def initialize_components():
    """Initialise tous les composants du système"""
    logger.info("Initialisation des composants...")
    
    # Création du générateur de problèmes
    problem_generator = ProblemGenerator()
    problem_generator.register_generator("math", MathProblemGenerator())
    # problem_generator.register_generator("logic", LogicProblemGenerator())
    # problem_generator.register_generator("pattern", PatternProblemGenerator())
    
    # Création du modèle
    model = AutonomousReasoningCore()
    model = model.to(config.model.device)
    
    # Compilation du modèle si supportée
    if config.compile_model:
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
        logger.info("Test de génération de problèmes...")
        problems = problem_generator.generate_batch("math", complexity=1, batch_size=5)
        logger.info(f"✅ {len(problems)} problèmes générés")
        
        # Test du modèle
        logger.info("Test du modèle de raisonnement...")
        with torch.no_grad():
            # Transférer les données sur le bon device
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
    """Fonction principale"""
    print("🤖 IA de Raisonnement Autonome")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Device: {config.model.device}")
    print(f"  - Dimensions: {config.model.input_dim}x{config.model.hidden_dim}")
    print(f"  - Complexité max: {config.training.max_complexity}")
    print("=" * 50)
    
    # Choix du mode d'exécution
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
