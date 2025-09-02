"""
Version modifiée de main.py pour charger automatiquement le modèle entraîné le plus récent
"""

import os
import torch
import logging
import shutil
import glob
from datetime import datetime
from autonomous_reasoning_ai.main import *
from autonomous_reasoning_ai.config import config
from autonomous_reasoning_ai.training.autonomous_trainer import AutonomousLearningTrainer

# Chemin fixe pour le modèle actuel
CURRENT_MODEL_PATH = "current_best_model.pt"

def find_latest_model():
    """Trouve le modèle entraîné le plus récent dans les dossiers de résultats"""
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        return None
    
    # Chercher tous les fichiers best_model.pt dans les sous-dossiers
    model_files = glob.glob(os.path.join(results_dir, "*", "best_model.pt"))
    
    if not model_files:
        return None
    
    # Trier par date de modification (le plus récent en dernier)
    model_files.sort(key=lambda x: os.path.getmtime(x))
    
    return model_files[-1]  # Le plus récent

def copy_latest_model_to_current():
    """Copie le modèle le plus récent vers le chemin fixe"""
    logger = logging.getLogger(__name__)
    
    latest_model = find_latest_model()
    
    if latest_model is None:
        logger.warning("❌ Aucun modèle entraîné trouvé dans le dossier results/")
        return False
    
    try:
        # Copier le modèle vers le chemin fixe
        shutil.copy2(latest_model, CURRENT_MODEL_PATH)
        
        # Obtenir les infos du modèle source
        model_dir = os.path.dirname(latest_model)
        experiment_name = os.path.basename(model_dir)
        modification_time = datetime.fromtimestamp(os.path.getmtime(latest_model))
        
        logger.info(f"✅ Modèle copié avec succès!")
        logger.info(f"📁 Source: {latest_model}")
        logger.info(f"🎯 Destination: {CURRENT_MODEL_PATH}")
        logger.info(f"🧪 Expérience: {experiment_name}")
        logger.info(f"📅 Dernière modification: {modification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la copie du modèle: {e}")
        return False

def initialize_components_with_latest_model():
    """Initialise tous les composants du système avec le modèle entraîné le plus récent"""
    logger = logging.getLogger(__name__)
    logger.info("Initialisation des composants avec modèle le plus récent...")
    
    # Étape 1: Copier le modèle le plus récent
    logger.info("🔍 Recherche du modèle le plus récent...")
    model_copied = copy_latest_model_to_current()
    
    # Création du générateur de problèmes
    problem_generator = ProblemGenerator()
    problem_generator.register_generator("math", MathProblemGenerator())
    
    # Création du modèle
    model = AutonomousReasoningCore()
    model = model.to(config.model.device)
    
    # Chargement du modèle entraîné
    if model_copied and os.path.exists(CURRENT_MODEL_PATH):
        logger.info(f"📂 Chargement du modèle depuis: {CURRENT_MODEL_PATH}")
        try:
            checkpoint = torch.load(CURRENT_MODEL_PATH, map_location=config.model.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint complet avec métadonnées
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Modèle chargé avec succès (checkpoint complet)!")
                
                # Afficher les métadonnées si disponibles
                if 'epoch' in checkpoint:
                    logger.info(f"📊 Époque: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    logger.info(f"📉 Loss: {checkpoint['loss']:.6f}")
                if 'accuracy' in checkpoint:
                    logger.info(f"🎯 Accuracy: {checkpoint['accuracy']:.4f}")
                if 'reward' in checkpoint:
                    logger.info(f"🏆 Reward: {checkpoint['reward']:.6f}")
            else:
                # Seulement les poids du modèle
                model.load_state_dict(checkpoint)
                logger.info("✅ Poids du modèle chargés avec succès!")
                
            # Passer en mode évaluation
            model.eval()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            logger.warning("🔄 Utilisation du modèle non entraîné")
    else:
        logger.warning(f"❌ Aucun modèle trouvé")
        logger.warning("🔄 Utilisation du modèle non entraîné")
    
    # Création du trainer (pour compatibilité)
    trainer = AutonomousLearningTrainer(model, problem_generator)
    
    return model, problem_generator, trainer

# Remplacer la fonction d'origine pour les tests
initialize_components = initialize_components_with_latest_model
