"""
Experiment Runner SIMPLIFIÉ
============================

Principe : Entraînement progressif avec un algorithme clair
"""
import logging
import time
from typing import Dict, List
from ..config import config
from ..data.problem_generator import ProblemGenerator
from ..training.simple_trainer import create_simple_trainer
from ..models.autonomous_reasoning_model import AutonomousReasoningModel
import torch

logger = logging.getLogger(__name__)

class SimpleExperimentRunner:
    """
    Runner d'expériences avec algorithme d'apprentissage simplifié
    
    FLOW :
    1. Commence au niveau 1
    2. Génère des problèmes simples
    3. Entraîne avec l'algorithme simple
    4. Mesure les progrès CLAIREMENT
    5. Augmente la complexité si critères atteints
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modèle
        self.model = AutonomousReasoningModel().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        # Optimizer simple
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Trainer simple
        self.trainer = create_simple_trainer(self.model, self.optimizer, self.device)
        
        # Problem generator
        self.problem_generator = ProblemGenerator()
        
        # État de l'expérience
        self.current_complexity = 1
        self.current_domain = 'math'
        self.total_steps = 0
        
        # Historique
        self.complexity_history = []
        self.performance_history = []
        
        logger.info("✅ SimpleExperimentRunner initialisé")
    
    def run_experiment(self, max_steps: int = 1000, save_interval: int = 100):
        """
        Lance l'expérience d'entraînement
        
        Args:
            max_steps: Nombre maximum d'étapes
            save_interval: Intervalle de sauvegarde
        """
        logger.info(f"🚀 Début de l'expérience - {max_steps} steps max")
        
        start_time = time.time()
        last_complexity_change = 0
        
        while self.total_steps < max_steps:
            # 1. Génère un batch de problèmes au niveau actuel
            problems = self.problem_generator.generate_batch(
                domain=self.current_domain,
                complexity=self.current_complexity,
                batch_size=config.training.batch_size
            )
            
            # 2. Entraîne sur ce batch
            batch_result = self.trainer.train_batch(problems)
            self.total_steps += batch_result['count']
            
            # 3. Log des progrès
            if self.total_steps % 10 == 0:  # Log toutes les 10 étapes
                self._log_progress(batch_result)
            
            # 4. Vérifie s'il faut augmenter la complexité
            if self.total_steps - last_complexity_change > 50:  # Au moins 50 steps au même niveau
                if self.trainer.should_increase_complexity():
                    self._increase_complexity()
                    last_complexity_change = self.total_steps
            
            # 5. Sauvegarde périodique
            if self.total_steps % save_interval == 0:
                self._save_checkpoint()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Expérience terminée en {elapsed_time:.1f}s")
        
        return self._get_final_results()
    
    def _log_progress(self, batch_result: Dict):
        """Log les progrès de manière claire"""
        metrics = self.trainer.get_progress_metrics()
        
        logger.info(
            f"Step {self.total_steps}: "
            f"Loss={batch_result['loss']:.4f}, "
            f"Acc={batch_result['accuracy']:.2%}, "
            f"Level={self.current_complexity}, "
            f"Trend={metrics['trend']}"
        )
        
        # Historique
        self.performance_history.append({
            'step': self.total_steps,
            'loss': batch_result['loss'],
            'accuracy': batch_result['accuracy'],
            'complexity': self.current_complexity,
            'trend': metrics['trend']
        })
    
    def _increase_complexity(self):
        """Augmente la complexité de manière contrôlée"""
        old_complexity = self.current_complexity
        
        # Augmentation progressive
        if self.current_complexity < 5:
            self.current_complexity += 1
            self.trainer.reset_metrics()  # Reset pour le nouveau niveau
            
            logger.info(f"📈 Complexité augmentée : {old_complexity} → {self.current_complexity}")
            
            # Historique
            self.complexity_history.append({
                'step': self.total_steps,
                'old_level': old_complexity,
                'new_level': self.current_complexity
            })
        else:
            logger.info(f"🎯 Complexité maximale atteinte ({self.current_complexity})")
    
    def _save_checkpoint(self):
        """Sauvegarde l'état actuel"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': self.total_steps,
            'complexity': self.current_complexity,
            'performance_history': self.performance_history
        }
        
        checkpoint_path = f"checkpoint_step_{self.total_steps}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint sauvé : {checkpoint_path}")
    
    def _get_final_results(self) -> Dict:
        """Résultats finaux de l'expérience"""
        final_metrics = self.trainer.get_progress_metrics()
        
        return {
            'total_steps': self.total_steps,
            'final_complexity': self.current_complexity,
            'final_loss': final_metrics['current_loss'],
            'final_accuracy': final_metrics['current_accuracy'],
            'complexity_progressions': len(self.complexity_history),
            'performance_history': self.performance_history,
            'complexity_history': self.complexity_history,
            'success': final_metrics['current_accuracy'] > 0.5 and final_metrics['current_loss'] < 0.5
        }
    
    def get_status(self) -> Dict:
        """État actuel de l'expérience"""
        metrics = self.trainer.get_progress_metrics()
        
        return {
            'step': self.total_steps,
            'complexity': self.current_complexity,
            'domain': self.current_domain,
            'current_loss': metrics['current_loss'],
            'current_accuracy': metrics['current_accuracy'],
            'trend': metrics['trend'],
            'ready_for_next_level': self.trainer.should_increase_complexity()
        }

def run_simple_experiment(max_steps: int = 1000) -> Dict:
    """
    Fonction simple pour lancer une expérience
    """
    runner = SimpleExperimentRunner()
    return runner.run_experiment(max_steps)

if __name__ == "__main__":
    # Test rapide
    results = run_simple_experiment(100)
    print("Résultats:", results)
