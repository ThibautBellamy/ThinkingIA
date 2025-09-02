"""
Algorithme d'apprentissage SIMPLIFIÉ et EFFICACE
==================================================

Principe : Un apprentissage par étapes claires et mesurables
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional
from ..config import config
from ..models.problem import Problem

logger = logging.getLogger(__name__)

class SimpleTrainer:
    """
    Trainer simplifié avec un algorithme d'apprentissage clair
    
    PRINCIPE :
    1. Génération d'UN problème à la fois
    2. UNE solution par problème
    3. Loss SIMPLE : CrossEntropy ou MSE
    4. Progression MESURABLE
    """
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Métriques simples
        self.step_count = 0
        self.loss_history = []
        self.accuracy_history = []
        
        # Loss function simple
        self.loss_fn = nn.MSELoss()
        
        logger.info("✅ SimpleTrainer initialisé")
    
    def train_step(self, problem: Problem) -> Dict:
        """
        UNE étape d'entraînement = UN problème = UNE solution
        
        Args:
            problem: Un seul problème à résoudre
            
        Returns:
            Dict avec loss, accuracy, solution
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Forward pass : UNE solution
        input_batch = problem.input_data.unsqueeze(0).to(self.device)
        result = self.model(input_batch)
        solution = result['solution'].squeeze(0)
        
        # 2. Calcul de la loss SIMPLE
        if hasattr(problem, 'target_solution'):
            # Supervised learning : on a la vraie réponse
            target = problem.target_solution.to(self.device)
            loss = self.loss_fn(solution, target)
        else:
            # Self-supervised : on évalue la cohérence
            consistency_score = result.get('consistency_score', 0.5)
            # Plus c'est cohérent, moins on pénalise
            loss = torch.tensor(1.0 - consistency_score, device=self.device, requires_grad=True)
        
        # 3. Backward pass
        loss.backward()
        self.optimizer.step()
        
        # 4. Calcul de l'accuracy
        accuracy = self._calculate_accuracy(problem, solution)
        
        # 5. Historique
        self.step_count += 1
        self.loss_history.append(loss.item())
        self.accuracy_history.append(accuracy)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'solution': solution.detach().cpu(),
            'step': self.step_count
        }
    
    def _calculate_accuracy(self, problem: Problem, solution: torch.Tensor) -> float:
        """
        Calcule l'accuracy de manière simple
        """
        if problem.validation_fn:
            # Test avec la fonction de validation
            try:
                predicted_value = solution.mean().item()
                is_correct = problem.validation_fn(predicted_value)
                return 1.0 if is_correct else 0.0
            except Exception:
                return 0.0
        else:
            # Accuracy basée sur la confiance du modèle
            confidence = solution.std().item()  # Plus stable = plus confiant
            return min(1.0, 1.0 / (1.0 + confidence))
    
    def train_batch(self, problems: List[Problem]) -> Dict:
        """
        Entraîne sur un batch de problèmes
        """
        batch_loss = 0.0
        batch_accuracy = 0.0
        successful_steps = 0
        
        for problem in problems:
            try:
                step_result = self.train_step(problem)
                batch_loss += step_result['loss']
                batch_accuracy += step_result['accuracy']
                successful_steps += 1
            except Exception as e:
                logger.warning(f"Erreur sur un problème : {e}")
                continue
        
        if successful_steps == 0:
            return {'loss': 1.0, 'accuracy': 0.0, 'count': 0}
        
        return {
            'loss': batch_loss / successful_steps,
            'accuracy': batch_accuracy / successful_steps,
            'count': successful_steps
        }
    
    def get_progress_metrics(self) -> Dict:
        """
        Métriques de progression CLAIRES
        """
        if len(self.loss_history) < 10:
            return {
                'trend': 'insufficient_data',
                'current_loss': self.loss_history[-1] if self.loss_history else 1.0,
                'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0
            }
        
        # Tendance des 10 derniers steps
        recent_loss = self.loss_history[-10:]
        recent_accuracy = self.accuracy_history[-10:]
        
        # Est-ce que ça s'améliore ?
        loss_improving = recent_loss[0] > recent_loss[-1]  # Loss diminue ?
        accuracy_improving = recent_accuracy[0] < recent_accuracy[-1]  # Accuracy augmente ?
        
        trend = 'improving' if (loss_improving and accuracy_improving) else 'stagnating'
        
        return {
            'trend': trend,
            'current_loss': recent_loss[-1],
            'current_accuracy': recent_accuracy[-1],
            'loss_change': recent_loss[0] - recent_loss[-1],
            'accuracy_change': recent_accuracy[-1] - recent_accuracy[0],
            'steps_trained': self.step_count
        }
    
    def should_increase_complexity(self) -> bool:
        """
        Critère SIMPLE pour augmenter la complexité
        """
        metrics = self.get_progress_metrics()
        
        # Conditions :
        # 1. Accuracy > 70%
        # 2. Loss < 0.3
        # 3. Tendance d'amélioration ou stable
        
        return (
            metrics['current_accuracy'] > 0.7 and
            metrics['current_loss'] < 0.3 and
            metrics['trend'] in ['improving', 'stable']
        )
    
    def reset_metrics(self):
        """Remet à zéro les métriques pour un nouveau niveau"""
        self.loss_history = []
        self.accuracy_history = []
        logger.info("📊 Métriques remises à zéro pour nouveau niveau")

def create_simple_trainer(model, optimizer, device):
    """Factory function pour créer un SimpleTrainer"""
    return SimpleTrainer(model, optimizer, device)
