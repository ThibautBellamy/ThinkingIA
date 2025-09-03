"""
Système d'apprentissage autonome
"""

from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict
import logging

from ..problem_generators import ProblemGenerator, Problem
from ..config import config

logger = logging.getLogger(__name__)

class AutonomousLearningTrainer:
    """Système d'apprentissage autonome"""
    
    def __init__(self, model, problem_generator: ProblemGenerator, config):
        self.model = model
        self.problem_generator = problem_generator
        self.success_history = []
        
        self.config = config  # ✅ Configuration YAML
         # ✅ PROTECTION : Vérifier si model existe avant d'initialiser l'optimiseur
        if model is not None:
            # Paramètres depuis la config YAML
            self.learning_rate = config.get('training.learning_rate')
            self.batch_size = config.get('training.batch_size')
            self.max_epochs = config.get('training.max_epochs')
            
            # Configuration de l'optimiseur depuis la config
            optimizer_name = config.get('training.optimizer.name', 'adamw')
            weight_decay = config.get('training.optimizer.weight_decay', 0.01)
            
            if optimizer_name.lower() == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_name.lower() == 'adam':
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.learning_rate
                )

            # Scheduler d'apprentissage - Anti-oscillation
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.8,    # Réduction douce
                patience=10,   # Plus de patience
                min_lr=1e-7,   # LR très bas
                threshold=0.001  # Seuil d'amélioration petit
            )
        else:
            # ✅ MODE TEST : Initialisation minimale quand model=None
            print("⚠️  Mode test détecté - model=None, initialisation minimale")
            self.optimizer = None
            self.scheduler = None
            self.learning_rate = config.get('training.learning_rate')
            self.batch_size = config.get('training.batch_size')
            self.max_epochs = config.get('training.max_epochs')
        
        # Historique des métriques
        self.metrics_history = {
            'loss': [],
            'reward': [],
            'consistency': [],
            'confidence': [],
            'reasoning_depth': []
        }
        
        # Configuration pour mixed precision
        if getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
    def entropy_based_reward(self, solutions: List[torch.Tensor]) -> float:
        """Calcule la récompense basée sur la diversité et qualité des solutions"""
        if len(solutions) < 2:
            return 0.1  # Récompense minimale
        
        try:
            # Mesure la variance entre solutions multiples
            solutions_tensor = torch.stack([s.flatten() for s in solutions])
            
            # Récompense de diversité (variance normalisée)
            variance = torch.var(solutions_tensor, dim=0).mean()
            diversity_reward = min(variance.item() * 2.0, 0.8)  # Cap à 0.8
            
            # Récompense de magnitude (éviter solutions nulles)
            magnitude = torch.abs(solutions_tensor).mean()
            magnitude_reward = min(magnitude.item() * 5.0, 0.5)  # Cap à 0.5
            
            # Récompense de stabilité (éviter divergence)
            stability = 1.0 / (1.0 + torch.std(solutions_tensor).item())
            stability_reward = min(stability * 0.3, 0.3)  # Cap à 0.3
            
            # Combinaison des récompenses
            total_reward = diversity_reward + magnitude_reward + stability_reward
            
            # Assurer un minimum et maximum
            return max(0.05, min(total_reward, 1.0))
            
        except Exception as e:
            return 0.1  # Récompense de sécurité
    
    def compute_loss_with_rewards(self, results: List[Dict], problems: List[Problem], solutions: List[torch.Tensor]) -> tuple:
        """Calcule la loss combinée avec les récompenses intégrées"""
        device = next(self.model.parameters()).device
        
        # Loss de base (comme avant)
        base_loss = self.compute_loss(results, problems, self.config)
        
        # Calcul des récompenses
        consistency_reward = self.entropy_based_reward(solutions)
        confidence_reward = sum(r['final_confidence'].mean().item() for r in results) / len(results)
        
        # Validation externe si disponible
        validation_reward = 0
        validation_count = 0
        for problem, result in zip(problems, results):
            if problem.validation_fn:
                try:
                    # Passer le tenseur complet à la fonction de validation
                    predicted_solution = result['solution']
                    if problem.validation_fn(predicted_solution):
                        validation_reward += 1.0
                    else:
                        validation_reward += 0.1  # Récompense partielle pour tentative
                    validation_count += 1
                except Exception as e:
                    # En cas d'erreur, donner une petite récompense
                    validation_reward += 0.05
                    validation_count += 1
        
        if validation_count > 0:
            validation_reward /= validation_count
        else:
            validation_reward = 0.2  # Récompense de base quand pas de validation
        
        # Calcul total avec minimum garanti - ✅ CORRIGÉ
        consistency_weight = self.config.get('training.self_supervision.consistency_weight', 0.5)
        confidence_weight = self.config.get('training.self_supervision.confidence_weight', 0.3)
        validation_weight = self.config.get('training.self_supervision.validation_weight', 0.2)
        
        total_reward = max(0.05, (
            consistency_weight * consistency_reward +
            confidence_weight * max(confidence_reward, 0.1) +
            validation_weight * validation_reward
        ))
        
        # **INNOVATION** : Modifier la loss avec les récompenses
        # Plus la récompense est élevée, plus on réduit la loss
        reward_bonus = torch.tensor(total_reward, device=device, dtype=base_loss.dtype)
        
        # Loss finale = loss de base * (2.0 - récompense) 
        # Si reward=1.0 -> loss *= 1.0 (réduction max)
        # Si reward=0.0 -> loss *= 2.0 (pénalité max)
        modified_loss = base_loss * (2.0 - reward_bonus)
        
        # Métriques pour le logging
        reward_metrics = {
            'consistency_reward': consistency_reward,
            'confidence_reward': confidence_reward,
            'validation_reward': validation_reward,
            'total_reward': total_reward
        }
        
        return modified_loss, reward_metrics
    
    def compute_loss(self, results: List[Dict], problems: List[Problem], config) -> torch.Tensor:
        """Calcule la loss combinée STABLE pour l'auto-supervision"""
        device = next(self.model.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ✅ CORRIGÉ - Utilisation de config.get()
        consistency_weight = config.get('training.self_supervision.consistency_weight', 0.5)
        confidence_weight = config.get('training.self_supervision.confidence_weight', 0.3)
        validation_weight = config.get('training.self_supervision.validation_weight', 0.2)
        
        for result, problem in zip(results, problems):
            # Loss de consistance STABLE (MSE vers target fixe)
            consistency_target = torch.tensor(0.9, device=device)  # Cible stable
            consistency_loss = F.mse_loss(result['consistency_score'], consistency_target)
            
            # Loss de confiance STABLE (MSE vers target fixe)
            confidence_target = torch.tensor(0.8, device=device)  # Cible stable
            confidence_loss = F.mse_loss(result['final_confidence'].mean(), confidence_target)
            
            # Loss de validation PROGRESSIVE (pas binaire)
            validation_loss = torch.tensor(0.5, device=device)  # Neutre par défaut
            if problem.validation_fn:
                predicted_solution = result['solution'].mean()
                if problem.validation_fn(predicted_solution):
                    # Récompense progressive basée sur la confiance
                    validation_loss = torch.tensor(0.0, device=device) * (1.0 - result['final_confidence'].mean())
                else:
                    # Pénalité douce, pas brutale
                    validation_loss = torch.tensor(0.3, device=device)
            
            # Combinaison STABLE des losses
            combined_loss = (
                consistency_weight * consistency_loss +
                confidence_weight * confidence_loss +
                validation_weight * validation_loss
            )
            
            total_loss = total_loss + combined_loss
        
        return total_loss / len(results)
    
    def self_supervised_training_step(self, problems: List[Problem]) -> Dict:
        """Une étape d'entraînement auto-supervisé"""
        self.model.train()
        
        # Forward pass
        results = []
        solutions = []
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                for problem in problems:
                    # Génère plusieurs solutions pour le même problème
                    problem_solutions = []
                    problem_results = []
                    
                    for attempt in range(2):  # Réduit de 3 à 2 tentatives par problème
                        input_batch = problem.input_data.unsqueeze(0).to(self.model.device)
                        result = self.model(input_batch)
                        
                        problem_solutions.append(result['solution'])
                        problem_results.append(result)
                    
                    # Prendre le meilleur résultat
                    best_idx = max(range(len(problem_results)), 
                                   key=lambda i: problem_results[i]['final_confidence'].mean())
                    
                    results.append(problem_results[best_idx])
                    solutions.extend(problem_solutions)
                
                # Calcul de la loss avec récompenses intégrées
                loss, reward_metrics = self.compute_loss_with_rewards(results, problems, solutions)
        else:
            for problem in problems:
                # Génère plusieurs solutions pour le même problème (réduit à 2)
                problem_solutions = []
                problem_results = []
                
                for attempt in range(2):  # Réduit de 3 à 2 tentatives par problème
                    input_batch = problem.input_data.unsqueeze(0).to(self.model.device)
                    result = self.model(input_batch)
                    
                    problem_solutions.append(result['solution'])
                    problem_results.append(result)
                
                # Prendre le meilleur résultat
                best_idx = max(range(len(problem_results)), 
                               key=lambda i: problem_results[i]['final_confidence'].mean())
                
                results.append(problem_results[best_idx])
                solutions.extend(problem_solutions)
            
            # Calcul de la loss avec récompenses intégrées
            loss, reward_metrics = self.compute_loss_with_rewards(results, problems, solutions)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping plus agressif pour éviter les oscillations
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # Récupération des métriques depuis reward_metrics
        consistency_reward = reward_metrics['consistency_reward']
        confidence_reward = reward_metrics['confidence_reward']
        validation_reward = reward_metrics['validation_reward']
        total_reward = reward_metrics['total_reward']
        
        # Mise à jour de l'historique
        self.metrics_history['loss'].append(loss.item())
        self.metrics_history['reward'].append(total_reward)
        self.metrics_history['consistency'].append(consistency_reward)
        self.metrics_history['confidence'].append(confidence_reward)
        self.metrics_history['reasoning_depth'].append(
            sum(r['reasoning_depth'] for r in results) / len(results)
        )
        
        return {
            'loss': loss.item(),
            'total_reward': total_reward,
            'average_reward': total_reward,
            'consistency_score': consistency_reward,
            'confidence_score': confidence_reward,
            'validation_score': validation_reward,
            'average_reasoning_depth': self.metrics_history['reasoning_depth'][-1]
        }
    
    def get_model_statistics(self):
        """Retourne les statistiques des poids du modèle pour TensorBoard"""
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[f"weights/{name}"] = param.data
                if param.grad is not None:
                    stats[f"gradients/{name}"] = param.grad.data
        
        return stats
    
    def train_epoch(self, domain: str, complexity: int, num_batches: int = 10) -> Dict:
        """Entraîne une époque complète pour un domaine et complexité donnés"""
        epoch_metrics = {
            'loss': 0,
            'total_reward': 0,
            'consistency_score': 0,
            'confidence_score': 0,
            'reasoning_depth': 0
        }
        
        for batch_idx in range(num_batches):
            # ✅ CORRIGÉ - Utilisation de self.config.get()
            problems = self.problem_generator.generate_batch(
                domain, complexity, self.config.get('training.batch_size')
            )
            
            # Entraînement sur ce batch
            batch_results = self.self_supervised_training_step(problems)
            
            # Accumulation des métriques
            for key in epoch_metrics:
                if key in batch_results:
                    epoch_metrics[key] += batch_results[key]
                elif key == 'reasoning_depth':
                    epoch_metrics[key] += batch_results['average_reasoning_depth']
        
        # Moyenne des métriques
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Ajouter la clé 'reward' pour compatibilité
        epoch_metrics['reward'] = epoch_metrics['total_reward']
        
        # Mise à jour du scheduler
        self.scheduler.step(epoch_metrics['total_reward'])
        
        logger.info(f"Époque terminée - Loss: {epoch_metrics['loss']:.4f}, "
                  f"Reward: {epoch_metrics['total_reward']:.4f}")
        
        return epoch_metrics
    
    def should_increase_complexity(self, recent_performance: List[float]) -> bool:
        """Détermine si la complexité doit être augmentée"""
        if len(recent_performance) < 5:
            return False
        
        # ✅ CORRIGÉ - Utilisation de self.config.get()
        avg_performance = sum(recent_performance[-5:]) / 5
        threshold = self.config.get('training.curriculum.complexity_threshold', 0.75)
        return avg_performance > threshold
    
    def get_training_summary(self) -> Dict:
        """Retourne un résumé de l'entraînement"""
        if not self.metrics_history['reward']:
            return {"status": "No training performed yet"}
        
        return {
            "total_steps": len(self.metrics_history['reward']),
            "current_loss": self.metrics_history['loss'][-1],
            "current_reward": self.metrics_history['reward'][-1],
            "best_reward": max(self.metrics_history['reward']),
            "average_consistency": sum(self.metrics_history['consistency']) / len(self.metrics_history['consistency']),
            "average_reasoning_depth": sum(self.metrics_history['reasoning_depth']) / len(self.metrics_history['reasoning_depth']),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def reset_success_history(self):
        """Réinitialise l'historique des succès pour un nouveau niveau de complexité"""
        self.success_history = []
        # NE PAS réinitialiser metrics_history pour garder la continuité TensorBoard
        # On garde l'historique complet pour visualisation
        logger.info("📊 Historique des succès réinitialisé (métriques préservées pour TensorBoard)")
