"""
Runner d'expérimentations pour l'IA de raisonnement autonome
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from ..config import config
from ..utils.validation import ValidationUtils

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Gestionnaire d'expérimentations pour l'IA de raisonnement autonome"""
    
    def __init__(self, model, trainer, problem_generator, config_obj=None):
        self.model = model
        self.trainer = trainer
        self.problem_generator = problem_generator
        self.config = config_obj or config
        
        # Configuration de l'expérimentation
        self.experiment_name = f"{self.config.experiment.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join(self.config.experiment.results_dir, self.experiment_name)
        
        # Création des répertoires
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "visualizations"), exist_ok=True)
        
        # TensorBoard
        if self.config.experiment.tensorboard_enabled:
            self.tb_writer = SummaryWriter(
                os.path.join(self.config.experiment.log_dir, self.experiment_name)
            )
        else:
            self.tb_writer = None
        
        # Historique de l'expérimentation
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'epochs': [],
            'best_performance': {},
            'complexity_progression': []
        }
        
        # Sauvegarde de la configuration
        self._save_experiment_config()
    
    def _save_experiment_config(self):
        """Sauvegarde la configuration de l'expérimentation"""
        config_path = os.path.join(self.experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _log_to_tensorboard(self, metrics: Dict, step: int, prefix: str = ""):
        """Log des métriques vers TensorBoard"""
        if not self.tb_writer:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tag = f"{prefix}/{key}" if prefix else key
                self.tb_writer.add_scalar(tag, value, step)
    
    def _log_model_histograms(self, step: int):
        """Log des histogrammes des poids et gradients"""
        if not self.tb_writer:
            return
        
        # Obtenir les statistiques du modèle
        model_stats = self.trainer.get_model_statistics()
        
        for name, tensor in model_stats.items():
            if tensor is not None and tensor.numel() > 0:
                self.tb_writer.add_histogram(name, tensor, step)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Sauvegarde un checkpoint"""
        checkpoint_path = os.path.join(
            self.experiment_dir, "checkpoints", f"epoch_{epoch}.pt"
        )
        
        ValidationUtils.save_model_checkpoint(
            self.model, self.trainer.optimizer, metrics, checkpoint_path
        )
        
        # Sauvegarde du meilleur modèle
        if not self.experiment_log['best_performance'] or \
           metrics.get('total_reward', 0) > self.experiment_log['best_performance'].get('total_reward', 0):
            
            best_path = os.path.join(self.experiment_dir, "best_model.pt")
            ValidationUtils.save_model_checkpoint(
                self.model, self.trainer.optimizer, metrics, best_path
            )
            
            self.experiment_log['best_performance'] = metrics.copy()
            self.experiment_log['best_performance']['epoch'] = epoch
    
    def _evaluate_model(self, complexity: int, domain: str) -> Dict:
        """Évalue le modèle sur un ensemble de test"""
        test_problems = self.problem_generator.generate_batch(
            domain, complexity, batch_size=20  # Réduit de 50 à 20
        )
        
        return ValidationUtils.evaluate_model_on_problems(
            self.model, test_problems, self.model.device
        )
    
    def _should_increase_complexity(self, complexity_history: List[Dict]) -> bool:
        """Détermine si la complexité doit être augmentée"""
        if len(complexity_history) < 3:
            return False
        
        # Vérifier les 3 dernières performances
        recent_rewards = [h['total_reward'] for h in complexity_history[-3:]]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        return avg_reward > self.config.training.complexity_threshold
    
    def run_curriculum_learning(self):
        """Execute le curriculum learning avec complexité progressive"""
        logger.info("🎓 Démarrage du curriculum learning")
        
        current_complexity = self.config.training.initial_complexity
        max_complexity = self.config.training.max_complexity
        
        complexity_history = []
        global_step = 0
        steps_since_complexity_change = 0  # Compteur pour l'adaptation
        
        while current_complexity <= max_complexity:
            logger.info(f"📊 Niveau de complexité: {current_complexity}")
            
            # Entraînement sur cette complexité
            for domain in self.problem_generator.get_available_domains():
                logger.info(f"🔬 Entraînement sur le domaine: {domain}")
                
                domain_history = []
                epochs_at_complexity = 0
                max_epochs_per_complexity = 20
                
                while epochs_at_complexity < max_epochs_per_complexity:
                    # Entraînement d'une époque (réduit à 5 batches)
                    epoch_metrics = self.trainer.train_epoch(
                        domain, current_complexity, num_batches=5
                    )
                    
                    # Évaluation
                    eval_metrics = self._evaluate_model(current_complexity, domain)
                    
                    # Combinaison des métriques (eval_metrics ne doit pas écraser reward!)
                    combined_metrics = {**eval_metrics, **epoch_metrics}  # epoch_metrics en dernier !
                    combined_metrics['complexity'] = current_complexity
                    combined_metrics['domain'] = domain
                    combined_metrics['global_step'] = global_step
                    
                    # Logging
                    self._log_to_tensorboard(combined_metrics, global_step, f"complexity_{current_complexity}")
                    
                    # Log des histogrammes tous les 50 steps (au lieu de 10)
                    if global_step % 50 == 0:
                        self._log_model_histograms(global_step)
                    
                    domain_history.append(combined_metrics)
                    
                    # Sauvegarde périodique
                    if global_step % self.config.training.save_every == 0:
                        self._save_checkpoint(global_step, combined_metrics)
                    
                    logger.info(f"Step {global_step}: Reward={combined_metrics['total_reward']:.4f}, "
                               f"Accuracy={combined_metrics['accuracy']:.4f}")
                    
                    epochs_at_complexity += 1
                    global_step += 1
                    steps_since_complexity_change += 1
                    
                    # Vérifier si on peut passer à la complexité suivante
                    # Critères réalistes : au moins 20 steps ET performance suffisante
                    if (steps_since_complexity_change >= self.config.training.min_steps_per_level and 
                        self._should_increase_complexity(domain_history)):
                        logger.info(f"✅ Performance suffisante atteinte pour {domain} après {steps_since_complexity_change} steps")
                        break
                
                complexity_history.extend(domain_history)
            
            # Évaluation globale à cette complexité
            complexity_evaluation = {}
            for domain in self.problem_generator.get_available_domains():
                eval_result = self._evaluate_model(current_complexity, domain)
                complexity_evaluation[domain] = eval_result
            
            self.experiment_log['complexity_progression'].append({
                'complexity': current_complexity,
                'evaluation': complexity_evaluation,
                'step': global_step
            })
            
            # Décision de progression basée sur les paramètres du config
            avg_reward = sum(
                eval_result['total_reward'] for eval_result in complexity_evaluation.values()
            ) / len(complexity_evaluation)
            
            avg_accuracy = sum(
                eval_result['accuracy'] for eval_result in complexity_evaluation.values()
            ) / len(complexity_evaluation)
            
            # Critères configurables depuis config.py
            reward_ok = avg_reward > self.config.training.complexity_threshold
            accuracy_ok = avg_accuracy > self.config.training.min_accuracy_threshold
            
            # Logique flexible (OU) ou stricte (ET) selon config
            if self.config.training.use_flexible_criteria:
                criteria_met = reward_ok or accuracy_ok  # L'un OU l'autre suffit
            else:
                criteria_met = reward_ok and accuracy_ok  # Les deux requis
            
            if criteria_met:
                current_complexity += 1
                logger.info(f"🚀 Progression vers la complexité {current_complexity} "
                           f"(Reward: {avg_reward:.3f}, Accuracy: {avg_accuracy:.3f})")
                
                # RÉINITIALISATION : Reset pour nouveau niveau
                self.trainer.reset_success_history()
                steps_since_complexity_change = 0
                logger.info("🔄 Historique réinitialisé pour le nouveau niveau de complexité")
            else:
                logger.info(f"⏳ Performance insuffisante, continuation à la complexité {current_complexity} "
                           f"(Reward: {avg_reward:.3f}, Accuracy: {avg_accuracy:.3f})")
    
    def run_standard_training(self):
        """Execute un entraînement standard sans curriculum"""
        logger.info("🏋️ Démarrage de l'entraînement standard")
        
        global_step = 0
        
        for epoch in range(self.config.training.max_epochs):
            logger.info(f"📈 Époque {epoch + 1}/{self.config.training.max_epochs}")
            
            epoch_metrics = []
            
            # Entraînement sur tous les domaines et complexités
            for domain in self.problem_generator.get_available_domains():
                for complexity in range(1, self.config.training.max_complexity + 1):
                    batch_metrics = self.trainer.train_epoch(
                        domain, complexity, num_batches=5
                    )
                    
                    # Évaluation
                    eval_metrics = self._evaluate_model(complexity, domain)
                    
                    combined_metrics = {**eval_metrics, **batch_metrics}  # batch_metrics en dernier !
                    combined_metrics['epoch'] = epoch
                    combined_metrics['complexity'] = complexity
                    combined_metrics['domain'] = domain
                    combined_metrics['global_step'] = global_step
                    
                    epoch_metrics.append(combined_metrics)
                    self._log_to_tensorboard(combined_metrics, global_step)
                    
                    global_step += 1
            
            # Moyenne des métriques de l'époque
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                if isinstance(epoch_metrics[0][key], (int, float)):
                    avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            
            self.experiment_log['epochs'].append(avg_metrics)
            
            # Sauvegarde
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch, avg_metrics)
            
            logger.info(f"Époque {epoch}: Reward moyen={avg_metrics.get('total_reward', 0):.4f}")
    
    def run(self, mode: str = "curriculum"):
        """Lance l'expérimentation"""
        logger.info(f"🚀 Démarrage de l'expérimentation: {self.experiment_name}")
        
        start_time = time.time()
        
        try:
            if mode == "curriculum":
                self.run_curriculum_learning()
            else:
                self.run_standard_training()
            
            # Finalisation
            end_time = time.time()
            duration = end_time - start_time
            
            self.experiment_log['end_time'] = datetime.now().isoformat()
            self.experiment_log['duration_seconds'] = duration
            
            # Génération du rapport final
            self._generate_final_report()
            
            logger.info(f"✅ Expérimentation terminée en {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'expérimentation: {e}")
            raise
        finally:
            if self.tb_writer:
                self.tb_writer.close()
    
    def _generate_final_report(self):
        """Génère le rapport final de l'expérimentation"""
        # Sauvegarde des logs
        log_path = os.path.join(self.experiment_dir, "experiment_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        
        # Génération des visualisations
        if self.trainer.metrics_history['reward']:
            viz_path = os.path.join(self.experiment_dir, "visualizations", "training_curves.png")
            ValidationUtils.create_performance_visualizations(
                self.trainer.metrics_history, viz_path
            )
        
        # Rapport de complexité
        complexity_report = ValidationUtils.generate_complexity_report(
            self.trainer, self.problem_generator
        )
        
        report_path = os.path.join(self.experiment_dir, "complexity_report.json")
        with open(report_path, 'w') as f:
            json.dump(complexity_report, f, indent=2)
        
        logger.info(f"📊 Rapport final sauvegardé dans: {self.experiment_dir}")
