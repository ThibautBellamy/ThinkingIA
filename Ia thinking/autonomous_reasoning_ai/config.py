"""
Configuration globale pour le système d'IA de raisonnement autonome
"""

from dataclasses import dataclass
from typing import Dict, Any
import torch

@dataclass
class ModelConfig:
    """Configuration du modèle de raisonnement"""
    input_dim: int = 64
    hidden_dim: int = 256
    max_reasoning_steps: int = 10
    dropout_rate: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    """Configuration d'entraînement"""
    learning_rate: float = 0.0001  # Divisé par 10 pour stabiliser
    batch_size: int = 16  # Réduit de 32 à 16
    max_epochs: int = 100
    patience: int = 10
    validation_split: float = 0.2
    
    # Curriculum learning - Tous les seuils centralisés
    initial_complexity: int = 1
    max_complexity: int = 5
    complexity_threshold: float = 0.65  # Seuil reward pour progression
    min_steps_per_level: int = 20       # Steps minimum par niveau
    min_accuracy_threshold: float = 0.1  # Seuil accuracy alternatif (10%)
    use_flexible_criteria: bool = True   # OU au lieu de ET
    
    # Auto-supervision - Simplifié
    consistency_weight: float = 0.8  # Focus sur la consistance
    confidence_weight: float = 0.1   # Moins d'importance
    validation_weight: float = 0.1   # Moins d'importance
    
    # Sauvegarde
    save_every: int = 10

@dataclass
class ProblemConfig:
    """Configuration des générateurs de problèmes"""
    domains: list = None
    complexity_range: tuple = (1, 10)
    problems_per_complexity: int = 100
    validation_problems: int = 20
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["math", "logic", "pattern"]

@dataclass
class ExperimentConfig:
    """Configuration des expérimentations"""
    experiment_name: str = "autonomous_reasoning_v1"
    log_dir: str = "logs"
    results_dir: str = "results"
    tensorboard_enabled: bool = True
    save_reasoning_traces: bool = True
    
    # Métriques à suivre
    metrics_to_track: list = None
    
    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = [
                "accuracy", "consistency", "confidence", 
                "reasoning_depth", "convergence_time"
            ]

class GlobalConfig:
    """Configuration globale du système"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.problems = ProblemConfig()
        self.experiment = ExperimentConfig()
        
        # Seed pour la reproductibilité
        self.random_seed = 42
        
        # Niveau de logging
        self.log_level = "INFO"
        
        # Optimisations
        self.use_mixed_precision = True
        self.compile_model = False  # Désactivé sur Windows sans compilateur C++
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "problems": self.problems.__dict__,
            "experiment": self.experiment.__dict__,
            "random_seed": self.random_seed,
            "log_level": self.log_level,
            "use_mixed_precision": self.use_mixed_precision,
            "compile_model": self.compile_model
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Met à jour la configuration depuis un dictionnaire"""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(getattr(self, section), (ModelConfig, TrainingConfig, ProblemConfig, ExperimentConfig)):
                for key, value in values.items():
                    setattr(getattr(self, section), key, value)
            else:
                setattr(self, section, values)

# Instance globale de configuration
config = GlobalConfig()
