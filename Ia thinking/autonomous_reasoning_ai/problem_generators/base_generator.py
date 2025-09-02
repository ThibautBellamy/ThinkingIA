"""
Classe de base pour les générateurs de problèmes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable, Any
import torch

@dataclass
class Problem:
    """Représente un problème à résoudre"""
    input_data: torch.Tensor
    complexity_level: int
    domain: str  # 'math', 'logic', 'pattern'
    validation_fn: Callable[[torch.Tensor], bool] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseGenerator(ABC):
    """Classe de base pour tous les générateurs de problèmes"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.generated_count = 0
        
    @abstractmethod
    def generate(self, complexity: int, batch_size: int) -> List[Problem]:
        """
        Génère un batch de problèmes pour un niveau de complexité donné
        
        Args:
            complexity: Niveau de complexité (1-10)
            batch_size: Nombre de problèmes à générer
            
        Returns:
            Liste de problèmes générés
        """
        pass
    
    @abstractmethod
    def get_complexity_description(self, complexity: int) -> str:
        """
        Retourne une description du niveau de complexité
        
        Args:
            complexity: Niveau de complexité
            
        Returns:
            Description textuelle de la complexité
        """
        pass
    
    def validate_complexity(self, complexity: int) -> bool:
        """Valide que le niveau de complexité est supporté"""
        return 1 <= complexity <= 10
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de génération"""
        return {
            "domain": self.domain,
            "generated_count": self.generated_count,
            "supported_complexity_range": (1, 10)
        }
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.generated_count = 0

class ProblemGenerator:
    """Générateur de problèmes avec complexité croissante"""
    
    def __init__(self):
        self.generators = {}
        self.register_default_generators()
    
    def register_generator(self, domain: str, generator: BaseGenerator):
        """Enregistre un générateur pour un domaine"""
        self.generators[domain] = generator
    
    def register_default_generators(self):
        """Enregistre les générateurs par défaut"""
        # Les générateurs sont enregistrés lors de l'initialisation dans main.py
        pass
    
    def generate_batch(self, domain: str, complexity: int, batch_size: int = 32) -> List[Problem]:
        """Génère un batch de problèmes pour un domaine et complexité donnés"""
        if domain not in self.generators:
            raise ValueError(f"Générateur pour le domaine '{domain}' non trouvé")
        
        generator = self.generators[domain]
        if not generator.validate_complexity(complexity):
            raise ValueError(f"Complexité {complexity} non supportée pour le domaine {domain}")
        
        return generator.generate(complexity, batch_size)
    
    def get_available_domains(self) -> List[str]:
        """Retourne les domaines disponibles"""
        return list(self.generators.keys())
    
    def get_generator_stats(self) -> dict:
        """Retourne les statistiques de tous les générateurs"""
        return {domain: gen.get_stats() for domain, gen in self.generators.items()}
