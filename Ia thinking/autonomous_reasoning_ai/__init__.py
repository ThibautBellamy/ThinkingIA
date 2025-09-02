"""
IA de Raisonnement Autonome

Un système d'intelligence artificielle capable de raisonnement autonome
avec apprentissage auto-supervisé et complexité croissante.
"""

__version__ = "1.0.0"
__author__ = "Votre Nom"

from .config import config
from .models import AutonomousReasoningCore
from .problem_generators import (
    BaseGenerator, Problem, ProblemGenerator,
    MathProblemGenerator, LogicProblemGenerator, PatternProblemGenerator
)

__all__ = [
    "config",
    "AutonomousReasoningCore", 
    "BaseGenerator", "Problem", "ProblemGenerator",
    "MathProblemGenerator", "LogicProblemGenerator", "PatternProblemGenerator"
]
