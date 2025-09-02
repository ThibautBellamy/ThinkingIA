"""
Générateur de problèmes de reconnaissance de patterns
"""

import torch
import random
import numpy as np
from typing import List
from .base_generator import BaseGenerator, Problem

class PatternProblemGenerator(BaseGenerator):
    """Générateur de problèmes de reconnaissance de patterns"""
    
    def __init__(self):
        super().__init__(domain="pattern")
    
    def generate(self, complexity: int, batch_size: int) -> List[Problem]:
        """Génère des problèmes de patterns selon la complexité"""
        if not self.validate_complexity(complexity):
            raise ValueError(f"Complexité {complexity} non supportée")
        
        problems = []
        
        for _ in range(batch_size):
            if complexity == 1:
                problem = self._generate_arithmetic_sequence(complexity)
            elif complexity == 2:
                problem = self._generate_geometric_sequence(complexity)
            elif complexity == 3:
                problem = self._generate_fibonacci_like(complexity)
            elif complexity == 4:
                problem = self._generate_matrix_pattern(complexity)
            elif complexity >= 5:
                problem = self._generate_fractal_pattern(complexity)
            
            problems.append(problem)
            self.generated_count += 1
        
        return problems
    
    def _generate_arithmetic_sequence(self, complexity: int) -> Problem:
        """Génère des séquences arithmétiques"""
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        sequence = [start + i*step for i in range(4)]
        next_value = start + 4*step
        
        problem_data = torch.tensor([float(x) for x in sequence], dtype=torch.float32)
        validation_fn = lambda x: abs(x - next_value) < 0.1
        
        metadata = {
            "pattern_type": "arithmetic_sequence",
            "start": start,
            "step": step,
            "sequence": sequence,
            "expected_next": next_value
        }
        
        # Padding
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _generate_geometric_sequence(self, complexity: int) -> Problem:
        """Génère des séquences géométriques"""
        start = random.randint(2, 5)
        ratio = random.randint(2, 3)
        sequence = [start * (ratio**i) for i in range(4)]
        next_value = start * (ratio**4)
        
        problem_data = torch.tensor([float(x) for x in sequence], dtype=torch.float32)
        validation_fn = lambda x: abs(x - next_value) < 0.1
        
        metadata = {
            "pattern_type": "geometric_sequence",
            "start": start,
            "ratio": ratio,
            "sequence": sequence,
            "expected_next": next_value
        }
        
        # Padding
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _generate_fibonacci_like(self, complexity: int) -> Problem:
        """Génère des séquences Fibonacci-like"""
        a, b = random.randint(1, 5), random.randint(1, 5)
        sequence = [a, b, a+b, a+2*b, 2*a+3*b]
        next_value = 3*a + 5*b  # Continuation de la séquence
        
        problem_data = torch.tensor([float(x) for x in sequence], dtype=torch.float32)
        validation_fn = lambda x: abs(x - next_value) < 0.1
        
        metadata = {
            "pattern_type": "fibonacci_like",
            "initial_values": [a, b],
            "sequence": sequence,
            "expected_next": next_value
        }
        
        # Padding
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _generate_matrix_pattern(self, complexity: int) -> Problem:
        """Génère des patterns matriciels 2D"""
        size = 3
        matrix = [[random.randint(0, 9) for _ in range(size)] for _ in range(size)]
        
        # Pattern: somme des diagonales
        diag_sum = sum(matrix[i][i] for i in range(size))
        
        problem_data = torch.tensor([
            float(matrix[i][j]) for i in range(size) for j in range(size)
        ], dtype=torch.float32)
        
        validation_fn = lambda x: abs(x - diag_sum) < 0.1
        
        metadata = {
            "pattern_type": "matrix_diagonal",
            "matrix": matrix,
            "size": size,
            "expected_diagonal_sum": diag_sum
        }
        
        # Padding
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _generate_fractal_pattern(self, complexity: int) -> Problem:
        """Génère des patterns fractals"""
        depth = min(complexity - 2, 4)
        pattern = self._create_fractal_sequence(depth)
        
        # La prédiction est la continuation du pattern
        next_value = pattern[-1] * 1.618  # Ratio doré comme exemple
        
        problem_data = torch.tensor(pattern, dtype=torch.float32)
        validation_fn = lambda x: abs(x - next_value) < 0.5
        
        metadata = {
            "pattern_type": "fractal",
            "depth": depth,
            "sequence": pattern,
            "expected_next": next_value
        }
        
        # Padding
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _create_fractal_sequence(self, depth: int) -> List[float]:
        """Crée une séquence fractale simple"""
        pattern = [1.0]
        
        for d in range(depth):
            new_pattern = []
            for p in pattern:
                new_pattern.extend([p, p*2, p])
            pattern = new_pattern[:min(len(new_pattern), 20)]  # Limite la taille
        
        return pattern
    
    def get_complexity_description(self, complexity: int) -> str:
        """Retourne une description du niveau de complexité"""
        descriptions = {
            1: "Séquences arithmétiques (1, 3, 5, 7, ?)",
            2: "Séquences géométriques (2, 4, 8, 16, ?)",
            3: "Séquences Fibonacci-like (combinaisons linéaires)",
            4: "Patterns matriciels 2D (diagonales, symétries)",
            5: "Patterns fractals simples",
            6: "Patterns récursifs complexes",
            7: "Transformations géométriques",
            8: "Patterns multi-dimensionnels",
            9: "Systèmes dynamiques simples",
            10: "Attracteurs et chaos déterministe"
        }
        
        return descriptions.get(complexity, f"Complexité {complexity} (non définie)")
