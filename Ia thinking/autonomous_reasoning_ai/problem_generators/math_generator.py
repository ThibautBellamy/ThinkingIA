"""
Générateur de problèmes mathématiques graduels
"""

import torch
import random
import numpy as np
from typing import List
from .base_generator import BaseGenerator, Problem

def safe_float_extract(x):
    """Extrait une valeur float d'un tensor ou d'un float de manière sécurisée"""
    try:
        if hasattr(x, 'mean'):
            return float(x.mean().item())
        else:
            return float(x)
    except:
        return 0.0

class MathProblemGenerator(BaseGenerator):
    """Générateur de problèmes mathématiques graduels"""
    
    def __init__(self):
        super().__init__(domain="math")
        
    def generate(self, complexity: int, batch_size: int) -> List[Problem]:
        """Génère des problèmes mathématiques selon la complexité"""
        if not self.validate_complexity(complexity):
            raise ValueError(f"Complexité {complexity} non supportée")
        
        problems = []
        
        for _ in range(batch_size):
            if complexity == 1:
                problem = self._generate_arithmetic(complexity)
            elif complexity == 2:
                problem = self._generate_linear_equation(complexity)
            elif complexity == 3:
                problem = self._generate_quadratic_equation(complexity)
            elif complexity == 4:
                problem = self._generate_system_equations(complexity)
            elif complexity >= 5:
                problem = self._generate_polynomial(complexity)
            
            problems.append(problem)
            self.generated_count += 1
        
        return problems
    
    def _generate_arithmetic(self, complexity: int) -> Problem:
        """Génère des problèmes d'arithmétique simple"""
        a, b = random.randint(1, 10), random.randint(1, 10)
        operation = random.choice([0, 1, 2, 3])  # +, -, *, /
        
        if operation == 0:  # Addition
            result = a + b
            problem_data = torch.tensor([float(a), float(b), 0.0], dtype=torch.float32)  # Sans le résultat !
            validation_fn = lambda x: abs(safe_float_extract(x) - result) < 0.5
            metadata = {"operation": "addition", "operands": [a, b], "expected": result}
            
        elif operation == 1:  # Soustraction
            result = a - b
            problem_data = torch.tensor([float(a), float(b), 1.0], dtype=torch.float32)  # Sans le résultat !
            validation_fn = lambda x: abs(safe_float_extract(x) - result) < 0.5
            metadata = {"operation": "subtraction", "operands": [a, b], "expected": result}
            
        elif operation == 2:  # Multiplication
            result = a * b
            problem_data = torch.tensor([float(a), float(b), 2.0], dtype=torch.float32)  # Sans le résultat !
            validation_fn = lambda x: abs(safe_float_extract(x) - result) < 1.0
            metadata = {"operation": "multiplication", "operands": [a, b], "expected": result}
            
        else:  # Division
            b = max(1, b)  # Éviter division par zéro
            result = a / b
            problem_data = torch.tensor([float(a), float(b), 3.0], dtype=torch.float32)  # Sans le résultat !
            validation_fn = lambda x: abs(safe_float_extract(x) - result) < 0.5
            metadata = {"operation": "division", "operands": [a, b], "expected": result}
        
        # Padding pour avoir la dimension standard
        padded_data = torch.zeros(64, dtype=torch.float32)
        padded_data[:len(problem_data)] = problem_data
        
        return Problem(
            input_data=padded_data,
            complexity_level=complexity,
            domain=self.domain,
            validation_fn=validation_fn,
            metadata=metadata
        )
    
    def _generate_linear_equation(self, complexity: int) -> Problem:
        """Génère des équations linéaires: ax + b = c"""
        a = random.randint(1, 10)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)
        
        # Solution: x = (c - b) / a
        solution = (c - b) / a
        
        problem_data = torch.tensor([float(a), float(b), float(c)], dtype=torch.float32)
        validation_fn = lambda x: abs(safe_float_extract(x) - solution) < 0.1
        
        metadata = {
            "equation_type": "linear",
            "coefficients": {"a": a, "b": b, "c": c},
            "expected_solution": solution
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
    
    def _generate_quadratic_equation(self, complexity: int) -> Problem:
        """Génère des équations quadratiques: ax² + bx + c = 0"""
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        
        # Solution using quadratic formula
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            x2 = (-b - np.sqrt(discriminant)) / (2*a)
            solution = x1  # Prendre la solution positive
            has_real_solution = True
        else:
            solution = 0  # Pas de solution réelle
            has_real_solution = False
        
        problem_data = torch.tensor([float(a), float(b), float(c), 1.0], dtype=torch.float32)
        
        if has_real_solution:
            validation_fn = lambda x: abs(safe_float_extract(x) - solution) < 0.1
        else:
            validation_fn = lambda x: False  # Aucune solution réelle acceptable
        
        metadata = {
            "equation_type": "quadratic",
            "coefficients": {"a": a, "b": b, "c": c},
            "discriminant": discriminant,
            "has_real_solution": has_real_solution,
            "expected_solution": solution if has_real_solution else None
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
    
    def _generate_system_equations(self, complexity: int) -> Problem:
        """Génère des systèmes d'équations 2x2"""
        # ax + by = e
        # cx + dy = f
        a, b, c, d = [random.randint(1, 10) for _ in range(4)]
        e, f = [random.randint(1, 20) for _ in range(2)]
        
        # Solution par déterminant
        det = a*d - b*c
        
        if det != 0:
            x = (e*d - b*f) / det
            y = (a*f - e*c) / det
            solution = x  # Retourner x comme solution principale
            has_solution = True
        else:
            solution = 0
            has_solution = False
        
        problem_data = torch.tensor([
            float(a), float(b), float(c), float(d), float(e), float(f)
        ], dtype=torch.float32)
        
        if has_solution:
            validation_fn = lambda sol: abs(sol - solution) < 0.1
        else:
            validation_fn = lambda sol: False
        
        metadata = {
            "equation_type": "system_2x2",
            "coefficients": {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f},
            "determinant": det,
            "has_solution": has_solution,
            "expected_solution_x": solution if has_solution else None,
            "expected_solution_y": y if has_solution else None
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
    
    def _generate_polynomial(self, complexity: int) -> Problem:
        """Génère des polynômes d'ordre supérieur"""
        degree = min(complexity - 2, 5)
        coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
        
        # Pour simplifier, on cherche une racine approximative
        # En pratique, on pourrait utiliser des méthodes numériques
        test_values = [x * 0.1 for x in range(-50, 51)]
        root = None
        min_value = float('inf')
        
        for x in test_values:
            value = sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
            if abs(value) < min_value:
                min_value = abs(value)
                root = x
        
        problem_data = torch.tensor([float(c) for c in coeffs], dtype=torch.float32)
        
        validation_fn = lambda x: abs(safe_float_extract(x) - root) < 0.5 if root is not None else False
        
        metadata = {
            "equation_type": "polynomial",
            "degree": degree,
            "coefficients": coeffs,
            "approximate_root": root
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
    
    def get_complexity_description(self, complexity: int) -> str:
        """Retourne une description du niveau de complexité"""
        descriptions = {
            1: "Arithmétique simple (addition, soustraction, multiplication, division)",
            2: "Équations linéaires (ax + b = c)",
            3: "Équations quadratiques (ax² + bx + c = 0)",
            4: "Systèmes d'équations 2x2",
            5: "Polynômes de degré 3",
            6: "Polynômes de degré 4",
            7: "Polynômes de degré 5",
            8: "Équations transcendantales simples",
            9: "Systèmes d'équations 3x3",
            10: "Problèmes d'optimisation mathématique"
        }
        
        return descriptions.get(complexity, f"Complexité {complexity} (non définie)")
