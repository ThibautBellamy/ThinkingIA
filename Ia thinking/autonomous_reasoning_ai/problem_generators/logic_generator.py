"""
Générateur de problèmes de logique
"""

import torch
import random
from typing import List
from .base_generator import BaseGenerator, Problem

class LogicProblemGenerator(BaseGenerator):
    """Générateur de problèmes de logique"""
    
    def __init__(self):
        super().__init__(domain="logic")
    
    def generate(self, complexity: int, batch_size: int) -> List[Problem]:
        """Génère des problèmes de logique selon la complexité"""
        if not self.validate_complexity(complexity):
            raise ValueError(f"Complexité {complexity} non supportée")
        
        problems = []
        
        for _ in range(batch_size):
            if complexity == 1:
                problem = self._generate_simple_syllogism(complexity)
            elif complexity == 2:
                problem = self._generate_propositional_logic(complexity)
            elif complexity >= 3:
                problem = self._generate_complex_logic(complexity)
            
            problems.append(problem)
            self.generated_count += 1
        
        return problems
    
    def _generate_simple_syllogism(self, complexity: int) -> Problem:
        """Génère des syllogismes simples"""
        # Si A alors B, A est vrai, donc B est vrai
        a, b = random.choice([0, 1]), random.choice([0, 1])
        
        if a == 1:  # Si A est vrai
            expected = b  # Alors B doit être la conclusion
        else:
            expected = random.choice([0, 1])  # Indéterminé
        
        problem_data = torch.tensor([float(a), float(b), float(expected)], dtype=torch.float32)
        validation_fn = lambda x: abs(x - expected) < 0.1
        
        metadata = {
            "logic_type": "syllogism",
            "premise_a": bool(a),
            "premise_b": bool(b),
            "expected": bool(expected)
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
    
    def _generate_propositional_logic(self, complexity: int) -> Problem:
        """Génère des problèmes de logique propositionnelle"""
        # (A AND B) OR (C AND D)
        a, b, c, d = [random.choice([0, 1]) for _ in range(4)]
        expected = (a and b) or (c and d)
        
        problem_data = torch.tensor([float(a), float(b), float(c), float(d)], dtype=torch.float32)
        validation_fn = lambda x: abs(x - float(expected)) < 0.1
        
        metadata = {
            "logic_type": "propositional",
            "variables": {"a": bool(a), "b": bool(b), "c": bool(c), "d": bool(d)},
            "formula": "(A AND B) OR (C AND D)",
            "expected": expected
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
    
    def _generate_complex_logic(self, complexity: int) -> Problem:
        """Génère des problèmes de logique complexes"""
        if complexity == 3:
            # Problème des trois interrupteurs
            switches = [random.choice([0, 1]) for _ in range(3)]
            lights = switches.copy()
            random.shuffle(lights)
            
            problem_data = torch.tensor([float(s) for s in switches + lights], dtype=torch.float32)
            validation_fn = lambda x: True  # Validation par correspondance
            
            metadata = {
                "logic_type": "three_switches",
                "switches": switches,
                "lights": lights
            }
            
        else:  # complexity >= 4
            # Logique temporelle ou sudoku simplifié
            size = min(complexity, 6)
            grid = [random.randint(0, size-1) for _ in range(size*size)]
            problem_data = torch.tensor([float(x) for x in grid], dtype=torch.float32)
            validation_fn = lambda x: True
            
            metadata = {
                "logic_type": "constraint_satisfaction",
                "grid_size": size,
                "initial_grid": grid
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
            1: "Syllogismes simples (Si A alors B)",
            2: "Logique propositionnelle (AND, OR, NOT)",
            3: "Puzzles logiques (trois interrupteurs)",
            4: "Problèmes de contraintes simples",
            5: "Logique temporelle basique",
            6: "Sudoku simplifié",
            7: "Problèmes de déduction complexes",
            8: "Logique modale basique",
            9: "Systèmes de contraintes complexes",
            10: "Problèmes de logique formelle avancée"
        }
        
        return descriptions.get(complexity, f"Complexité {complexity} (non définie)")
