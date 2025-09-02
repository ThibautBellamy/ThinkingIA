#!/usr/bin/env python3
"""
Test de la nouvelle architecture modulaire
"""

import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test des imports de la nouvelle architecture"""
    print("🔍 Test des imports de la nouvelle architecture...")
    
    try:
        # Test config
        from autonomous_reasoning_ai.config import config
        print("✅ Config importée")
        
        # Test models
        from autonomous_reasoning_ai.models import AutonomousReasoningCore
        print("✅ AutonomousReasoningCore importé")
        
        # Test problem generators
        from autonomous_reasoning_ai.problem_generators import (
            BaseGenerator, Problem, ProblemGenerator,
            MathProblemGenerator, LogicProblemGenerator, PatternProblemGenerator
        )
        print("✅ Générateurs de problèmes importés")
        
        # Test training
        from autonomous_reasoning_ai.training import AutonomousLearningTrainer
        print("✅ Trainer importé")
        
        # Test utils
        from autonomous_reasoning_ai.utils import ValidationUtils
        print("✅ Utilitaires importés")
        
        # Test experiments
        from autonomous_reasoning_ai.experiments import ExperimentRunner
        print("✅ ExperimentRunner importé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_functionality():
    """Test rapide des fonctionnalités"""
    print("\n🧪 Test des fonctionnalités de base...")
    
    try:
        from autonomous_reasoning_ai.config import config
        from autonomous_reasoning_ai.models import AutonomousReasoningCore
        from autonomous_reasoning_ai.problem_generators import ProblemGenerator, MathProblemGenerator
        
        # Test configuration
        print(f"✅ Configuration chargée - Device: {config.model.device}")
        
        # Test générateur de problèmes
        problem_gen = ProblemGenerator()
        problem_gen.register_generator("math", MathProblemGenerator())
        
        problems = problem_gen.generate_batch("math", complexity=1, batch_size=3)
        print(f"✅ {len(problems)} problèmes mathématiques générés")
        
        # Test modèle
        model = AutonomousReasoningCore()
        print(f"✅ Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
        
        # Test forward pass
        import torch
        with torch.no_grad():
            result = model(problems[0].input_data.unsqueeze(0))
        
        print(f"✅ Forward pass réussi:")
        print(f"   - Profondeur de raisonnement: {result['reasoning_depth']}")
        print(f"   - Confiance: {result['final_confidence'].item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de fonctionnalité: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structure():
    """Vérifie la structure des répertoires"""
    print("\n📁 Vérification de la structure...")
    
    base_dir = "autonomous_reasoning_ai"
    expected_files = [
        "config.py",
        "models/__init__.py",
        "models/reasoning_core.py",
        "problem_generators/__init__.py",
        "problem_generators/base_generator.py",
        "problem_generators/math_generator.py",
        "problem_generators/logic_generator.py",
        "problem_generators/pattern_generator.py",
        "training/__init__.py",
        "training/autonomous_trainer.py",
        "utils/__init__.py",
        "utils/validation.py",
        "experiments/__init__.py",
        "experiments/experiment_runner.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {missing_files}")
        return False
    
    print("✅ Structure complète !")
    return True

def main():
    """Fonction principale de test"""
    print("🤖 Test de la nouvelle architecture modulaire")
    print("=" * 60)
    
    success = True
    
    # Test de structure
    if not test_structure():
        success = False
    
    # Test des imports
    if not test_imports():
        success = False
    
    # Test des fonctionnalités
    if not test_quick_functionality():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Tous les tests sont passés ! Architecture prête !")
        print("\n📝 Prochaines étapes:")
        print("   1. cd autonomous_reasoning_ai")
        print("   2. python main.py test")
        print("   3. python main.py train")
    else:
        print("❌ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
