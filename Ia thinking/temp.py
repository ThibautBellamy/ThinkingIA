#!/usr/bin/env python3
"""
Test de tous les fichiers utilisant la configuration YAML
"""

import sys
import importlib
from autonomous_reasoning_ai.config import config

def test_config_loading():
    """Test du chargement de la configuration"""
    print("🔧 Test du chargement de la configuration...")
    
    try:
        # Vérifier que la config se charge
        assert config.get('project.name') == 'ThinkingIA'
        assert config.get('model.input_dim') is not None
        assert config.get('training.learning_rate') is not None
        print("✅ Configuration chargée correctement")
        return True
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

def test_imports():
    """Test des imports de tous les modules"""
    print("\n📦 Test des imports des modules...")
    
    modules_to_test = [
        'autonomous_reasoning_ai.models.reasoning_core',
        'autonomous_reasoning_ai.training.autonomous_trainer',
        'autonomous_reasoning_ai.experiments.experiment_runner',
        'autonomous_reasoning_ai.utils.validation',
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = "✅ OK"
            print(f"✅ {module_name}")
        except Exception as e:
            results[module_name] = f"❌ {str(e)}"
            print(f"❌ {module_name}: {e}")
    
    return results

def test_reasoning_core():
    """Test du modèle AutonomousReasoningCore"""
    print("\n🧠 Test du modèle AutonomousReasoningCore...")
    
    try:
        from autonomous_reasoning_ai.models import AutonomousReasoningCore
        
        # Test avec config
        model = AutonomousReasoningCore()
        print(f"✅ Modèle créé: {model.input_dim} → {model.hidden_dim}")
                
        return True
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return False

def test_autonomous_trainer():
    """Test du trainer autonome"""
    print("\n🎯 Test de AutonomousLearningTrainer...")
    
    try:
        from autonomous_reasoning_ai.training import AutonomousLearningTrainer
        
        # Test avec None (mode test)
        trainer = AutonomousLearningTrainer(None, None, config)
        print("✅ Trainer créé en mode test")
        
        # Test avec modèle réel
        from autonomous_reasoning_ai.models import AutonomousReasoningCore
        model = AutonomousReasoningCore()
        trainer_real = AutonomousLearningTrainer(model, None, config)
        print("✅ Trainer créé avec modèle réel")
        
        return True
    except Exception as e:
        print(f"❌ Erreur trainer: {e}")
        return False

def test_experiment_runner():
    """Test de ExperimentRunner"""
    print("\n🧪 Test de ExperimentRunner...")
    
    try:
        from autonomous_reasoning_ai.experiments import ExperimentRunner
        
        # Test avec None
        experiment = ExperimentRunner(None, None, None, config)
        print("✅ ExperimentRunner créé")
        
        return True
    except Exception as e:
        print(f"❌ Erreur ExperimentRunner: {e}")
        return False

def test_validation_utils():
    """Test des utilitaires de validation"""
    print("\n✅ Test des utilitaires de validation...")
    
    try:
        from autonomous_reasoning_ai.utils import validation
        
        # Test de fonctions de validation si elles existent
        print("✅ Module validation importé")
        
        return True
    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return False

def test_main_integration():
    """Test d'intégration avec main.py"""
    print("\n🚀 Test d'intégration main.py...")
    
    try:
        from autonomous_reasoning_ai import main
        
        # Test que les fonctions principales existent
        assert hasattr(main, 'setup_environment')
        assert hasattr(main, 'initialize_components')
        
        print("✅ main.py intégration OK")
        return True
    except Exception as e:
        print(f"❌ Erreur main.py: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🔥 TEST COMPLET DE TOUS LES FICHIERS UTILISANT CONFIG")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config_loading),
        ("Imports", test_imports),
        ("ReasoningCore", test_reasoning_core),
        ("AutonomousTrainer", test_autonomous_trainer),
        ("ExperimentRunner", test_experiment_runner),
        ("Validation", test_validation_utils),
        ("Main Integration", test_main_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[test_name] = f"❌ EXCEPTION: {e}"
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"{test_name:20} : {result}")
    
    # Statistiques
    passed = sum(1 for r in results.values() if r == "✅ PASS")
    total = len(results)
    
    print(f"\n🎯 RÉSULTAT FINAL: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) ont échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())
