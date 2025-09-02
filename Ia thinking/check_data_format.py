"""
Vérification du format des données d'entraînement vs test
"""

import torch
from autonomous_reasoning_ai.problem_generators import MathProblemGenerator

def check_data_format():
    """Vérifie le format des données"""
    print("🔍 === VÉRIFICATION FORMAT DES DONNÉES ===")
    print("=" * 50)
    
    # Créer un générateur de problèmes
    math_gen = MathProblemGenerator()
    
    # Générer quelques problèmes de niveau 1
    problems = math_gen.generate(complexity=1, batch_size=3)
    
    print(f"📊 Problèmes générés: {len(problems)}")
    
    for i, problem in enumerate(problems):
        print(f"\n🔬 Problème {i+1}:")
        print(f"   📝 Métadonnées: {problem.metadata}")
        print(f"   📐 Shape input: {problem.input_data.shape}")
        print(f"   🔢 Premières valeurs: {problem.input_data[:10].tolist()}")
        
        # Test de la fonction de validation
        if problem.metadata['operation'] == 'addition':
            a, b = problem.metadata['operands']
            expected = problem.metadata['expected']
            
            # Tester différentes réponses pour voir comment la validation fonctionne
            test_values = [expected, expected + 0.1, expected - 0.1, expected + 1, 0]
            print(f"   🧪 Tests de validation:")
            for val in test_values:
                is_valid = problem.validation_fn(val)
                print(f"      Valeur {val}: {'✅' if is_valid else '❌'}")
    
    # Vérifier le format attendu par notre encodage de questions personnalisées
    print(f"\n🔧 Test du format d'encodage manuel:")
    test_input = torch.zeros(64)
    test_input[0] = 5.0   # Premier nombre
    test_input[1] = 3.0   # Deuxième nombre  
    test_input[2] = 0.0   # Opération (addition)
    # Pas de résultat attendu en position 3 !
    
    print(f"   📐 Format manuel: {test_input[:10].tolist()}")
    
    # Comparer avec le format du générateur
    addition_problem = None
    for problem in problems:
        if problem.metadata['operation'] == 'addition':
            addition_problem = problem
            break
    
    if addition_problem:
        print(f"   📐 Format générateur: {addition_problem.input_data[:10].tolist()}")
        print(f"   🔍 Différences détectées: {'Oui' if not torch.allclose(test_input[:4], addition_problem.input_data[:4], atol=0.1) else 'Non'}")

if __name__ == "__main__":
    check_data_format()
