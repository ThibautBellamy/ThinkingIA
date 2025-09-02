"""
Debug du format des données pour questions personnalisées vs générateur
"""

import torch
from autonomous_reasoning_ai.problem_generators import MathProblemGenerator
from test_model_interactive import create_math_problem_tensor

def debug_data_format():
    """Compare les formats de données"""
    print("🔍 === DEBUG FORMAT DES DONNÉES ===")
    print("=" * 50)
    
    # Générer un problème avec le générateur officiel
    math_gen = MathProblemGenerator()
    problems = math_gen.generate(complexity=1, batch_size=5)
    
    # Trouver un problème d'addition
    addition_problem = None
    for prob in problems:
        if prob.metadata['operation'] == 'addition':
            addition_problem = prob
            break
    
    if addition_problem:
        a, b = addition_problem.metadata['operands']
        expected = addition_problem.metadata['expected']
        
        print(f"📊 Problème d'addition trouvé: {a} + {b} = {expected}")
        print(f"🧪 Format générateur:")
        print(f"   📐 Shape: {addition_problem.input_data.shape}")
        print(f"   🔢 Data: {addition_problem.input_data[:10].tolist()}")
        
        # Créer la même addition avec notre fonction manuelle
        manual_tensor = create_math_problem_tensor(a, b, 'addition')
        print(f"\n🔧 Format manuel:")
        print(f"   📐 Shape: {manual_tensor.shape}")
        print(f"   🔢 Data: {manual_tensor[:10].tolist()}")
        
        # Comparer les différences
        print(f"\n🔍 Analyse des différences:")
        print(f"   Position 0 (a): Générateur={addition_problem.input_data[0]:.3f}, Manuel={manual_tensor[0]:.3f}")
        print(f"   Position 1 (b): Générateur={addition_problem.input_data[1]:.3f}, Manuel={manual_tensor[1]:.3f}")
        print(f"   Position 2 (op): Générateur={addition_problem.input_data[2]:.3f}, Manuel={manual_tensor[2]:.3f}")
        print(f"   Position 3: Générateur={addition_problem.input_data[3]:.3f}, Manuel={manual_tensor[3]:.3f}")
        
        # Test de différence
        diff = torch.abs(addition_problem.input_data - manual_tensor).max().item()
        print(f"   📏 Différence max: {diff:.6f}")
        
        if diff > 0.001:
            print(f"   ⚠️ DIFFÉRENCE DÉTECTÉE ! Les formats ne correspondent pas.")
        else:
            print(f"   ✅ Formats identiques")
            
    else:
        print("❌ Aucun problème d'addition trouvé")
    
    # Tester avec d'autres opérations
    print(f"\n🧮 Test des autres opérations:")
    
    test_cases = [
        (5, 3, 'addition', 0.0),
        (10, 4, 'soustraction', 1.0),
        (6, 7, 'multiplication', 2.0),
    ]
    
    for a, b, op_name, expected_code in test_cases:
        manual_tensor = create_math_problem_tensor(a, b, op_name)
        print(f"   {op_name}: [{manual_tensor[0]:.1f}, {manual_tensor[1]:.1f}, {manual_tensor[2]:.1f}] (attendu: code {expected_code})")

if __name__ == "__main__":
    debug_data_format()
