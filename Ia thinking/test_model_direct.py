"""
Test direct du modèle avec problèmes générés vs encodage manuel
"""

import torch
from main_with_trained_model import initialize_components
from autonomous_reasoning_ai.problem_generators import MathProblemGenerator
from test_model_interactive import create_math_problem_tensor

def test_model_direct():
    """Test direct du modèle avec différents formats"""
    print("🧪 === TEST DIRECT DU MODÈLE ===")
    print("=" * 50)
    
    # Charger le modèle
    print("📂 Chargement du modèle...")
    model, problem_generator, _ = initialize_components()
    model.eval()
    
    # Test 1: Problèmes générés automatiquement
    print("\n🤖 Test 1: Problèmes générés automatiquement")
    print("-" * 40)
    
    math_gen = MathProblemGenerator()
    problems = math_gen.generate(complexity=1, batch_size=5)
    
    correct_count = 0
    for i, problem in enumerate(problems):
        with torch.no_grad():
            input_data = problem.input_data.unsqueeze(0).to(model.device)
            output = model(input_data)
        
        # Le modèle retourne un dictionnaire avec plusieurs valeurs
        if isinstance(output, dict):
            solution_tensor = output.get('solution', output.get('answer', output.get('response', torch.tensor(0.0))))
            
            # Extraire la valeur scalaire du tensor solution
            if hasattr(solution_tensor, 'item'):
                if solution_tensor.numel() == 1:
                    result = solution_tensor.item()
                else:
                    # Si c'est un tensor multi-éléments, prendre la moyenne ou le premier élément
                    result = solution_tensor.mean().item()
            else:
                result = float(solution_tensor)
        else:
            result = output.item() if hasattr(output, 'item') else output
        
        is_correct = problem.validation_fn(result)
        
        print(f"   Problème {i+1}: {problem.metadata}")
        print(f"   🤖 Réponse modèle: {result:.6f}")
        print(f"   📊 Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"   🔑 Keys disponibles: {list(output.keys())}")
        print(f"   ✅ Correct: {'Oui' if is_correct else 'Non'}")
        print()
        
        if is_correct:
            correct_count += 1
    
    print(f"📊 Résultat générés: {correct_count}/{len(problems)} = {100*correct_count/len(problems):.1f}%")
    
    # Test 2: Même problèmes mais avec encodage manuel
    print("\n🔧 Test 2: Mêmes problèmes avec encodage manuel")
    print("-" * 40)
    
    correct_manual = 0
    for i, problem in enumerate(problems):
        if problem.metadata['operation'] in ['addition', 'soustraction', 'multiplication']:
            a, b = problem.metadata['operands']
            expected = problem.metadata['expected']
            
            # Encoder manuellement
            manual_tensor = create_math_problem_tensor(a, b, problem.metadata['operation'])
            
            with torch.no_grad():
                input_data = manual_tensor.unsqueeze(0).to(model.device)
                output = model(input_data)
            
            # Extraire la réponse du dictionnaire ou tensor
            if isinstance(output, dict):
                solution_tensor = output.get('solution', output.get('answer', output.get('response', torch.tensor(0.0))))
                
                # Extraire la valeur scalaire du tensor solution
                if hasattr(solution_tensor, 'item'):
                    if solution_tensor.numel() == 1:
                        result = solution_tensor.item()
                    else:
                        # Si c'est un tensor multi-éléments, prendre la moyenne
                        result = solution_tensor.mean().item()
                else:
                    result = float(solution_tensor)
            else:
                result = output.item() if hasattr(output, 'item') else output
            
            is_correct = problem.validation_fn(result)
            
            print(f"   Problème {i+1}: {a} {'+' if problem.metadata['operation']=='addition' else '-' if problem.metadata['operation']=='soustraction' else '*'} {b} = {expected}")
            print(f"   🤖 Réponse modèle: {result:.6f}")
            print(f"   ✅ Correct: {'Oui' if is_correct else 'Non'}")
            print()
            
            if is_correct:
                correct_manual += 1
    
    print(f"📊 Résultat manuels: {correct_manual}/{len(problems)} = {100*correct_manual/len(problems):.1f}%")
    
    # Test 3: Questions spécifiques simples
    print("\n🎯 Test 3: Questions spécifiques simples")
    print("-" * 40)
    
    test_questions = [
        (5, 3, 'addition', 8),
        (10, 4, 'soustraction', 6),
        (6, 2, 'multiplication', 12),
        (2, 2, 'addition', 4),
        (1, 1, 'addition', 2),
    ]
    
    for a, b, op, expected in test_questions:
        manual_tensor = create_math_problem_tensor(a, b, op)
        
        with torch.no_grad():
            input_data = manual_tensor.unsqueeze(0).to(model.device)
            output = model(input_data)
        
        # Extraire la réponse
        if isinstance(output, dict):
            solution_tensor = output.get('solution', output.get('answer', output.get('response', torch.tensor(0.0))))
            
            # Extraire la valeur scalaire du tensor solution
            if hasattr(solution_tensor, 'item'):
                if solution_tensor.numel() == 1:
                    result = solution_tensor.item()
                else:
                    # Si c'est un tensor multi-éléments, prendre la moyenne
                    result = solution_tensor.mean().item()
            else:
                result = float(solution_tensor)
        else:
            result = output.item() if hasattr(output, 'item') else output
        
        is_correct = abs(result - expected) < 0.5
        
        op_symbol = '+' if op == 'addition' else '-' if op == 'soustraction' else '*'
        print(f"   Question: {a} {op_symbol} {b} = {expected}")
        print(f"   🤖 Réponse modèle: {result:.6f}")
        print(f"   ✅ Correct: {'Oui' if is_correct else 'Non'}")
        print()
    
    # Test 4: Inspection des activations
    print("\n🔍 Test 4: Inspection des activations internes")
    print("-" * 40)
    
    # Test avec une addition simple
    test_tensor = create_math_problem_tensor(5, 3, 'addition')
    print(f"📊 Input: {test_tensor[:5].tolist()}")
    
    with torch.no_grad():
        input_data = test_tensor.unsqueeze(0).to(model.device)
        
        # Hook pour capturer les activations intermédiaires
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Enregistrer des hooks sur quelques couches
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and len(list(module.named_modules())) == 1:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        output = model(input_data)
        
        # Nettoyer les hooks
        for hook in hooks:
            hook.remove()
        
        # Extraire la réponse finale
        if isinstance(output, dict):
            solution_tensor = output.get('solution', output.get('answer', output.get('response', torch.tensor(0.0))))
            
            # Extraire la valeur scalaire du tensor solution
            if hasattr(solution_tensor, 'item'):
                if solution_tensor.numel() == 1:
                    final_result = solution_tensor.item()
                else:
                    # Si c'est un tensor multi-éléments, analyser plus en détail
                    final_result = solution_tensor.mean().item()
                    print(f"📊 Solution tensor shape: {solution_tensor.shape}")
                    print(f"📊 Solution tensor values: {solution_tensor[:5].tolist()} ...")
            else:
                final_result = float(solution_tensor)
        else:
            final_result = output.item() if hasattr(output, 'item') else output
        
        print(f"🔢 Output final: {final_result:.6f}")
        print(f"📊 Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"🔑 Keys: {list(output.keys())}")
        print(f"📊 Nombre de couches capturées: {len(activations)}")
        
        # Afficher quelques statistiques sur les activations
        for name, activation in list(activations.items())[:3]:
            if activation.numel() > 0:
                print(f"   {name}: mean={activation.mean().item():.6f}, std={activation.std().item():.6f}")

if __name__ == "__main__":
    test_model_direct()
