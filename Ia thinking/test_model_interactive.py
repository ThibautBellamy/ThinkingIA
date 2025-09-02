"""
Test interactif du modèle d'IA de raisonnement autonome
"""

import torch
import numpy as np
from main_with_trained_model import initialize_components  # Utilise la version avec modèle entraîné
from autonomous_reasoning_ai.problem_generators import MathProblemGenerator, PatternProblemGenerator
from autonomous_reasoning_ai.utils.validation import ValidationUtils
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_on_problems():
    """Test le modèle sur différents types de problèmes"""
    
    print("🤖 Test Interactif du Modèle d'IA")
    print("=" * 50)
    
    # Chargement des composants
    print("📱 Chargement du modèle...")
    model, problem_generator, trainer = initialize_components()
    device = model.device
    
    # Test sur différentes complexités
    complexities = [1, 2, 3, 4, 5]
    domains = ['math']  # Seul le domaine 'math' est disponible actuellement
    
    print(f"\n🧪 Test sur {len(complexities)} niveaux de complexité")
    print(f"🎯 Domaines disponibles: {domains}")
    
    results_summary = []
    
    for complexity in complexities:
        print(f"\n📊 === NIVEAU {complexity} ===")
        
        for domain in domains:
            print(f"\n🔬 Domaine: {domain}")
            
            # Génération de problèmes de test
            test_problems = problem_generator.generate_batch(
                domain, complexity, batch_size=5
            )
            
            # Évaluation
            eval_result = ValidationUtils.evaluate_model_on_problems(
                model, test_problems, device
            )
            
            # Affichage des résultats
            print(f"   📈 Accuracy: {eval_result['accuracy']:.1%}")
            print(f"   🎯 Reward: {eval_result['total_reward']:.3f}")
            print(f"   🧠 Confiance moy: {np.mean(eval_result['confidence']):.3f}")
            print(f"   🔄 Consistance moy: {np.mean(eval_result['consistency_scores']):.3f}")
            
            results_summary.append({
                'complexity': complexity,
                'domain': domain,
                'accuracy': eval_result['accuracy'],
                'reward': eval_result['total_reward'],
                'confidence': np.mean(eval_result['confidence']),
                'consistency': np.mean(eval_result['consistency_scores'])
            })
    
    # Résumé global
    print(f"\n📋 === RÉSUMÉ GLOBAL ===")
    print(f"{'Niveau':<8} {'Domaine':<8} {'Accuracy':<10} {'Reward':<8} {'Confiance':<10}")
    print("-" * 50)
    
    for result in results_summary:
        print(f"{result['complexity']:<8} {result['domain']:<8} "
              f"{result['accuracy']:<10.1%} {result['reward']:<8.3f} "
              f"{result['confidence']:<10.3f}")
    
    return results_summary

def test_specific_problem():
    """Test le modèle sur un problème spécifique"""
    
    print("\n🎯 === TEST SPÉCIFIQUE ===")
    
    # Chargement du modèle
    model, problem_generator, trainer = initialize_components()
    
    # Génération d'un problème mathématique simple
    math_gen = MathProblemGenerator()
    problem = math_gen.generate_addition_problem(complexity=1)
    
    print(f"🧮 Problème mathématique:")
    print(f"   Input: {problem.input_data}")
    print(f"   Expected: {problem.expected_output}")
    
    # Test du modèle
    model.eval()
    with torch.no_grad():
        input_tensor = problem.input_data.unsqueeze(0).to(model.device)
        output = model(input_tensor)
        
        print(f"\n🤖 Résultat du modèle:")
        print(f"   Solution: {output['solution']}")
        print(f"   Confiance: {output['final_confidence'].mean().item():.3f}")
        print(f"   Consistance: {output['consistency_score'].item():.3f}")
        print(f"   Profondeur: {output['reasoning_depth']}")
        
        # Validation
        if problem.validation_fn:
            is_correct = problem.validation_fn(output['solution'])
            print(f"   ✅ Correct: {is_correct}")
        
        return output

def interactive_menu():
    """Menu interactif pour tester le modèle"""
    
    while True:
        print(f"\n🎮 === MENU DE TEST ===")
        print("1. 📊 Test complet (tous niveaux)")
        print("2. 🎯 Test problème spécifique")
        print("3. 🧮 Test mathématiques niveau 1")
        print("4. 🔄 Test patterns niveau 2")
        print("5. 💬 Poser ma propre question mathématique")
        print("6. ❌ Quitter")
        
        choice = input("\nChoix (1-6): ").strip()
        
        if choice == "1":
            test_model_on_problems()
        elif choice == "2":
            test_specific_problem()
        elif choice == "3":
            test_math_level_1()
        elif choice == "4":
            test_patterns_level_2()
        elif choice == "5":
            test_custom_math_question()
        elif choice == "6":
            print("👋 Au revoir!")
            break
        else:
            print("❌ Choix invalide!")

def test_math_level_1():
    """Test spécifique sur les maths niveau 1"""
    print("\n🧮 === TEST MATHS NIVEAU 1 ===")
    
    model, problem_generator, trainer = initialize_components()
    
    # Test sur 10 problèmes
    problems = problem_generator.generate_batch('math', 1, batch_size=10)
    
    correct = 0
    total_confidence = 0
    
    for i, problem in enumerate(problems):
        model.eval()
        with torch.no_grad():
            input_tensor = problem.input_data.unsqueeze(0).to(model.device)
            output = model(input_tensor)
            
            confidence = output['final_confidence'].mean().item()
            total_confidence += confidence
            
            if problem.validation_fn and problem.validation_fn(output['solution']):
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"   Problème {i+1}: {status} (Confiance: {confidence:.3f})")
    
    accuracy = correct / len(problems)
    avg_confidence = total_confidence / len(problems)
    
    print(f"\n📊 Résultats Maths Niveau 1:")
    print(f"   Accuracy: {accuracy:.1%} ({correct}/{len(problems)})")
    print(f"   Confiance moyenne: {avg_confidence:.3f}")

def test_patterns_level_2():
    """Test spécifique sur les patterns niveau 2"""
    print("\n🔄 === TEST PATTERNS NIVEAU 2 ===")
    
    model, problem_generator, trainer = initialize_components()
    
    # Test sur patterns niveau 2
    problems = problem_generator.generate_batch('pattern', 2, batch_size=5)
    
    for i, problem in enumerate(problems):
        model.eval()
        with torch.no_grad():
            input_tensor = problem.input_data.unsqueeze(0).to(model.device)
            output = model(input_tensor)
            
            print(f"\n🔄 Pattern {i+1}:")
            print(f"   Input: {problem.input_data.shape}")
            print(f"   Solution: {output['solution'].shape}")
            print(f"   Confiance: {output['final_confidence'].mean().item():.3f}")
            print(f"   Consistance: {output['consistency_score'].item():.3f}")

def test_custom_math_question():
    """Permet à l'utilisateur de poser sa propre question mathématique"""
    print("\n💬 === QUESTION MATHÉMATIQUE PERSONNALISÉE ===")
    print("🧮 Types de questions supportées:")
    print("   • Addition: ex. 5 + 3")
    print("   • Soustraction: ex. 10 - 4") 
    print("   • Multiplication: ex. 6 * 7")
    print("   • Questions simples avec nombres entiers")
    
    # Charger le modèle
    model, problem_generator, trainer = initialize_components()
    
    while True:
        print("\n" + "="*50)
        question = input("🤔 Votre question mathématique (ou 'quit' pour quitter): ").strip()
        
        if question.lower() in ['quit', 'quitter', 'q', 'exit']:
            print("👋 Retour au menu principal")
            break
            
        if not question:
            print("❌ Veuillez entrer une question!")
            continue
            
        try:
            # Parse la question mathématique
            result = parse_and_solve_math_question(question, model)
            
            if result:
                print(f"\n🤖 === RÉPONSE DU MODÈLE ===")
                print(f"   📝 Question: {question}")
                print(f"   🔢 Réponse calculée: {result['calculated_answer']}")
                print(f"   🤖 Réponse IA: {result['ai_response']:.3f}")
                print(f"   ✅ Correct: {'Oui' if result['is_correct'] else 'Non'}")
                print(f"   🎯 Confiance: {result['confidence']:.3f}")
                print(f"   🧠 Consistance: {result['consistency']:.3f}")
                print(f"   🔄 Profondeur raisonnement: {result['reasoning_depth']}")
                
                if result['is_correct']:
                    print("   🎉 Excellente réponse!")
                else:
                    print(f"   📚 La bonne réponse était: {result['calculated_answer']}")
            else:
                print("❌ Format de question non reconnu!")
                print("💡 Essayez: '5 + 3', '10 - 2', '4 * 6', etc.")
                
        except Exception as e:
            print(f"❌ Erreur lors du traitement: {e}")
            print("💡 Vérifiez le format de votre question")

def parse_and_solve_math_question(question, model):
    """Parse une question mathématique et la résout avec le modèle"""
    import re
    
    # Nettoyer la question
    question = question.replace(" ", "").lower()
    
    # Patterns pour différents types d'opérations
    patterns = [
        (r'(\d+)\+(\d+)', lambda a, b: int(a) + int(b), 'addition'),
        (r'(\d+)\-(\d+)', lambda a, b: int(a) - int(b), 'soustraction'),
        (r'(\d+)\*(\d+)', lambda a, b: int(a) * int(b), 'multiplication'),
        (r'(\d+)x(\d+)', lambda a, b: int(a) * int(b), 'multiplication'),
    ]
    
    for pattern, calc_func, operation_type in patterns:
        match = re.match(pattern, question)
        if match:
            num1, num2 = match.groups()
            
            # Calcul de la vraie réponse
            calculated_answer = calc_func(num1, num2)
            
            # Création d'un problème pour le modèle
            problem_tensor = create_math_problem_tensor(int(num1), int(num2), operation_type)
            
            # Test avec le modèle
            model.eval()
            with torch.no_grad():
                input_tensor = problem_tensor.unsqueeze(0).to(model.device)
                output = model(input_tensor)
                
                # Extraction de la réponse du modèle
                ai_response = output['solution'].mean().item()
                confidence = output['final_confidence'].mean().item()
                consistency = output['consistency_score'].item()
                reasoning_depth = output['reasoning_depth']
                
                # Vérification si c'est correct (tolérance de 0.1)
                is_correct = abs(ai_response - calculated_answer) < 0.1
                
                return {
                    'calculated_answer': calculated_answer,
                    'ai_response': ai_response,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'consistency': consistency,
                    'reasoning_depth': reasoning_depth,
                    'operation_type': operation_type
                }
    
    return None

def create_math_problem_tensor(num1, num2, operation_type):
    """Crée un tenseur d'entrée pour un problème mathématique"""
    # Utiliser le même format que le générateur : [num1, num2, operation_code]
    operation_codes = {
        'addition': 0.0,        # Même encodage que le générateur
        'soustraction': 1.0, 
        'multiplication': 2.0,
        'division': 3.0
    }
    
    # Pas de normalisation - utiliser les valeurs directes comme le générateur
    op_code = operation_codes.get(operation_type, 0.0)
    
    # Création du tenseur d'entrée (même format que le générateur)
    input_data = torch.zeros(64)  # Dimension d'entrée du modèle
    input_data[0] = float(num1)  # Pas de normalisation
    input_data[1] = float(num2)  # Pas de normalisation
    input_data[2] = op_code      # Code d'opération
    # Note: Pas de résultat attendu - le modèle doit le calculer !
    
    return input_data

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n👋 Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
