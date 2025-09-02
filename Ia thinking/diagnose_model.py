"""
Script de diagnostic du modèle entraîné
"""

import torch
import os
import numpy as np
from main_with_trained_model import initialize_components  # Utilise la version avec modèle entraîné
from autonomous_reasoning_ai.utils.validation import ValidationUtils

def diagnose_model():
    """Diagnostique approfondi du modèle"""
    print("🔍 === DIAGNOSTIC DU MODÈLE ===")
    print("=" * 50)
    
    # 1. Vérification des fichiers de modèle
    print("\n📁 1. Vérification des fichiers...")
    model_path = r'results\autonomous_reasoning_v1_20250902_175324\best_model.pt'
    
    if os.path.exists(model_path):
        print("✅ Fichier de modèle trouvé")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"📊 Informations du checkpoint:")
        for key, value in checkpoint.items():
            if key != 'model_state_dict':
                print(f"   {key}: {value}")
        
        if 'model_state_dict' in checkpoint:
            print(f"🧠 Nombre de paramètres sauvegardés: {len(checkpoint['model_state_dict'])}")
        else:
            print("❌ État du modèle manquant dans le checkpoint!")
            return
    else:
        print(f"❌ Fichier de modèle non trouvé: {model_path}")
        return
    
    # 2. Chargement et test du modèle
    print("\n🤖 2. Chargement du modèle...")
    try:
        model, problem_generator, trainer = initialize_components()
        print("✅ Modèle chargé avec succès")
        print(f"🖥️ Device: {model.device}")
        print(f"🧠 Paramètres totaux: {sum(p.numel() for p in model.parameters())}")
        print(f"📊 Paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # 3. Test de forward pass simple
    print("\n🧪 3. Test de forward pass...")
    try:
        # Créer un tensor de test simple
        test_input = torch.zeros(1, 64, device=model.device)
        test_input[0, 0] = 5.0  # Premier nombre
        test_input[0, 1] = 3.0  # Deuxième nombre
        test_input[0, 2] = 0.0  # Opération (addition)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass réussi")
        print(f"📤 Input shape: {test_input.shape}")
        print(f"📥 Output shape: {output.shape}")
        print(f"🔢 Output value: {output.item():.6f}")
        print(f"🎯 Expected (5+3): 8")
        print(f"❓ Proche du résultat attendu: {'Oui' if abs(output.item() - 8) < 1 else 'Non'}")
        
    except Exception as e:
        print(f"❌ Erreur lors du forward pass: {e}")
        return
    
    # 4. Test avec différents problèmes simples
    print("\n🧮 4. Test sur problèmes mathématiques simples...")
    test_cases = [
        (2, 3, 0, 5),   # 2+3=5
        (10, 4, 1, 6),  # 10-4=6
        (3, 4, 2, 12),  # 3*4=12
        (8, 2, 3, 4),   # 8/2=4
    ]
    
    operations = ["addition", "soustraction", "multiplication", "division"]
    
    for i, (a, b, op, expected) in enumerate(test_cases):
        test_input = torch.zeros(1, 64, device=model.device)
        test_input[0, 0] = float(a)
        test_input[0, 1] = float(b)
        test_input[0, 2] = float(op)
        
        with torch.no_grad():
            output = model(test_input)
        
        result = output.item()
        correct = abs(result - expected) < 0.5
        
        print(f"   {operations[op]}: {a} {['+', '-', '*', '/'][op]} {b} = {expected}")
        print(f"   🤖 Modèle: {result:.3f} {'✅' if correct else '❌'}")
    
    # 5. Test avec le générateur de problèmes
    print("\n🎲 5. Test avec générateur de problèmes...")
    try:
        problems = problem_generator.generate_batch('math', 1, batch_size=3)
        validator = ValidationUtils()
        
        for i, problem in enumerate(problems):
            with torch.no_grad():
                output = model(problem.input_data.unsqueeze(0))
            
            result = output.item()
            is_correct = problem.validation_fn(result)
            
            print(f"   Problème {i+1}:")
            print(f"   📝 Métadonnées: {problem.metadata}")
            print(f"   🤖 Réponse modèle: {result:.3f}")
            print(f"   ✅ Correct: {'Oui' if is_correct else 'Non'}")
            
    except Exception as e:
        print(f"❌ Erreur avec le générateur: {e}")
    
    # 6. Analyse des poids du modèle
    print("\n⚖️ 6. Analyse des poids...")
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param.abs() < 1e-6).sum().item()
        
        if param.numel() < 20:  # Afficher seulement les petites couches
            print(f"   {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    print(f"\n📊 Statistiques globales:")
    print(f"   Total paramètres: {total_params}")
    print(f"   Paramètres quasi-nuls: {zero_params} ({100*zero_params/total_params:.1f}%)")
    
    # 7. Recommandations
    print("\n💡 7. Recommandations:")
    
    if zero_params / total_params > 0.9:
        print("   ⚠️ Trop de paramètres sont quasi-nuls - le modèle n'a peut-être pas appris")
        print("   🔧 Solution: Relancer l'entraînement avec un learning rate plus élevé")
    
    # Test si le modèle donne toujours la même sortie
    outputs = []
    for _ in range(5):
        test_input = torch.randn(1, 64, device=model.device)
        with torch.no_grad():
            output = model(test_input)
        outputs.append(output.item())
    
    output_variance = np.var(outputs)
    if output_variance < 1e-6:
        print("   ⚠️ Le modèle donne toujours la même sortie - il n'a pas appris la diversité")
        print("   🔧 Solution: Vérifier l'architecture et relancer l'entraînement")
    else:
        print(f"   ✅ Variance des sorties: {output_variance:.6f} (bon signe)")
    
    print("\n🏁 Diagnostic terminé!")

if __name__ == "__main__":
    diagnose_model()
