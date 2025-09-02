"""
Script utilitaire pour gérer le modèle courant
"""

import os
import sys
import glob
import shutil
from datetime import datetime

CURRENT_MODEL_PATH = "current_best_model.pt"

def list_available_models():
    """Liste tous les modèles disponibles"""
    print("📁 === MODÈLES DISPONIBLES ===")
    print("=" * 50)
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ Aucun dossier 'results' trouvé")
        return []
    
    model_files = glob.glob(os.path.join(results_dir, "*", "best_model.pt"))
    
    if not model_files:
        print("❌ Aucun modèle trouvé")
        return []
    
    # Trier par date de modification
    model_files.sort(key=lambda x: os.path.getmtime(x))
    
    models_info = []
    for i, model_path in enumerate(model_files):
        model_dir = os.path.dirname(model_path)
        experiment_name = os.path.basename(model_dir)
        modification_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        models_info.append({
            'index': i,
            'path': model_path,
            'experiment': experiment_name,
            'modified': modification_time,
            'size_mb': size_mb
        })
        
        marker = "🌟 [PLUS RÉCENT]" if i == len(model_files) - 1 else ""
        print(f"{i+1:2d}. {experiment_name}")
        print(f"    📅 {modification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📦 {size_mb:.2f} MB {marker}")
        print()
    
    return models_info

def update_current_model(model_index=None):
    """Met à jour le modèle courant"""
    models = list_available_models()
    
    if not models:
        return False
    
    if model_index is None:
        # Utiliser le plus récent par défaut
        selected_model = models[-1]
        print(f"🎯 Sélection automatique du modèle le plus récent:")
    else:
        if model_index < 0 or model_index >= len(models):
            print(f"❌ Index invalide: {model_index}")
            return False
        selected_model = models[model_index]
        print(f"🎯 Modèle sélectionné:")
    
    print(f"   📁 {selected_model['experiment']}")
    print(f"   📅 {selected_model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Copier le modèle
        shutil.copy2(selected_model['path'], CURRENT_MODEL_PATH)
        print(f"✅ Modèle copié vers {CURRENT_MODEL_PATH}")
        
        # Afficher l'état actuel
        print(f"\n📋 Modèle courant mis à jour:")
        print(f"   📂 Source: {selected_model['path']}")
        print(f"   🎯 Courant: {CURRENT_MODEL_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la copie: {e}")
        return False

def show_current_model():
    """Affiche des infos sur le modèle courant"""
    print("🎯 === MODÈLE COURANT ===")
    print("=" * 30)
    
    if os.path.exists(CURRENT_MODEL_PATH):
        modification_time = datetime.fromtimestamp(os.path.getmtime(CURRENT_MODEL_PATH))
        size_mb = os.path.getsize(CURRENT_MODEL_PATH) / (1024 * 1024)
        
        print(f"✅ Modèle courant: {CURRENT_MODEL_PATH}")
        print(f"📅 Dernière modification: {modification_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📦 Taille: {size_mb:.2f} MB")
    else:
        print(f"❌ Aucun modèle courant trouvé: {CURRENT_MODEL_PATH}")

def main():
    """Interface en ligne de commande"""
    if len(sys.argv) < 2:
        print("📖 === GESTIONNAIRE DE MODÈLE ===")
        print("Usage:")
        print("  python manage_model.py list       - Liste les modèles disponibles")
        print("  python manage_model.py update     - Met à jour avec le plus récent")
        print("  python manage_model.py update N   - Met à jour avec le modèle N")
        print("  python manage_model.py current    - Affiche le modèle courant")
        print()
        show_current_model()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_available_models()
    elif command == "update":
        if len(sys.argv) > 2:
            try:
                model_index = int(sys.argv[2]) - 1  # Interface 1-based
                update_current_model(model_index)
            except ValueError:
                print("❌ Index invalide")
        else:
            update_current_model()  # Plus récent
    elif command == "current":
        show_current_model()
    else:
        print(f"❌ Commande inconnue: {command}")

if __name__ == "__main__":
    main()
