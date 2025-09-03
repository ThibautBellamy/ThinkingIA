#!/usr/bin/env python3
"""
Test du système de configuration YAML
"""

from autonomous_reasoning_ai.config import config

def test_config():
    """Test de base de la configuration"""
    
    print("🧪 Test du système de configuration YAML")
    print("=" * 50)
    
    # Test d'accès par dictionnaire
    print(f"Nom du projet: {config['project']['name']}")
    
    # Test d'accès par attribut
    print(f"Device: {config.model.device}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Test de la méthode get()
    batch_size = config.get('training.batch_size', 16)
    print(f"Batch size: {batch_size}")
    
    # Test avec une clé qui n'existe pas
    try:
        inexistant = config.get('inexistant.key')
    except KeyError as e:
        print(f"✅ Gestion d'erreur OK: {e}")
    
    print("=" * 50)
    print("✅ Tous les tests passés !")

if __name__ == "__main__":
    test_config()
