#!/usr/bin/env python3
"""
Chargeur de configuration YAML pour ThinkingIA
"""

import yaml
import torch
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config_data = None
        
        # Charger la configuration
        self.load_config()
        
        # Post-traitement automatique
        self._post_process_config()
        
        # Validation
        self._validate_config()
    
    def load_config(self):
        """Charge le fichier YAML de configuration"""
        
        if not self.config_path.exists():
            print(f"⚠️  Fichier de config {self.config_path} non trouvé")
            print("📝 Création d'un fichier de configuration par défaut...")
            self.create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            print(f"✅ Configuration chargée depuis {self.config_path}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Erreur lors du parsing YAML: {e}")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de la config: {e}")
    
    def create_default_config(self):
        """Crée un fichier de configuration par défaut"""
        
        # Copie le contenu YAML par défaut (celui d'au-dessus)
        # Pour cette démo, on va créer une version simplifiée
        
        default_config = {
            'project': {
                'name': 'ThinkingIA',
                'version': '1.0.0',
                'random_seed': 42
            },
            'model': {
                'input_dim': 128,
                'hidden_dim': 512,
                'output_dim': 64,
                'device': 'auto'
            },
            'training': {
                'learning_rate': 0.0003,
                'batch_size': 32,
                'max_epochs': 100
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, indent=2, default_flow_style=False)
        
        print(f"📄 Fichier de configuration par défaut créé: {self.config_path}")
        
        # Recharger
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config_data = yaml.safe_load(f)
    
    def _post_process_config(self):
        """Post-traitement automatique de la configuration"""
        
        # Auto-détection du device
        if self.config_data['model']['device'] == 'auto':
            self.config_data['model']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Création automatique des répertoires
        if 'experiment' in self.config_data and 'paths' in self.config_data['experiment']:
            paths = self.config_data['experiment']['paths']
            base_dir = self.config_path.parent
            
            for path_name, path_value in paths.items():
                full_path = base_dir / path_value
                full_path.mkdir(parents=True, exist_ok=True)
                # Mettre à jour avec le chemin absolu
                paths[path_name] = str(full_path)
        
        print(f"🔧 Post-traitement terminé - Device: {self.config_data['model']['device']}")
    
    def _validate_config(self):
        """Validation basique de la configuration"""
        
        try:
            # Vérifications obligatoires
            assert self.config_data['model']['input_dim'] > 0, "input_dim doit être > 0"
            assert self.config_data['model']['hidden_dim'] > 0, "hidden_dim doit être > 0"
            assert 0 < self.config_data['training']['learning_rate'] < 1, "learning_rate doit être entre 0 et 1"
            
            # Vérification de cohérence
            if 'attention_heads' in self.config_data['model']:
                hidden_dim = self.config_data['model']['hidden_dim']
                attention_heads = self.config_data['model']['attention_heads']
                assert hidden_dim % attention_heads == 0, f"hidden_dim ({hidden_dim}) doit être divisible par attention_heads ({attention_heads})"
            
            print("✅ Validation de la configuration réussie")
            
        except AssertionError as e:
            raise ValueError(f"Erreur de validation: {e}")
        except KeyError as e:
            raise ValueError(f"Paramètre manquant dans la configuration: {e}")
    
    def get(self, key_path: str, default=None):
        """
        Récupère une valeur par chemin (ex: 'model.hidden_dim')
        """
        keys = key_path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Clé '{key_path}' non trouvée dans la configuration")
    
    def __getitem__(self, key):
        """Permet l'accès direct config['model']"""
        return self.config_data[key]
    
    def __getattr__(self, name):
        """Permet l'accès par attribut config.model"""
        if name in self.config_data:
            return DictToObject(self.config_data[name])
        raise AttributeError(f"Configuration '{name}' non trouvée")

class DictToObject:
    """Convertit un dictionnaire en objet avec accès par attribut"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"

# Instance globale de configuration
config = ConfigLoader()

# Pour compatibilité avec votre code existant
def get_config():
    return config
