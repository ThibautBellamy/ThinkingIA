# fine_tuner.py
import yaml
import json
import sys
import os
from transformers import (
    CamembertForSequenceClassification, 
    CamembertTokenizer,
    TrainingArguments, 
    Trainer
)
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

class GenericTextDataset(Dataset):
    """
    Dataset PyTorch générique pour tout type de classification de texte
    """
    def __init__(self, texts, labels, tokenizer, config):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
        # Configuration du tokenizer depuis le YAML
        self.max_length = config.get('max_length', 128)
        self.truncation = config.get('truncation', True)
        self.padding = config.get('padding', 'max_length')
        self.return_tensors = config.get('return_tensors', 'pt')
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CamembertFineTunerGeneric:
    """
    Fine-tuner générique configurable via YAML
    """
    def __init__(self, config_path):
        print(f"🔧 Chargement de la configuration depuis {config_path}")
        
        # Chargement de la configuration YAML
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Validation de la config
        self._validate_config()
        
        # Initialisation du modèle et tokenizer
        model_name = self.config['model']['name']
        num_labels = self.config['model']['num_labels']
        
        print(f"📚 Chargement du modèle : {model_name}")
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        print(f"✅ Modèle chargé avec {num_labels} classes")
    
    def _validate_config(self):
        """Validation basique de la configuration"""
        required_sections = ['model', 'training', 'tokenizer', 'data']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Section manquante dans la config: {section}")
    
    def charger_dataset(self, dataset_path):
        """
        Charge un dataset générique depuis un fichier JSON
        Format attendu: [{"text": "...", "label": 0}, ...]
        """
        print(f"📊 Chargement du dataset depuis {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # Configuration du split train/val
        data_config = self.config['data']
        test_size = data_config.get('test_size', 0.2)
        random_state = data_config.get('random_state', 42)
        stratify = data_config.get('stratify', True)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels if stratify else None
        )
        
        print(f"📈 Dataset divisé: {len(train_texts)} train, {len(val_texts)} validation")
        
        # Affichage des statistiques
        unique_labels = set(labels)
        print(f"🏷️  Classes détectées: {len(unique_labels)} ({sorted(unique_labels)})")
        
        return train_texts, val_texts, train_labels, val_labels
    
    def compute_metrics(self, eval_pred):
        """
        Calcule les métriques configurables
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Métriques configurables
        metrics_config = self.config.get('metrics', {})
        
        results = {}
        
        if metrics_config.get('accuracy', True):
            results['accuracy'] = accuracy_score(labels, predictions)
        
        if metrics_config.get('f1', True):
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            results.update({
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
        
        return results
    
    def create_training_arguments(self):
        """
        Crée les arguments d'entraînement depuis la config
        """
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config.get('output_dir', './camembert-finetuned'),
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 64),
            warmup_steps=training_config.get('warmup_steps', 500),
            weight_decay=training_config.get('weight_decay', 0.01),
            logging_dir=training_config.get('logging_dir', './logs'),
            logging_steps=training_config.get('logging_steps', 100),
            eval_strategy=training_config.get('eval_strategy', 'steps'),  # ✅ CHANGÉ
            eval_steps=training_config.get('eval_steps', 500),
            save_strategy=training_config.get('save_strategy', 'steps'),
            save_steps=training_config.get('save_steps', 1000),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'accuracy'),
            greater_is_better=training_config.get('greater_is_better', True),
            report_to=training_config.get('report_to', None),
            seed=training_config.get('seed', 42),
        )

    
    def fine_tuner(self, dataset_path):
        """
        Lance le fine-tuning avec la configuration YAML
        """
        # Chargement du dataset
        train_texts, val_texts, train_labels, val_labels = self.charger_dataset(dataset_path)
        
        # Création des datasets PyTorch
        print("🔧 Création des datasets PyTorch...")
        tokenizer_config = self.config['tokenizer']
        
        train_dataset = GenericTextDataset(
            train_texts, train_labels, self.tokenizer, tokenizer_config
        )
        val_dataset = GenericTextDataset(
            val_texts, val_labels, self.tokenizer, tokenizer_config
        )
        
        # Arguments d'entraînement
        training_args = self.create_training_arguments()
        
        # Création du trainer
        print("🔧 Initialisation du Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Lancement de l'entraînement
        print("🚀 Début du fine-tuning...")
        print(f"📊 Configuration utilisée:")
        print(f"   - Modèle: {self.config['model']['name']}")
        print(f"   - Époques: {self.config['training']['num_train_epochs']}")
        print(f"   - Batch size: {self.config['training']['per_device_train_batch_size']}")
        print(f"   - Max length: {self.config['tokenizer']['max_length']}")
        
        trainer.train()
        
        # Sauvegarde
        output_dir = self.config['training']['output_dir']
        print(f"💾 Sauvegarde du modèle dans {output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Sauvegarde de la config utilisée
        with open(f"{output_dir}/config_used.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print("✅ Fine-tuning terminé avec succès !")
        return trainer

def main():
    if len(sys.argv) != 3:
        print("Usage: python fine_tuner.py <config.yaml> <dataset.json>")
        print("Exemple: python fine_tuner.py config_questions.yaml dataset_questions_affirmations.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    
    # Vérification des fichiers
    if not os.path.exists(config_path):
        print(f"❌ Fichier de config non trouvé: {config_path}")
        sys.exit(1)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset non trouvé: {dataset_path}")
        sys.exit(1)
    
    try:
        fine_tuner = CamembertFineTunerGeneric(config_path)
        fine_tuner.fine_tuner(dataset_path)
    except Exception as e:
        print(f"❌ Erreur lors du fine-tuning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
