# fine_tuner_generation.py
import yaml
import json
import sys
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ReasoningDataset(Dataset):
    """
    Dataset pour l'entraînement à la génération de raisonnement
    """
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Concaténer input + output pour l'entraînement génératif
        full_text = f"{example['input']}\n{example['output']}<|endoftext|>"
        
        # Tokenisation
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Pour la génération de texte, labels = input_ids
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # Même chose que input_ids pour génération
        }

class ReasoningFineTuner:
    """
    Fine-tuner spécialisé pour la génération de raisonnement
    """
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        model_name = self.config['model']['name']
        
        # Tokenizer et modèle génératif
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ajouter le pad_token si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"📚 Modèle génératif chargé : {model_name}")
    
    def load_reasoning_dataset(self, dataset_path):
        """
        Charge le dataset d'énigmes logiques
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Dataset chargé : {len(data)} exemples de raisonnement")
        
        # Split train/validation
        train_data, val_data = train_test_split(
            data, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        return train_data, val_data
    
    def create_datasets(self, train_data, val_data):
        """
        Crée les datasets PyTorch
        """
        max_length = self.config['tokenizer']['max_length']
        
        train_dataset = ReasoningDataset(train_data, self.tokenizer, max_length)
        val_dataset = ReasoningDataset(val_data, self.tokenizer, max_length)
        
        return train_dataset, val_dataset
    
    def fine_tune(self, dataset_path):
        """
        Lance le fine-tuning pour génération de raisonnement
        """
        # Chargement des données
        train_data, val_data = self.load_reasoning_dataset(dataset_path)
        train_dataset, val_dataset = self.create_datasets(train_data, val_data)
        
        # Configuration d'entraînement
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            warmup_steps=training_config['warmup_steps'],
            weight_decay=training_config['weight_decay'],
            logging_dir=training_config['logging_dir'],
            logging_steps=training_config['logging_steps'],
            eval_strategy=training_config['eval_strategy'],
            eval_steps=training_config['eval_steps'],
            save_strategy=training_config['save_strategy'],
            save_steps=training_config['save_steps'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            seed=training_config['seed'],
            report_to=None,  # Désactiver wandb
            prediction_loss_only=True,  # Important pour génération
        )
        
        # Data collator pour génération de texte
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Pas de masked language modeling
            pad_to_multiple_of=8
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("🚀 Début du fine-tuning génératif...")
        trainer.train()
        
        print("💾 Sauvegarde du modèle...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_config['output_dir'])
        
        print("✅ Fine-tuning terminé !")

def main():
    if len(sys.argv) != 3:
        print("Usage: python fine_tuner_generation.py <config.yaml> <dataset.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    dataset_path = sys.argv[2]
    
    fine_tuner = ReasoningFineTuner(config_path)
    fine_tuner.fine_tune(dataset_path)

if __name__ == "__main__":
    main()
