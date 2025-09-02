"""
Fonctions de validation et utilitaires
"""

import torch
import numpy as np
from typing import List, Dict, Callable
import matplotlib.pyplot as plt
import seaborn as sns

class ValidationUtils:
    """Utilitaires pour la validation et l'évaluation"""
    
    @staticmethod
    def evaluate_model_on_problems(model, problems: List, device: str = "cpu") -> Dict:
        """Évalue le modèle sur une liste de problèmes"""
        model.eval()
        results = {
            'accuracy': 0,
            'confidence': [],
            'reasoning_depths': [],
            'consistency_scores': [],
            'convergence_times': []
        }
        
        correct_predictions = 0
        total_problems = len(problems)
        
        with torch.no_grad():
            for problem in problems:
                input_data = problem.input_data.unsqueeze(0).to(device)
                output = model(input_data)
                
                # Vérification de la solution
                if problem.validation_fn:
                    try:
                        # Passer le tenseur complet à la fonction de validation
                        predicted_solution = output['solution']
                        if problem.validation_fn(predicted_solution):
                            correct_predictions += 1
                    except Exception as e:
                        # En cas d'erreur de validation, ne pas compter comme correct
                        pass
                
                # Collecte des métriques
                results['confidence'].append(output['final_confidence'].mean().item())
                results['reasoning_depths'].append(output['reasoning_depth'])
                results['consistency_scores'].append(output['consistency_score'].item())
                results['convergence_times'].append(output['convergence_step'])
        
        results['accuracy'] = correct_predictions / total_problems if total_problems > 0 else 0
        
        # Calcul du total_reward (cohérent avec l'entraînement)
        if results['confidence']:
            avg_confidence = sum(results['confidence']) / len(results['confidence'])
            avg_consistency = sum(results['consistency_scores']) / len(results['consistency_scores'])
            
            # Même formule que dans l'entraînement
            results['total_reward'] = max(0.05, (avg_confidence + avg_consistency) / 2)
        else:
            results['total_reward'] = 0.05
        
        return results
    
    @staticmethod
    def analyze_reasoning_traces(traces: List[Dict]) -> Dict:
        """Analyse les traces de raisonnement"""
        analysis = {
            'avg_steps': 0,
            'confidence_evolution': [],
            'attention_patterns': [],
            'convergence_patterns': []
        }
        
        if not traces:
            return analysis
        
        total_steps = sum(len(trace.get('reasoning_trace', [])) for trace in traces)
        analysis['avg_steps'] = total_steps / len(traces)
        
        # Analyse de l'évolution de la confiance
        for trace in traces:
            if 'reasoning_trace' in trace:
                confidences = [step.get('step_confidence', torch.tensor(0)).mean().item() 
                             for step in trace['reasoning_trace']]
                analysis['confidence_evolution'].append(confidences)
        
        return analysis
    
    @staticmethod
    def create_performance_visualizations(metrics_history: Dict, save_path: str = None):
        """Crée des visualisations des performances"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss over time
        axes[0, 0].plot(metrics_history.get('loss', []))
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Reward over time
        axes[0, 1].plot(metrics_history.get('reward', []))
        axes[0, 1].set_title('Training Reward')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reward')
        
        # Consistency over time
        axes[1, 0].plot(metrics_history.get('consistency', []))
        axes[1, 0].set_title('Consistency Score')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Consistency')
        
        # Reasoning depth over time
        axes[1, 1].plot(metrics_history.get('reasoning_depth', []))
        axes[1, 1].set_title('Average Reasoning Depth')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Depth')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_complexity_report(trainer, problem_generator, max_complexity: int = None) -> Dict:
        """Génère un rapport de performance par complexité"""
        from ..config import config
        if max_complexity is None:
            max_complexity = config.training.max_complexity
            
        report = {}
        
        for complexity in range(1, max_complexity + 1):
            for domain in problem_generator.get_available_domains():
                try:
                    # Génère des problèmes de test
                    test_problems = problem_generator.generate_batch(
                        domain, complexity, batch_size=20
                    )
                    
                    # Évalue le modèle
                    results = ValidationUtils.evaluate_model_on_problems(
                        trainer.model, test_problems, trainer.model.device
                    )
                    
                    report[f"{domain}_complexity_{complexity}"] = results
                    
                except Exception as e:
                    report[f"{domain}_complexity_{complexity}"] = {
                        'error': str(e),
                        'accuracy': 0
                    }
        
        return report
    
    @staticmethod
    def compare_model_versions(models: Dict[str, torch.nn.Module], 
                             test_problems: List, 
                             device: str = "cpu") -> Dict:
        """Compare différentes versions du modèle"""
        comparison = {}
        
        for model_name, model in models.items():
            results = ValidationUtils.evaluate_model_on_problems(
                model, test_problems, device
            )
            comparison[model_name] = results
        
        return comparison
    
    @staticmethod
    def save_model_checkpoint(model, optimizer, metrics, filepath: str):
        """Sauvegarde un checkpoint du modèle"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'max_reasoning_steps': model.max_reasoning_steps
            }
        }
        
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_model_checkpoint(model, optimizer, filepath: str) -> Dict:
        """Charge un checkpoint du modèle"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('metrics', {})
