"""
Cœur du système de raisonnement autonome
"""

from venv import logger
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ..config import config

class AutonomousReasoningCore(nn.Module):
    """Cœur du système de raisonnement autonome"""
    
    def __init__(self):
        super().__init__()
        self.config = config

        # ✅ CONVERSION EXPLICITE EN INT
        self.input_dim = int(config.get('model.input_dim', 128))
        self.hidden_dim = int(config.get('model.hidden_dim', 512))
        self.max_reasoning_steps = int(config.get('model.max_reasoning_steps', 5))
        self.dropout_rate = float(config.get('model.dropout_rate', 0.1))
      
        # Validation des valeurs
        assert isinstance(self.input_dim, int), f"input_dim doit être int, reçu: {type(self.input_dim)}"
        assert isinstance(self.hidden_dim, int), f"hidden_dim doit être int, reçu: {type(self.hidden_dim)}"
        assert self.input_dim > 0, f"input_dim doit être > 0, reçu: {self.input_dim}"
        assert self.hidden_dim > 0, f"hidden_dim doit être > 0, reçu: {self.hidden_dim}"
        
        print("✅ Validation des dimensions réussie !")
        
        # Bloc de raisonnement récurrent (innovation clé)
        self.reasoning_lstm = nn.LSTM(
            self.hidden_dim, 
            self.hidden_dim, 
            batch_first=True
            # Pas de dropout pour LSTM à 1 couche (évite le warning)
        )
        
        # Dropout séparé pour la régularisation
        self.reasoning_dropout = nn.Dropout(self.dropout_rate)
        
        # Contrôleur de profondeur de raisonnement
        self.reasoning_depth_controller = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Mécanisme d'attention pour le raisonnement
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=8, 
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Décodeur vers solution
        self.solution_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        # Auto-critique interne
        self.self_critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Mémoire de travail
        self.working_memory = nn.Parameter(torch.randn(1, 10, self.hidden_dim))
    
    @property
    def device(self):
        """Retourne le device du modèle"""
        return next(self.parameters()).device
        
    def forward(self, problem_encoding: torch.Tensor, max_reasoning_steps: int = None) -> Dict:
        """Forward pass avec raisonnement itératif"""
        
        max_steps = max_reasoning_steps or self.max_reasoning_steps
        batch_size = problem_encoding.size(0)
        
        # Encodage initial du problème
        encoded = self.input_encoder(problem_encoding)
        
        # Initialisation de la mémoire de travail
        memory = self.working_memory.expand(batch_size, -1, -1)
        
        # Raisonnement itératif jusqu'à convergence
        reasoning_states = []
        current_state = encoded.unsqueeze(1)
        
        # États cachés pour LSTM
        hidden = None
        
        for step in range(max_steps):
            # Étape de raisonnement avec LSTM
            reasoning_output, hidden = self.reasoning_lstm(current_state, hidden)
            
            # Application du dropout pour la régularisation
            reasoning_output = self.reasoning_dropout(reasoning_output)
            
            # Attention avec la mémoire de travail
            attended_memory, attention_weights = self.attention(
                reasoning_output, memory, memory
            )
            
            # Fusion du raisonnement et de la mémoire
            fused_reasoning = reasoning_output + attended_memory
            
            # Le modèle décide s'il doit continuer à raisonner
            continue_reasoning = self.reasoning_depth_controller(
                fused_reasoning[:, -1, :]
            )
            
            # Auto-évaluation de cette étape
            step_confidence = self.self_critic(fused_reasoning[:, -1, :])
            
            reasoning_states.append({
                'state': fused_reasoning,
                'continue_prob': continue_reasoning,
                'step_confidence': step_confidence,
                'attention_weights': attention_weights,
                'step': step
            })
            
            # Arrêt si suffisamment confiant (seuil adaptatif)
            if continue_reasoning.mean() < 0.3:
                break
                
            # Mise à jour de l'état pour la prochaine itération
            current_state = fused_reasoning
            
            # Mise à jour de la mémoire de travail
            memory = memory + 0.1 * attended_memory
        
        # Solution finale basée sur le dernier état de raisonnement
        final_reasoning_state = reasoning_states[-1]['state'][:, -1, :]
        final_solution = self.solution_decoder(final_reasoning_state)
        
        # Auto-évaluation finale
        final_confidence = self.self_critic(final_reasoning_state)
        
        # Calcul de la consistance du raisonnement
        consistency_score = self._compute_consistency(reasoning_states)
        
        return {
            'solution': final_solution,
            'reasoning_trace': reasoning_states,
            'final_confidence': final_confidence,
            'reasoning_depth': len(reasoning_states),
            'consistency_score': consistency_score,
            'convergence_step': step
        }
    
    def _compute_consistency(self, reasoning_states: List[Dict]) -> torch.Tensor:
        """Calcule la consistance du processus de raisonnement"""
        if len(reasoning_states) < 2:
            return torch.tensor(1.0)
        
        # Variance des confiances entre les étapes
        confidences = torch.stack([state['step_confidence'].squeeze() 
                                 for state in reasoning_states])
        consistency = 1.0 / (1.0 + torch.var(confidences, dim=0))
        
        return consistency.mean()
    
    def get_reasoning_summary(self, output: Dict) -> Dict:
        """Génère un résumé du processus de raisonnement"""
        trace = output['reasoning_trace']
        
        summary = {
            'total_steps': output['reasoning_depth'],
            'convergence_step': output['convergence_step'],
            'final_confidence': output['final_confidence'].mean().item(),
            'consistency_score': output['consistency_score'].item(),
            'step_confidences': [state['step_confidence'].mean().item() 
                               for state in trace],
            'continue_probabilities': [state['continue_prob'].mean().item() 
                                     for state in trace]
        }
        
        return summary
