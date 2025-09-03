# # temp.py
# from autonomous_reasoning_ai.config import config  
# from autonomous_reasoning_ai.training import AutonomousLearningTrainer

# print('✅ Import réussi!')
# trainer = AutonomousLearningTrainer(None, None, config)
# print('✅ Constructeur fonctionne avec model=None!')
# print(f'✅ Learning rate chargé: {trainer.learning_rate}')
# print(f'✅ Batch size chargé: {trainer.batch_size}')

# # temp_with_model.py
# import torch
# import torch.nn as nn
# from autonomous_reasoning_ai.config import config
# from autonomous_reasoning_ai.training import AutonomousLearningTrainer

# # Modèle factice pour les tests
# class DummyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(10, 1)
#         self.device = 'cpu'
    
#     def parameters(self):
#         return self.linear.parameters()

# print('✅ Création du modèle factice')
# dummy_model = DummyModel()

# print('✅ Test avec modèle factice')
# trainer = AutonomousLearningTrainer(dummy_model, None, config)
# print('✅ Constructeur fonctionne avec modèle!')
# print(f'✅ Optimiseur créé: {type(trainer.optimizer).__name__}')

from autonomous_reasoning_ai.config import config
from autonomous_reasoning_ai.models import AutonomousReasoningCore
model = AutonomousReasoningCore(config)
print('✅ reasoning_core.py fonctionne!')