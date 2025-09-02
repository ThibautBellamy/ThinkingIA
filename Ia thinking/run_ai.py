#!/usr/bin/env python3
"""
Script de lancement pour l'IA de raisonnement autonome
"""

import sys
import os

# Ajouter le répertoire courant au path Python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import et lancement du module principal
if __name__ == "__main__":
    from autonomous_reasoning_ai.main import main
    main()
