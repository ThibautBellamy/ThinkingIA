# test_ollama.py
import ollama

def tester_ollama():
    try:
        # Test simple
        response = ollama.generate(
            model='llama3.1',
            prompt='Bonjour ! Peux-tu me dire en une phrase qui tu es ?'
        )
        
        print("✅ Ollama fonctionne !")
        print(f"Réponse: {response['response']}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    tester_ollama()

