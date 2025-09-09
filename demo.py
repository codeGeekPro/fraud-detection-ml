#!/usr/bin/env python3
"""
Démonstration rapide du pipeline de détection de fraudes.

Ce script montre comment utiliser le script principal run_all.py
pour exécuter différentes parties du pipeline.
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("⚠️  Erreurs :", result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} - SUCCÈS")
        else:
            print(f"❌ {description} - ÉCHEC (code: {result.returncode})")

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {str(e)}")
        return False

def get_user_choice(prompt, default='n'):
    """Demande un choix à l'utilisateur avec une valeur par défaut."""
    try:
        choice = input(prompt).lower().strip()
        return choice if choice in ['y', 'n'] else default
    except EOFError:
        # En cas d'EOF (comme avec echo), utiliser la valeur par défaut
        print(f"{prompt}{default} (défaut)")
        return default

def main():
    """Démonstration du pipeline."""
    print("🎯 DÉMONSTRATION DU PIPELINE DE DÉTECTION DE FRAUDES")
    print("="*60)

    # Vérification de l'environnement
    success = run_command(
        "python scripts/run_all.py --check",
        "VÉRIFICATION DE L'ENVIRONNEMENT"
    )

    if not success:
        print("\n❌ Environnement non prêt. Veuillez corriger les problèmes ci-dessus.")
        sys.exit(1)

    # Entraînement (optionnel - peut prendre du temps)
    train_choice = get_user_choice("\n🤔 Voulez-vous lancer l'entraînement ? (y/N) : ")
    if train_choice == 'y':
        run_command(
            "python scripts/run_all.py --train",
            "ENTRAÎNEMENT DES MODÈLES"
        )

    # Évaluation
    eval_choice = get_user_choice("\n🤔 Voulez-vous lancer l'évaluation ? (y/N) : ")
    if eval_choice == 'y':
        run_command(
            "python scripts/run_all.py --evaluate",
            "ÉVALUATION DES MODÈLES"
        )

    # API
    api_choice = get_user_choice("\n🤔 Voulez-vous lancer l'API ? (y/N) : ")
    if api_choice == 'y':
        print("\n🌐 Lancement de l'API...")
        print("📡 L'API sera accessible sur : http://localhost:8000")
        print("📖 Documentation : http://localhost:8000/docs")
        print("💡 Appuyez sur Ctrl+C dans le terminal pour arrêter l'API")

        try:
            subprocess.run("python scripts/run_all.py --api", shell=True)
        except KeyboardInterrupt:
            print("\n🛑 API arrêtée")

    print("\n" + "="*60)
    print("🎉 DÉMONSTRATION TERMINÉE !")
    print("="*60)
    print("\n📚 Commandes utiles :")
    print("  • Pipeline complet : python scripts/run_all.py --all")
    print("  • Avec API : python scripts/run_all.py --all --api")
    print("  • Aide : python scripts/run_all.py --help")

if __name__ == "__main__":
    main()
