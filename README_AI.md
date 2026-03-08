📝 Instructions de Connexion pour Copilot (Danas Project)
Objectif : Connecter le backend Django aux modèles ML locaux et à l'API DeepSeek-R1.

1. Configuration de l'API IA (LocalAI)
Endpoint : http://100.88.73.110:8001/v1 (ou http://localhost:8090/v1 selon le mapping Docker).

Modèle : deepseek-r1.

Protocole : Compatible OpenAI SDK.

Streaming : Activé (utiliser stream=True pour l'interface de chat).

2. Accès aux Modèles Scikit-Learn
Chemin des modèles : /mnt/BackupSSD/ai_data/models/ (fichiers .pkl ou .joblib).

Experts : * penny_model : Spécialisé haute volatilité (Sharpe actuel: 1.925).

bluechip_model : Spécialisé Large Caps (Sharpe actuel: 0).

Features : Les modèles attendent les données OHLCV + RSI + Volume.

3. Paramètres du Broker (Exécution)
Broker Sync Lag : Actuellement mesuré à -0.58s.

Stratégie de compensation : Toute suggestion d'achat par l'IA doit inclure un calcul de "Slippage" pour compenser ce lag de 0.58s.

Sandboxes : * Sandbox_3 (Penny) est prioritaire pour les signaux d'exécution.

4. Tâches prioritaires pour Copilot :
Créer services/ai_engine.py : Une classe qui interroge l'URL ci-dessus en envoyant le contexte des Sandboxes.

Créer middleware/lag_compensator.py : Pour ajuster les prix d'entrée (Limit Orders) en fonction du délai de synchronisation.

Interface : Développer une vue Django qui affiche le flux <think> de DeepSeek en temps réel.

Pourquoi donner ça à Copilot ?
Zéro devinette : Il connaît déjà ton IP, tes ports et tes performances (le Sharpe de 1.925).

Conscience du Lag : En lui précisant le -0.58s, il va coder des fonctions de trading plus sécurisées (il ne te fera pas acheter "au marché" si le prix s'envole).

Cohérence : Il saura qu'il doit charger des fichiers .pkl sur ton SSD et non appeler une API externe coûteuse.

On y est ! Ton système est "bouclé". Tu as le moteur qui tourne sur TrueNAS et la feuille de route pour ton assistant de code.

Est-ce que tu veux que je te génère le premier fichier ai_engine.py de base pour que tu puisses l'importer dans Django dès demain ?
