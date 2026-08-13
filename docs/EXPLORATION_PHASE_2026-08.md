# Phase d'exploration temporaire — WATCHLIST / AI_BLUECHIP / AI_PENNY

**Démarrée le 2026-08-12.** Objectif unique : générer ≥20 trades FERMÉS par
sandbox pour sortir du verrou circulaire de ré-entraînement documenté dans
`docs/ML_PIPELINE_AUDIT.md` (le réentraînement automatique exige ≥20 trades
fermés pour promouvoir un nouveau modèle, mais aucun trade ne se déclenche
jamais pour en générer). **Ce n'est pas une nouvelle stratégie permanente.**

Rythme cible : 5-15 nouveaux trades/semaine, toutes sandboxes confondues.
Taille de position : $1000 fixe pendant cette phase (voir section dédiée).
Durée : jusqu'à ≥20 trades fermés par sandbox, pas de date limite fixe.

**Règle de fin de phase, non négociable** : une fois ≥20 trades fermés
atteints pour un sandbox, **ne pas resserrer automatiquement** — revenir
discuter avec l'utilisateur pour décider de la recalibration en fonction
des résultats réels observés.

---

## Étape 0 — Bugs corrigés AVANT tout assouplissement (ordre appliqué)

Chaque fix testé et déployé séparément avant le suivant, voir historique
git pour le détail complet de chacun :

1. **`_env_list`** (`portfolio/tasks.py`) — variable d'env présente-mais-vide
   (`AI_BLUECHIP_WATCHLIST=`) ne retombait jamais sur le défaut. Corrigé
   avec une cascade `or` qui traite une chaîne vide comme non définie.
   Confirmé : `_get_watchlist('AI_BLUECHIP', ...)` passe de `[]` à 26 tickers.
2. **`refresh_ai_bluechip_watchlist`** — timeout HTTP de 60s alors que
   `/api/ai/opportunities/` prend réellement ~112s (confirmé en direct).
   Timeout relevé à 180s (configurable via `AI_BLUECHIP_REFRESH_TIMEOUT_S`).
3. **Alias de features WATCHLIST** (`feature_registry.py`) — pointait vers
   `BLUECHIP_FEATURE_NAMES` (27 features, jamais utilisée par aucun
   pipeline réel) au lieu de `FUSION_FEATURE_NAMES` (54 features,
   réellement utilisée par le chemin de ré-entraînement). Corrigé, test de
   régression ajouté.
4. **`PurgedTimeSeriesSplit`** branché sur `retrain_from_paper_trades_daily`
   (remplace `_split_time`'s coupure chronologique brute par un split avec
   fenêtre de purge + embargo). Infrastructure déjà validée par le chemin
   manuel (`manage.py ml_scanner`), jamais branchée sur ce chemin
   automatique avant ce fix. Fait explicitement AVANT l'assouplissement des
   seuils pour que les premiers trades d'exploration soient validés par la
   méthode la plus solide dès le départ.

---

## Étape 1 — Seuils assouplis, calibrés à partir de données réelles

**Limite de données rencontrée, à noter honnêtement** : la demande
initiale visait 90-180 jours d'historique de signal. En pratique :
- `SystemLog` a une politique de rétention (voir item 6,
  `TECH_DEBT_NOTES.md`) — la fenêtre de données réellement disponible
  pour WATCHLIST/AI_PENNY s'est avérée être ~2-3 jours de densité de logs,
  pas 90 jours pleins.
- AI_BLUECHIP n'avait **aucune** donnée historique du tout (son chemin SIM
  scannait 0 ticker jusqu'au fix de l'étape 0.1) — calibré sur un seul
  scan réel déclenché manuellement juste après le fix, pas un historique.

Ces limites sont documentées ici plutôt que cachées derrière une fausse
précision.

### WATCHLIST — non calibrable via seuil, documenté tel que prévu par le prompt

Les 79 snapshots disponibles montrent `final.max = 0.0` **dans 100% des
cas**, sans exception — cohérent avec le diagnostic de la Partie C
(formule RSI structurellement plafonnée à 0 dès RSI≥30). Aucun seuil,
même à 0.30, n'aurait produit de déclenchement historiquement. Seuil
**laissé inchangé à 0.55** — l'abaisser ne changerait rien tant que la
formule elle-même n'est pas révisée, ce qui est explicitement hors scope
de cette phase (décision de stratégie, pas un bug). **WATCHLIST ne
contribuera probablement pas au rythme cible pendant cette phase** —
AI_BLUECHIP et AI_PENNY devront porter le volume.

### AI_BLUECHIP — calibré sur un scan réel post-fix (donnée unique, pas un historique)

Scan déclenché manuellement le 2026-08-12 juste après le fix 0.1 (premier
scan avec watchlist non vide) : `n=4` signaux valides sur 26 tickers,
`max=0.7545, mean=0.3341, median=0.2352`. Seuil réel avant ce fix : 0.85
(SIM) / ~0.55 (Alpaca, via repli sur `PAPER_BUY_THRESHOLD`, jamais
explicitement configuré séparément — asymétrie découverte pendant ce
travail, pas corrigée ici, notée pour référence future).

**`AI_BLUECHIP_BUY_THRESHOLD` : 0.85 → 0.55** (`deploy/.env`, backup dans
`deploy/.env.pre_exploration_phase_backup_20260812`). Choix : aligné sur
`PAPER_BUY_THRESHOLD` (déjà en usage réel pour WATCHLIST et tous les
chemins Alpaca) plutôt que d'inventer un chiffre bespoke avec une
précision qu'un seul scan ne justifie pas. Comfortablement sous le max
observé (0.75), bien au-dessus de la médiane (0.24).

### AI_PENNY — calibré sur historique réel (81 snapshots, période antérieure à un changement de seuil)

Historique montrant un point de bascule net : seuil 0.55 → 75% des runs
avaient ≥1 signal au-dessus (max observés groupés 0.56-0.57) ; seuil
0.60 → 0%. Seuil réel actuel en prod : 0.82 (bien au-delà de ce point de
bascule, confirmé nul historiquement).

**`AI_PENNY_BUY_THRESHOLD` : 0.82 → 0.55** (même fichier/backup). Même
raisonnement d'alignement sur `PAPER_BUY_THRESHOLD` que AI_BLUECHIP —
choix délibéré d'un seul chiffre cohérent plutôt que 3 valeurs différentes
mal justifiées par des données de qualité inégale entre sandboxes.

### Fréquences réelles d'exécution (pour interpréter le rythme observé)

| Sandbox / chemin | Fréquence | Runs/semaine (heures de marché) |
|---|---|---|
| SIM (AI_BLUECHIP, AI_PENNY) | */15min, 9h-16h ET, lun-ven | ~140 |
| Alpaca (WATCHLIST, AI_BLUECHIP, AI_PENNY) | */2min, 9h-16h ET, lun-ven | ~1050 |

Ces fréquences sont élevées — un seuil légèrement trop permissif peut
faire dépasser largement la cible de 5-15/semaine. D'où le choix
délibérément conservateur (0.55, pas plus bas) malgré des données
suggérant qu'un seuil encore plus bas produirait "plus d'occasions" —
mieux vaut sous-viser et ajuster à la hausse après observation réelle
(étape 5) que sur-viser et devoir intervenir en urgence.

### Correction en cours de route : les éditions `.env` ci-dessus étaient inertes

Découvert en dry-runnant l'étape 2 (pas supposé) : `_execute_paper_trades_
for_sandbox` (chemin SIM) lit `{PREFIX}_BUY_THRESHOLD` via `_env_float`,
**puis l'écrase sans condition quelques lignes plus loin** avec des
valeurs codées en dur par sandbox (`AI_BLUECHIP=0.80`, `WATCHLIST=0.55`,
`AI_PENNY=0.20`). Éditer `AI_BLUECHIP_BUY_THRESHOLD`/`AI_PENNY_BUY_
THRESHOLD` dans `deploy/.env` (fait plus haut dans cette section)
**n'avait donc aucun effet réel sur ce chemin**. Confirmé en direct :
après l'édition `.env`, `SystemLog` continuait à logger
`buy_threshold=0.8` pour AI_BLUECHIP. Découverte bonus : le seuil SIM
réel d'AI_PENNY était déjà `0.20` codé en dur — la valeur `0.82` dans
`.env` n'était jamais utilisée par ce chemin, lue puis immédiatement
jetée.

**`deploy/.env` restauré à ses valeurs d'origine** (`0.85`/`0.82`,
backup toujours dans `deploy/.env.pre_exploration_phase_backup_20260812`)
— laisser un changement inerte en place aurait été trompeur. **Le vrai
mécanisme** : nouvel override explicite `EXPLORATION_PHASE_BUY_THRESHOLD`
(défaut 0.55), gated par `EXPLORATION_PHASE_ENABLED`, appliqué APRÈS le
bloc codé en dur dans le chemin SIM et après la lecture d'env dans le
chemin Alpaca (qui n'a pas d'override codé en dur, mais retombait sur
`PAPER_BUY_THRESHOLD` par coïncidence plutôt que par un choix explicite).

**Vérifié en direct après le vrai fix** : `buy_threshold effectif = 0.55`
pour WATCHLIST, AI_BLUECHIP, ET AI_PENNY simultanément (requête
`SystemLog` fraîche après un cycle complet des 3 sandboxes, mode
exploration activé) — confirmé pour de vrai cette fois, pas juste
supposé depuis une édition de fichier de config.

---

## Étape 2 — Taille de position fixe $1000, mode exploration

Nouveau flag explicite `EXPLORATION_PHASE_ENABLED` (`portfolio/tasks.py`,
`_exploration_phase_enabled()`) — `false` par défaut, aucun effet tant
qu'il n'est pas activé. `_exploration_position_value(capital_disponible)`
retourne `min(EXPLORATION_PHASE_POSITION_SIZE, capital_disponible)` —
$1000 par défaut, toujours plafonné au capital réel disponible du
sandbox (garde-fou de base demandé explicitement).

Branché aux deux points de calcul de taille de position réels (les seuls
qui créent une `PaperTrade` d'achat pour WATCHLIST/AI_BLUECHIP/AI_PENNY) :
- `_execute_paper_trades_for_sandbox` (chemin SIM, `broker='SIM'`)
- `_execute_alpaca_paper_trades_for_sandbox` (chemin Alpaca réel,
  `broker='ALPACA'`)

Quand le flag est actif, le calcul habituel (dynamic sizing, boost
multi-modèle, facteur de régime, facteur de confiance, facteur de
ré-entrée, multiplicateurs de tier) est entièrement sauté au profit du
montant fixe — pas juste une valeur par défaut modifiée, une branche
séparée et explicite (`if _exploration_phase_enabled(): ... else: ...`).

## Étape 3 — Garde-fous vérifiés actifs

Vérifié en lisant le code autour de chaque point d'insertion (pas juste
supposé) :
- **Circuit breaker** (`_daily_equity_circuit_breaker`) : appelé au tout
  début des deux fonctions (avant la boucle par symbole), bien avant tout
  code lié à l'exploration — untouched, un déclenchement retourne
  toujours immédiatement sans jamais atteindre le calcul de position.
- **Tous les `blocked_*`** (exposition, confiance, corrélation, spread,
  ATR spike, tendance BTC/secteur, blacklist de pertes, etc.) : tous
  évalués AVANT le bloc de calcul de taille modifié, aucun réordonnancement.
- **Stop-loss / sortie** : formule de `stop_distance`/`stop_loss`
  complètement inchangée, calculée après (pas avant, pas dans) le bloc
  modifié.

Seul ce qui devait changer a changé : le calcul du MONTANT à investir
une fois qu'une décision d'achat a déjà été validée par tous les
garde-fous existants.

## Étape 4 — Flag `exploration_phase`

Pas de nouveau champ DB (pas de migration) — utilise les métadonnées déjà
disponibles sur `PaperTrade`, comme demandé :
- `entry_features['exploration_phase'] = True` (champ JSON déjà utilisé
  pour stocker les features au moment de l'entrée).
- `notes` : suffixe explicite `"[EXPLORATION PHASE 2026-08 -- voir
  docs/EXPLORATION_PHASE_2026-08.md]"` ajouté sur les deux chemins
  (SIM et Alpaca), pour qu'un humain qui lit la DB directement (pas juste
  une requête JSON) voie immédiatement le contexte.

Requête pour isoler ces trades plus tard :
`PaperTrade.objects.filter(entry_features__exploration_phase=True)`.

## Étape 5 — Tests, déploiement, surveillance

**Tests unitaires** (`portfolio/tests_exploration_phase.py`) : 6 tests
sur `_exploration_phase_enabled`/`_exploration_position_value`
(désactivé par défaut, respect du flag, jamais au-dessus du capital
disponible, override respecté). Exécutés via `manage.py test` sur le
NAS : 6/6 OK. Suite existante liée aux paper trades
(`tests_paper_trade_decision_stats`, `tests_bluechip_watchlist_merge`,
`tests_after_hours_schedule_gate`) : 9/9 OK, aucune régression.

**Dry-run en conteneur jetable** (`docker compose run --rm -e
EXPLORATION_PHASE_ENABLED=true ...`) avant tout déploiement en prod —
a justement permis de trouver et corriger le bug des seuils codés en
dur (section précédente) avant qu'il n'atteigne la prod. Deuxième
dry-run après le vrai fix : `buy_threshold effectif = 0.55` confirmé
pour les 3 sandboxes. 0 trade créé sur ces runs précis (normal — dépend
des conditions de marché à l'instant T, pas un signe de problème ; le
marché était fermé au moment du test).

**Déployé en prod le 2026-08-12** : `EXPLORATION_PHASE_ENABLED=true`,
`EXPLORATION_PHASE_POSITION_SIZE=1000`,
`EXPLORATION_PHASE_BUY_THRESHOLD=0.55` dans `deploy/.env`. `backend`,
`celery_worker`, `celery_beat`, `celery_worker_analysis` reconstruits et
recréés. Vérifié en direct sur le conteneur en cours d'exécution (pas
juste supposé) : `EXPLORATION_PHASE_ENABLED=true` bien lu.

**Bonus découvert pendant les tests, hors scope, non corrigé** :
`InconsistentVersionWarning` de scikit-learn à chaque chargement de
modèle (`.pkl` entraînés avec sklearn 1.8.0, servis avec 1.9.0 installé
dans l'image actuelle). Pas un crash, mais un vrai risque de dérive de
comportement du modèle non quantifié. Noté ici pour référence future,
pas traité ce soir (hors scope de cette phase).

### À surveiller dans les prochains jours (marché fermé au moment du déploiement)

- Le rythme réel de trades approche-t-il 5-15/semaine combiné, sans
  exploser dessus (risque identifié : fréquence d'exécution très élevée,
  surtout le chemin Alpaca à */2min) ?
- Les tailles de position sont-elles bien $1000 (plafonnées au capital
  disponible) sur les trades réels créés ?
- `entry_features__exploration_phase=True` permet d'isoler ces trades à
  tout moment : `PaperTrade.objects.filter(sandbox=X,
  entry_features__exploration_phase=True)`.
- Compte des trades FERMÉS par sandbox, vers l'objectif de 20 :
  `PaperTrade.objects.filter(sandbox=X, status='CLOSED',
  entry_features__exploration_phase=True).count()`.
- WATCHLIST : confirmer qu'elle reste effectivement à ~0 trade (cohérent
  avec le diagnostic formule/RSI) plutôt que de supposer que ça reste
  vrai indéfiniment.

**Rappel non négociable** : à ≥20 trades fermés pour un sandbox donné,
ne pas resserrer automatiquement — revenir discuter avec l'utilisateur.
