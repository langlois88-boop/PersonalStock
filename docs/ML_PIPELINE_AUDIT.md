# Audit du pipeline de décision ML — WATCHLIST / AI_BLUECHIP / AI_PENNY

**Date : 2026-08-12. Audit diagnostique uniquement — aucune correction appliquée
dans cette session.** Toutes les données citées ci-dessous viennent de
vérifications en direct (code source lu ET exécuté, `SystemLog` de production,
requêtes DB réelles, fichiers `.pkl` sur le NAS) — pas de suppositions.

**Constat de départ, confirmé empiriquement avant tout diagnostic** :

```
PaperTrade créés, 90 derniers jours : 0 (WATCHLIST, AI_BLUECHIP, AI_PENNY confondus)
PaperTrade créés, 180 derniers jours : WATCHLIST=4, AI_BLUECHIP=1, AI_PENNY=4
"Paper trade" logs sur 30 jours : 678 exécutions, created=0 dans TOUS les cas
```

Les 3 sandboxes tournent en continu (aucun crash, `TaskRunLog` en SUCCESS) mais
ne produisent quasiment plus rien depuis des mois.

---

## Étape 1 — Cartographie du pipeline, sandbox par sandbox

### Tronc commun aux 3 sandboxes

Avant de diverger, WATCHLIST/AI_BLUECHIP/AI_PENNY partagent le même chemin
d'intake et de calcul du signal brut :

- **Intake** : `DataFusionEngine(symbol).fuse_all()` — agrège yfinance +
  Alpaca (si `use_alpaca=True`) selon le sandbox, calcule les indicateurs
  techniques bruts (RSI, MACD, Bollinger, ADX, volume, etc.), plus sentiment
  news et macro (FRED). Fréquence : à la demande, à chaque appel de
  `_model_signal()` (pas de cache de features observé à ce niveau).
- **Feature engineering** : registre central `ml_engine/feature_registry.py`.
  **Un seul fichier**, mais 6 listes nommées à l'intérieur (STABLE, PENNY,
  BLUECHIP, CRYPTO, FUSION, RECOMMENDER) — pas toutes réellement utilisées
  (voir "Feature registry" plus bas). Le modèle FUSION (celui réellement
  chargé pour WATCHLIST/AI_BLUECHIP/AI_PENNY) utilise `FUSION_FEATURE_NAMES`,
  **52 features** (momentum, tendance, volatilité, volume, structure de
  prix, bougies/patterns, macro, order book, fondamentaux, fractional
  differencing — liste complète dans le fichier).
- **Modèle** : `_model_signal()` (`portfolio/tasks.py:4937`) appelle
  `load_or_train_model(fusion_df, model_path)` — **charge le `.pkl` existant
  s'il existe, ne vérifie JAMAIS sa fraîcheur**. Ne ré-entraîne à la volée
  (`train_fusion_model`) que si le fichier est absent, sur les données du
  SEUL symbole en train d'être scanné (pas un vrai entraînement de fond).
- **Signal brut** : `payload['model'].predict_proba(features)[0][1]`, puis
  passé à `apply_feature_weighting_to_signal()` (pondération post-hoc, pas
  vérifiée en détail dans cet audit — hors scope immédiat).

### WATCHLIST

| | |
|---|---|
| Modèle utilisé | **`data_fusion_brain_bluechip_v1.pkl`** — partagé avec AI_BLUECHIP, WATCHLIST n'a pas son propre modèle (`universe='BLUECHIP'` en dur au call site) |
| Dernier entraînement réel | **~18-24 février 2026** (voir "Étape 3") — ~6 mois |
| `base_model_signal` → `final` | **Remplacement complet**, pas une transformation. `_signal()` (`tasks.py:5479`) écrase `payload['signal']` par `_mean_reversion_score(symbol)` — un score RSI14 + bonus Bollinger, sans aucun rapport avec `base_model_signal`. Déjà documenté dans le code par un commentaire explicite avant cette session. |
| Formule exacte | `rsi_score = max(0, (30 - rsi) / 30)` (plafonne à 0 dès RSI≥30) `+ 0.15` si prix ≤ bande de Bollinger basse |
| `buy_threshold` | **0.55** (confirmé via `SystemLog.metadata['buy_threshold']`, valeur réelle en prod) |
| Watchlist source | `SandboxWatchlist(sandbox='WATCHLIST')`, 10 tickers — **4 étaient invalides** (suffixe d'échange périmé), corrigés dans la session précédente (voir Partie C déjà close) |

**Déjà diagnostiqué et partiellement corrigé avant cet audit** (Partie C,
session précédente, réutilisé ici tel que demandé, pas refait) : cause
confirmée = formule structurellement quasi jamais au-dessus de 0.55 (il
faudrait RSI ≲ 18) + 4 tickers morts sur 10. Le nettoyage des tickers est
fait ; la calibration reste **volontairement** non touchée (décision de
stratégie, pas un bug de code).

### AI_BLUECHIP

| | |
|---|---|
| Modèle utilisé | `data_fusion_brain_bluechip_v1.pkl` (le même que WATCHLIST) |
| Dernier entraînement réel | ~18-24 février 2026 — ~6 mois |
| `base_model_signal` → `final` | Composite pondéré, PAS un remplacement : `composite = 0.2×base_signal + 0.4×fundamentals_score + 0.4×corr_score`, + bonus croissance jusqu'à +0.10 si `revenue_growth ≥ 10%` (`tasks.py:5493`) |
| `buy_threshold` | **0.85** (`AI_BLUECHIP_BUY_THRESHOLD=0.85` dans `deploy/.env`, confirmé) — extrêmement élevé pour un `predict_proba` combiné à seulement 20% de poids |
| Watchlist source, chemin SIM (`execute_paper_trades_ai_bluechip`) | **`SandboxWatchlist(sandbox='AI_BLUECHIP').symbols = []`** (vide, confirmé en DB) **ET** `AI_BLUECHIP_WATCHLIST=` (chaîne vide) dans `deploy/.env` — **bug confirmé** : `_env_list()` utilise `os.getenv(key, default)`, qui ne retombe sur le défaut QUE si la variable est absente, pas si elle est présente-mais-vide. Résultat vérifié en direct : `_get_watchlist('AI_BLUECHIP', 'AI_BLUECHIP', ...)` retourne **`[]`** — ce chemin scanne **zéro ticker**, littéralement, depuis potentiellement des mois. |
| Watchlist source, chemin Alpaca (`execute_alpaca_paper_trades_ai_bluechip`) | Retombe sur `PAPER_WATCHLIST` (26 tickers réels : SPY, AAPL, MSFT, NVDA, AMZN, RY, TD, ATD, CNR, KO, ENB, SHOP, ASML, ADBE, INTC, CSCO, VZ, T, IBM, ORCL, CRM, AVGO, TXN, QCOM, NFLX, META) — **ce chemin scanne bien des tickers réels**, donc le problème ici est le seuil (0.85), pas l'univers. |
| Mécanisme de remplissage auto (`AI_BLUECHIP_AUTOFILL=true`) | `refresh_ai_bluechip_watchlist` (Beat, ~horaire) — **exécuté en direct pendant cet audit, échoue systématiquement** : `HTTPConnectionPool(host='backend', port=8000): Read timed out (60s)` en appelant son propre endpoint interne `/api/ai/opportunities/`. Échec avalé proprement (`except Exception: return {'status': 'error', ...}`) — la tâche "réussit" du point de vue Celery (pas de crash, pas d'alerte), mais ne remplit jamais la watchlist. C'est la cause directe de la ligne DB vide. |

### AI_PENNY

| | |
|---|---|
| Modèle utilisé | `data_fusion_brain_penny_v1.pkl` (dédié, pas partagé) |
| Dernier entraînement réel | ~18-24 février 2026 — ~6 mois |
| `base_model_signal` → `final` | Composite pondéré : `weighted = base_weight×base_signal + (1-base_weight)×intraday_score` (`base_weight` par défaut 0.6), `intraday_score = 0.45×rvol_score + 0.35×sentiment_score + 0.20×breakout_score`. `composite = max(base_signal, weighted)` si contexte intraday dispo, sinon juste `base_signal` (`tasks.py:5524`) |
| `buy_threshold` | **0.82** (`AI_PENNY_BUY_THRESHOLD=0.82`, confirmé) — également très élevé |
| Watchlist source | `SandboxWatchlist(sandbox='AI_PENNY').symbols` = **13 tickers réels** (UP, HAIN, MXCT, ARAY, SLQT, BRCC, ADV, NOTE, CANG, TEAD, NDLS, PSQH, SONI.CN) — **confirmée non vide**, contrairement à AI_BLUECHIP. `AI_PENNY_WATCHLIST=` est vide en `.env` (même bug latent que AI_BLUECHIP) mais **actuellement inoffensif** car la ligne DB a priorité et existe. Risque dormant si la DB redevient vide un jour. |
| Autres chemins vers PaperTrade | AI_PENNY a plusieurs tâches liées en plus de `_execute_(alpaca_)paper_trades_for_sandbox` (`penny_sniper_alert`, `run_penny_ai_scout`, `penny_opportunity_scanner` vus dans le code) — **pas toutes tracées en détail dans cet audit**, faute de temps ; les 4 trades sur 180j pourraient venir de l'un de ces chemins alternatifs plutôt que du chemin principal. À creuser si une priorisation future s'y intéresse. |

### Feature registry — risque de divergence vérifié, pas juste supposé

Un seul fichier (`feature_registry.py`), mais lecture complète confirme des
incohérences internes déjà documentées par ses propres commentaires :

- `BLUECHIP_FEATURE_NAMES` (27 features) : commentaire explicite — **"aucun
  pipeline n'importe cette liste actuellement"**. Purement aspirationnelle.
- `STABLE_FEATURE_NAMES` : sert au pipeline legacy ; un pipeline plus récent
  (`pipelines/stable_pipeline.py`) calcule un set **différent** de 10
  features sous un nom local, délibérément non fusionné (documenté comme
  tel, pas un oubli).
- `PENNY_FEATURE_NAMES`/`CRYPTO_FEATURE_NAMES` : commentaires indiquent des
  désynchronisations déjà trouvées et corrigées par le passé (features
  listées mais jamais calculées, silencieusement remplacées par des zéros).
- **Piège actif confirmé** : `get_feature_names('WATCHLIST')` résout vers
  l'alias `BLUECHIP` — la liste "jamais utilisée nulle part" ci-dessus.
  Mais le chemin de ré-entraînement réel (`retrain_from_paper_trades_daily`)
  importe `FEATURE_COLUMNS` directement depuis `FUSION` (52 features), sans
  jamais appeler `get_feature_names()`. **Deux fonctions différentes
  peuvent donc résoudre "les features de WATCHLIST" vers deux listes
  different (27 vs 52)** selon laquelle est appelée — personne n'a
  encore été mordu par ça (rien n'appelle `get_feature_names('WATCHLIST')`
  en production actuellement, vérifié), mais le piège est réel et prêt à
  se déclencher au premier nouveau code qui l'utiliserait sans le savoir.

### Labeling — pas de divergence réelle trouvée (risque suspecté, mais infirmé)

Un seul fichier consolidé, `ml_engine/training/labeling.py`, contient les
deux variantes légitimes :
- `triple_barrier_labels` (barrières fixes en %) — utilisé par
  `pipelines/stable_pipeline.py`.
- `triple_barrier_labels_adaptive` (barrières ATR-adaptatives par univers)
  — utilisé par `backtester.py`/`train_fusion_model_from_labels`, donc par
  le chemin de ré-entraînement réel des 3 sandboxes de cet audit.

`pipeline_fixes.py` contenait auparavant une implémentation séparée —
son commentaire actuel dit explicitement qu'elle "a déménagé vers"
`training/labeling.py`. **Vérifié : pas de duplication active, juste un
héritage de code déjà nettoyé avant cette session.** Le risque suspecté ne
s'est pas confirmé ici.

### Validation à l'entraînement — confirmé, PAS un principe déclaré seulement

- **Chemin manuel** (`manage.py ml_scanner` → `pipelines/{stable,penny,crypto}_pipeline.py`
  → `ml_engine/training/trainer.py`) : utilise réellement
  `PurgedTimeSeriesSplit` (`ml_engine/validation.py`), importé ET instancié
  (`tscv = PurgedTimeSeriesSplit(...)`, ligne 85 de `trainer.py`). Vérifié
  fonctionnel, pas juste présent dans le code.
- **Chemin automatique quotidien** (`retrain_from_paper_trades_daily`,
  Beat 7h40 UTC) : utilise sa propre fonction `_split_time()` — un simple
  tri chronologique + coupure par ratio (défaut 80/20) ou par date fixe.
  **Aucun embargo/purge, aucun appel à `PurgedTimeSeriesSplit`.**

Conclusion : "resté au stade de principe" est **faux** pour le chemin
manuel (fonctionne réellement), mais **vrai** pour le chemin automatique
quotidien qui alimente en pratique les modèles WATCHLIST/AI_BLUECHIP/
AI_PENNY.

---

## Étape 2 — Cause réelle de l'absence de trades, confirmée par isolation

### Reproduction en direct, `blocked_*` réels sur 30 jours (toutes sandboxes)

```
blocked_threshold           : 1982   <- de très loin le plus fréquent
blocked_no_price             :  867
blocked_no_signal            :  316
blocked_existing_position    :   77
blocked_halt_or_flash        :   38
blocked_intraday             :   30
blocked_bearish_pattern      :   22
blocked_confidence           :   13
blocked_reentry_unconfirmed  :    1
created                      :    0   <- confirmé sur les 678 exécutions du mois
--- tous les autres blocked_* (correlation, volume_z, atr_spike, btc_trend,
    exception, pump_dump, daily_trend, low_capital, reddit_hype,
    spread_wide, exposure_cap, sector_trend, win_rate_gate, zero_quantity,
    loss_blacklist, market_sentiment, penny_rvol_breakout,
    recent_loss_reentry, order_book_imbalance, weak_short_sentiment) : 0
```

**Le funnel de décision meurt presque entièrement aux 3 premières portes**
(`blocked_threshold`, `blocked_no_price`, `blocked_no_signal`) — les
filtres plus sophistiqués en aval (corrélation, sentiment, capital,
exposition, momentum) **ne sont quasiment jamais atteints**, pas parce
qu'ils bloquent tout, mais parce que rien n'arrive jusqu'à eux.

### Circuit breaker — réel, fréquent, mais pas la cause principale

**21 déclenchements en 30 jours** (`SystemLog`, message "Circuit breaker"),
12 pour AI_BLUECHIP, 9 pour AI_PENNY, aucun trouvé pour WATCHLIST dans les
messages (WATCHLIST n'utilise pas ce mécanisme, seul le chemin Alpaca de
`_execute_alpaca_paper_trades_for_sandbox` l'appelle). ~0.7 déclenchement/
jour est significatif — probablement disproportionné par rapport à
l'intention "protection contre un vrai risque" — mais mathématiquement,
1982 blocages sur seuil éclipse largement 21 déclenchements de circuit
breaker comme explication du "zéro trade".

### Causes confirmées, par sandbox

- **WATCHLIST** : déjà diagnostiqué (Partie C) — formule RSI structurellement
  sous le seuil + 4 tickers morts (corrigés). `blocked_threshold` dominant
  ici colle exactement à ce diagnostic.
- **AI_BLUECHIP** : **double cause confirmée** — (1) chemin SIM scanne
  littéralement 0 ticker (bug `_env_list` + watchlist DB vide + tâche de
  remplissage auto qui timeout systématiquement), (2) chemin Alpaca a de
  vrais tickers mais un seuil de 0.85 quasiment inatteignable pour un
  composite dont seulement 20% vient du signal du modèle.
- **AI_PENNY** : watchlist réelle et non vide (contrairement à AI_BLUECHIP),
  donc la cause principale ici est le seuil (0.82) + circuit breaker
  fréquent (9/30j) + possible fraîcheur du modèle (6 mois) — pas un
  problème d'univers vide comme AI_BLUECHIP.

### Corrélation avec l'item 7 (Beat en rafale) — vérifiée, pas supposée

Item 7 (fichier de planning Beat perdu à chaque `--force-recreate`,
corrigé le 2026-08-12) explique des **exécutions EN TROP** de tâches
planifiées, pas des blocages de logique métier. Vérifié : la cause de
l'échec de `refresh_ai_bluechip_watchlist` est un timeout HTTP interne
(60s), sans rapport avec la fraîcheur de l'état de Beat — cette tâche
échouerait de la même façon qu'elle tourne 1× ou 10× dans la même heure.
**Conclusion : pas de corrélation causale trouvée entre l'item 7 et
l'absence de trades.** Item 7 a pu gonfler le volume de tentatives
(scans en double), et donc gonfler proportionnellement TOUS les compteurs
`blocked_*` y compris ceux ci-dessus, mais n'explique pas pourquoi si peu
de choses passent les filtres.

---

## Étape 3 — Âge et pertinence des modèles

**Confirmé par les fichiers réels sur le NAS** (pas une estimation) :

```
data_fusion_brain_bluechip_v1.pkl   Feb 18  01:58   (WATCHLIST + AI_BLUECHIP)
data_fusion_brain_penny_v1.pkl      Feb 18  01:59   (AI_PENNY)
data_fusion_brain_v1.pkl            Feb 15  22:03
```

`ModelRegistry` (table applicative) confirme la même fenêtre : 2 entrées
seulement (BLUECHIP, PENNY), version `202602240210xx` — **~24 février**,
**`win_rate=0.0`, `paper_trades=None`** — ces valeurs vides suggèrent un
entraînement bootstrap initial, PAS une promotion réussie du chemin
automatique quotidien. **Aucune preuve qu'un ré-entraînement automatique
ait jamais abouti depuis.**

**Modèles vieux d'environ 6 mois** au moment de cet audit (2026-08-12) —
un régime de marché potentiellement très différent de février n'est pas
exclu (pas creusé plus loin ici, hors scope "pas d'analyse macro").

**Mécanisme automatique existe mais est structurellement bloqué** :
`retrain_from_paper_trades_daily` (Beat, quotidien) ne peut promouvoir un
nouveau modèle QUE si `paper_trades_count ≥ 20` trades fermés sur 90 jours
pour le sandbox — **une dépendance circulaire confirmée** : pas de trades
→ pas assez d'échantillons pour valider un nouveau modèle → le modèle ne
se met jamais à jour → (si la staleness contribue au manque de trades) →
le problème s'auto-entretient. Ce n'est ni un bug, ni une condition jamais
remplie par accident — c'est une garde de sécurité (éviter de promouvoir
un modèle sur un échantillon minuscule) qui a un effet de bord non prévu
dans ce scénario précis.

---

## Étape 4 — Synthèse, priorisation (rien corrigé)

### Par sandbox, cause confirmée de l'absence de trades récents

| Sandbox | Cause confirmée | Modèle | Risque du fix |
|---|---|---|---|
| WATCHLIST | Formule RSI structurellement sous le seuil (0.55) + 4/10 tickers morts (**déjà corrigés**, Partie C) | ~6 mois, partagé avec BLUECHIP | Calibration = décision de stratégie, pas de fix de code isolé |
| AI_BLUECHIP | Chemin SIM : univers vide (bug `_env_list` + DB vide + autofill qui timeout). Chemin Alpaca : seuil 0.85 quasi inatteignable | ~6 mois | Watchlist = isolé/config, faible risque. Seuil/autofill = touche l'exécution réelle |
| AI_PENNY | Seuil 0.82 très élevé + circuit breaker fréquent (9/30j) + modèle vieux de 6 mois | ~6 mois, dédié | Touche l'exécution réelle d'ordres |

### Pistes d'amélioration, classées par risque et effort (pas exhaustif)

1. **Corriger le bug `_env_list` (variable d'env présente-mais-vide ignore le défaut)**
   — Risque : **faible, isolé** (une fonction utilitaire, 4 lignes).
   Effort : faible. Impact : débloquerait immédiatement le chemin SIM
   d'AI_BLUECHIP (et le risque dormant identique sur AI_PENNY).
2. **Réparer ou retirer `refresh_ai_bluechip_watchlist`** (timeout HTTP
   systématique vers son propre endpoint interne) — Risque : **faible,
   isolé** au remplissage de watchlist, ne touche pas l'exécution d'ordres.
   Effort : moyen (il faut diagnostiquer `/api/ai/opportunities/`
   séparément, pas fait dans cet audit).
3. **Faire pointer `retrain_from_paper_trades_daily` vers
   `PurgedTimeSeriesSplit`** au lieu de `_split_time()` — Risque :
   **touche l'entraînement des modèles réels** (pas l'exécution d'ordres
   directement, mais influence quel modèle finit actif). Effort : faible
   (infrastructure déjà validée par le chemin manuel). Déjà noté comme
   bon rapport impact/risque par l'utilisateur — d'accord avec cette
   évaluation.
4. **Nettoyer la résolution de features pour WATCHLIST** (`get_feature_names`
   alias vers BLUECHIP, jamais utilisé nulle part) — Risque : **quasi nul**
   (piège dormant, personne ne l'appelle actuellement). Effort : trivial.
   Élimine un risque silencieux futur plutôt qu'un problème actif.
5. **Revoir la dépendance circulaire de la garde de promotion**
   (`paper_trades_count ≥ 20` empêche tout ré-entraînement tant qu'il n'y
   a pas déjà assez de trades) — Risque : **le plus élevé de la liste**,
   touche directement le mécanisme qui décide quel modèle trade du
   capital réel. Effort : élevé (nécessite de repenser la garde, pas
   juste la contourner). À traiter en dernier, avec le plus de prudence.

**Non classé faute de diagnostic suffisant dans cet audit** : la
calibration des seuils (0.55/0.82/0.85) eux-mêmes, et les chemins
alternatifs d'AI_PENNY (`penny_sniper_alert`, `run_penny_ai_scout`,
`penny_opportunity_scanner`) qui pourraient avoir leur propre logique de
seuil non couverte ici.

---

## Rapport final

**Pourquoi aucun trade récent, résumé** : pas un seul bug commun aux 3
sandboxes, mais un ensemble cohérent de causes de même nature — des
signaux qui n'atteignent (presque) jamais leur seuil d'achat
(`blocked_threshold` = 1982 sur 2354 blocages non triviaux du mois),
aggravé pour WATCHLIST par un remplacement de signal jamais aligné sur le
modèle ML, et pour AI_BLUECHIP par un univers de scan carrément vide sur
son chemin principal (bug de config + tâche de remplissage cassée).

**Âge des modèles** : les 3 sandboxes tournent sur des modèles entraînés
il y a **~6 mois** (18-24 février 2026), avec un mécanisme de
ré-entraînement automatique existant mais structurellement verrouillé par
une dépendance circulaire (pas de trades → pas de promotion possible).

**Même problème partout, ou différent par sandbox ?** Le symptôme racine
(seuils trop élevés / signal jamais assez fort) est **commun aux 3**,
mais WATCHLIST et AI_BLUECHIP ont chacun une cause additionnelle
spécifique et différente (formule RSI mal calibrée pour l'un, univers de
scan vide pour l'autre).

**Top 3-5 priorisées par impact réel** (reprend la liste ci-dessus,
ordre d'impact estimé, pas d'effort) :
1. Fix `_env_list` (débloque AI_BLUECHIP immédiatement, risque minimal)
2. `PurgedTimeSeriesSplit` sur le ré-entraînement quotidien (déjà validé
   ailleurs, meilleur rapport impact/risque selon l'utilisateur — d'accord)
3. Réparer/retirer `refresh_ai_bluechip_watchlist`
4. Nettoyer l'alias de features WATCHLIST (impact faible mais gratuit)
5. Revoir la garde de promotion circulaire (impact potentiellement le
   plus élevé à terme, mais le plus risqué — à faire en dernier, avec
   son propre diagnostic dédié avant tout changement)

Aucune correction n'a été appliquée dans cette session, conformément à la
consigne.
