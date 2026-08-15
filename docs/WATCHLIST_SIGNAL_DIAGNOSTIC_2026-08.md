# Diagnostic — chemin de signal WATCHLIST (RSI seul vs FUSION/base_model_signal)

**Statut : DÉCISION PRISE ET IMPLÉMENTÉE le 2026-08-15 — Option A choisie
par Eric.** Voir "Décision et implémentation" tout en bas pour le détail
du changement de code, du recalibrage des seuils et du fix connexe
(item 17, TECH_DEBT_NOTES.md) découvert nécessaire pour qu'Option A
fonctionne réellement. Le reste du document (étapes 1-3 + addendum) est
conservé tel quel comme trace du diagnostic qui a mené à cette décision.

Contexte de départ (déjà établi la semaine dernière) : WATCHLIST devait être
une liste de tickers surveillés pour N'IMPORTE QUEL type d'opportunité, pas
une stratégie de survente technique spécifique. Le chemin de décision réel
utilise `_mean_reversion_score` (RSI + Bollinger), resté à 0.0 sur 100% des
79 échantillons historiques échantillonnés.

**Mise à jour** : le "toujours 0.0" et le "signal NVDA/AMZN dupliqué à 16
décimales" ont été ré-investigués (voir Addendum tout en bas) — **ce ne
sont PAS la même cause racine**. Conclusion courte : la formule
`_mean_reversion_score` fonctionne correctement aujourd'hui (vérifiée en
direct, varie normalement avec le RSI réel) ; le "toujours 0.0" reflète une
réalité de marché (les tickers WATCHLIST actuels ne sont simplement jamais
assez survendus), pas un bug de calcul. Le "dupliqué à 16 décimales" était
dans `base_model_signal` (chemin FUSION) début février, ne se reproduit
plus aujourd'hui, et est sans rapport avec le RSI. Détails complets et
preuves dans l'Addendum.

## Étape 1 — Contenu réel de FUSION_FEATURE_NAMES

**Correction factuelle** : `FUSION_FEATURE_NAMES` contient **54 features**,
pas 52 (compté directement dans `ml_engine/feature_registry.py`, ligne
141-213). Écart mineur, mais à corriger dans toute future référence.

### Liste complète (54)

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | `sma_ratio_10_50` | 28 | `return_5d` |
| 2 | `sma_ratio_20_50` | 29 | `mom_zscore_20` |
| 3 | `ema_ratio_9_20` | 30 | `linear_slope_20` |
| 4 | `price_to_ema9` | 31 | `slope_accel_20` |
| 5 | `price_to_ema20` | 32 | `donchian_pct_20` |
| 6 | `rsi_14` | 33 | `fib_distance_50` |
| 7 | `rsi_7` | 34 | `price_to_vwap_20` |
| 8 | `stoch_k_14` | 35 | `candle_body_pct` |
| 9 | `stoch_d_14` | 36 | `close_pos_in_range` |
| 10 | `williams_r_14` | 37 | `gap_pct` |
| 11 | `cci_20` | 38 | `pattern_doji` |
| 12 | `macd_hist` | 39 | `pattern_hammer` |
| 13 | `macd_line` | 40 | `pattern_engulfing` |
| 14 | `bb_pct_b_20` | 41 | `pattern_morning_star` |
| 15 | `bb_bandwidth_20` | 42 | `spy_corr_60` |
| 16 | `adx_14` | 43 | `tsx_corr_60` |
| 17 | `adx_di_ratio` | 44 | `sector_beta_60` |
| 18 | `rubber_band_20` | 45 | `VIXCLS` |
| 19 | `volume_zscore_20` | 46 | `DCOILWTICO` |
| 20 | `rvol_20` | 47 | `CPIAUCSL` |
| 21 | `RVOL10` | 48 | `bid_ask_spread_pct` |
| 22 | `obv_zscore` | 49 | `order_book_imbalance` |
| 23 | `VPT_roc` | 50 | `trade_velocity` |
| 24 | `volatility_20` | 51 | `sentiment_score` |
| 25 | `vol_regime` | 52 | `news_count` |
| 26 | `atr_pct_14` | 53 | `dividend_yield` |
| 27 | `return_20d` | 54 | `frac_diff_close` |

### Features de survente/sur-achat (RSI, stochastique, Williams %R, Bollinger, équivalent)

FUSION en contient **9**, contre 1 seule dimension (RSI) + 1 bonus binaire
(Bollinger lower touch) pour `_mean_reversion_score` aujourd'hui :

| Feature FUSION | Type d'oscillateur | Ce qu'elle capture |
|---|---|---|
| `rsi_14` | RSI 14 jours | Identique à ce que `_mean_reversion_score` utilise déjà |
| `rsi_7` | RSI 7 jours | RSI court terme, plus réactif — absent du chemin WATCHLIST actuel |
| `stoch_k_14` / `stoch_d_14` | Stochastique %K/%D | Oscillateur de survente/sur-achat classique, indépendant du RSI |
| `williams_r_14` | Williams %R | Oscillateur de survente/sur-achat classique, échelle inversée |
| `cci_20` | CCI 20 jours | Survente/sur-achat au-delà de ±100, plus sensible aux extrêmes |
| `bb_pct_b_20` | Bollinger %B | Position exacte dans les bandes (0 = bande basse, 1 = bande haute) — version continue de ce que `_mean_reversion_score` ne capture qu'en binaire (touche ou pas la bande basse) |
| `rubber_band_20` | "Rubber Band" (snapback) | Mean-reversion à proprement parler — distance à une moyenne, potentiel de retour |
| `donchian_pct_20` | Position dans le canal Donchian | Proximité du plus bas/plus haut sur 20 jours, proche du signal survente/sur-achat |

**Conclusion étape 1 (survente)** : FUSION ne perd rien de ce que
`_mean_reversion_score` capture aujourd'hui — elle capture le même RSI(14)
et une version continue (pas binaire) du signal Bollinger, PLUS 6 autres
angles de survente/sur-achat indépendants (RSI court terme, stochastique,
Williams %R, CCI, rubber band, Donchian).

### Autres catégories de signal dans FUSION

| Catégorie | Features | Ce que ça capture |
|---|---|---|
| Tendance | `sma_ratio_10_50/20_50`, `ema_ratio_9_20`, `price_to_ema9/20`, `macd_hist/line`, `adx_14`, `adx_di_ratio` | Direction et force de tendance |
| Volatilité/régime | `bb_bandwidth_20`, `atr_pct_14`, `volatility_20`, `vol_regime` | Contexte de volatilité (calme vs agité) |
| Volume | `volume_zscore_20`, `rvol_20`, `RVOL10`, `obv_zscore`, `VPT_roc` | Anomalies/confirmation de volume |
| Momentum prix | `return_20d`, `return_5d`, `mom_zscore_20`, `linear_slope_20`, `slope_accel_20` | Vitesse et accélération du prix |
| Structure de prix | `donchian_pct_20`, `fib_distance_50`, `price_to_vwap_20`, `close_pos_in_range` | Position relative dans des repères de prix |
| Bougie/patterns | `candle_body_pct`, `gap_pct`, `pattern_doji/hammer/engulfing/morning_star` | Signaux chandeliers japonais |
| Macro | `spy_corr_60`, `tsx_corr_60`, `sector_beta_60`, `VIXCLS`, `DCOILWTICO`, `CPIAUCSL` | Contexte marché large (indices, VIX, pétrole, inflation) |
| Microstructure | `bid_ask_spread_pct`, `order_book_imbalance`, `trade_velocity` | Carnet d'ordres, liquidité à court terme |
| Sentiment/fondamental | `sentiment_score`, `news_count`, `dividend_yield` | News et fondamentaux légers |
| Autre | `frac_diff_close` | Différenciation fractionnaire (stationnarité du prix sans perdre toute la mémoire long terme) |

C'est nettement plus proche de l'intention originale de WATCHLIST ("n'importe
quel type d'opportunité") que RSI seul : tendance, volume, macro, sentiment,
microstructure, patterns chandeliers sont TOUS absents du chemin actuel.

### Comment `base_model_signal` se calcule (`_model_signal`, `portfolio/tasks.py:5033`)

Ce n'est **pas une combinaison de règles simple** — c'est la sortie d'un
vrai modèle ML entraîné :

1. `DataFusionEngine(symbol).fuse_all()` calcule les 54 features ci-dessus pour le ticker.
2. `load_or_train_model(...)` charge un `RandomForestClassifier` (300 arbres, `max_depth=5`) entraîné via validation croisée temporelle purgée (`PurgedTimeSeriesSplit`) sur des labels triple-barrière (`train_fusion_model`, `ml_engine/backtester.py:313`).
3. `payload['model'].predict_proba(features)[0][1]` — la probabilité prédite que le trade soit gagnant, calculée sur toutes les features que le modèle a vues à l'entraînement (les 54 de FUSION, sauf CRYPTO).
4. `apply_feature_weighting_to_signal(...)` applique un ajustement optionnel (désactivé par défaut, derrière un flag) — pas une deuxième couche de modèle, juste une pondération conditionnelle.
5. `explanations` (top-5 features par contribution, façon SHAP simplifié) est calculé et conservé — donc même en passant par le modèle ML, on garde une trace interprétable de "pourquoi" ce signal, pas une boîte noire totalement opaque.

**Fait confirmé et important** : `get_model_path('WATCHLIST')` ne matche
aucune branche explicite dans `backtester.py::get_model_path` — elle retombe
sur le cas par défaut, qui est `BLUECHIP_MODEL_PATH`
(`data_fusion_brain_bluechip_v1.pkl`). **WATCHLIST n'a jamais eu son propre
modèle entraîné** : elle partage littéralement le même fichier `.pkl` que
AI_BLUECHIP. `base_model_signal` pour WATCHLIST aujourd'hui est donc déjà
la sortie de CE modèle partagé — juste jetée après calcul (voir étape 2).

## Étape 2 — Comparaison des deux chemins

### Ce que `_mean_reversion_score` capture aujourd'hui (`portfolio/tasks.py:5208`)

```python
rsi_score = clip((30 - rsi) / 30, 0, 1)          # 0 si RSI >= 30, jusqu'à 1.0 si RSI proche de 0
bollinger_bonus = 0.15 si close <= bande_basse_bollinger sinon 0
score = clip(rsi_score + bollinger_bonus, 0, 1)
```

- **1 seule famille de signal** : survente pure (RSI + confirmation Bollinger).
- Source de données indépendante : appel `yfinance` direct dans la fonction elle-même (pas via `DataFusionEngine`).
- Résultat lisible/interprétable en un coup d'œil (un chiffre RSI, une bande).
- Aucune notion de tendance, volume, macro, sentiment, news.
- Confirmé la semaine dernière : 0.0 sur 100% de 79 échantillons (root cause non ré-investiguée cette session).

### Ce que `base_model_signal`/FUSION capterait à la place

- **54 features** couvrant survente (9 features, voir tableau ci-dessus), tendance, volatilité, volume, momentum, structure de prix, patterns chandeliers, macro, microstructure, sentiment.
- Sortie d'un `RandomForestClassifier` entraîné, pas une formule manuelle — apprend des interactions entre facteurs plutôt qu'un seuil fixe sur un seul indicateur.
- Interprétabilité partielle conservée via `explanations` (top-5 contributions), mais pas un simple "RSI = 22, donc survente" — un score composite dont la justification demande de lire les explanations.
- Aujourd'hui déjà **calculé pour WATCHLIST à chaque évaluation**, puis explicitement écrasé (voir ci-dessous) et gardé uniquement à titre diagnostique dans `payload['base_model_signal']`.

### WATCHLIST peut-elle utiliser ce chemin sans dupliquer de code ?

**Oui, sans aucune nouvelle pipeline de feature engineering.** Preuve directe
dans `_signal()` (`portfolio/tasks.py:5565`) :

```python
def _signal(symbol, intraday_ctx=None):
    base_payload = _model_signal(symbol, universe, model_path, use_alpaca=use_alpaca)  # <- tourne pour TOUS les sandboxes, WATCHLIST inclus
    base_signal = float(base_payload.get('signal') or 0) if base_payload else 0.0
    ...
    payload['base_model_signal'] = base_signal  # gardé pour diagnostic seulement

    if sandbox == 'WATCHLIST':
        score, rsi, lower_band = _mean_reversion_score(symbol)   # <- ÉCRASE payload['signal'] juste après
        payload['signal'] = score
        payload['model_name'] = 'MEAN_REVERSION'
        ...
        return payload
```

`_model_signal` (qui calcule les 54 features FUSION via `DataFusionEngine`
et fait tourner le modèle) s'exécute **déjà, inconditionnellement, à chaque
cycle WATCHLIST** — le calcul entier a lieu, coûte le même temps CPU/réseau
qu'aujourd'hui, et son résultat est simplement jeté trois lignes plus bas.
Basculer WATCHLIST sur `base_model_signal` reviendrait à supprimer le bloc
`if sandbox == 'WATCHLIST': ...` (ou à le remplacer par une ligne qui garde
`payload['signal'] = base_signal`) — **aucun changement de pipeline de
données, aucune duplication de code, aucun nouveau modèle à entraîner.**

`model_path` pour WATCHLIST résout déjà vers `BLUECHIP_MODEL_PATH` (même
fichier `.pkl` que AI_BLUECHIP) — confirmé à l'étape 1.

### Confirmation : FUSION capte déjà la survente — rien n'est perdu

Confirmé à l'étape 1 : FUSION contient 9 features de survente/sur-achat,
dont le RSI(14) exact utilisé aujourd'hui et une version continue (pas
binaire) du signal Bollinger. **Basculer WATCHLIST sur `base_model_signal`
ne fait perdre aucune capacité actuelle** — elle en gagne :
tendance, volume, macro, sentiment, microstructure, patterns chandeliers,
en plus d'une survente plus riche (8 dimensions au lieu d'1). Ce n'est donc
**pas** le compromis "large éventail vs signal survente perdu" — les deux
sont présents dans FUSION.

### Découverte annexe (non demandée, mais directement pertinente)

En traçant `retrain_from_paper_trades_daily` (`_train_for_sandbox`,
`portfolio/tasks.py:9483`), un **second bug distinct** a été trouvé, lié au
même override `MEAN_REVERSION` :

- `_train_for_sandbox('WATCHLIST')` filtre les trades avec `model_name='BLUECHIP'` (ligne `model_name = 'PENNY' if selected_sandbox == 'AI_PENNY' else 'BLUECHIP'`).
- Mais chaque `PaperTrade` créé pour WATCHLIST est stampé `model_name='MEAN_REVERSION'` (via `payload['model_name'] = 'MEAN_REVERSION'` dans `_signal()`, propagé à `PaperTrade.objects.create(model_name=(signal_payload or {}).get('model_name', universe), ...)`).
- **Conséquence** : le réentraînement quotidien de WATCHLIST ne trouve jamais aucun trade (`status: no_trades`), quel que soit le volume réel de trades WATCHLIST fermés — le filtre et les données réelles ne se rencontrent jamais.
- Ce bug se corrigerait de lui-même en basculant sur `base_model_signal` : `payload['model_name']` resterait `universe` = `'BLUECHIP'` (valeur retournée par `_model_signal`), qui matche exactement ce que `_train_for_sandbox` cherche.
- **Pas corrigé dans cette session** — documenté ici parce qu'il change la balance des options (voir étape 3) : aujourd'hui, même en gardant `_mean_reversion_score` (Option B), le réentraînement resterait cassé tant que ce mismatch `model_name` n'est pas traité séparément.

## Étape 3 — Options pour Eric (aucune implémentée)

### Option A — Basculer WATCHLIST sur `base_model_signal`/FUSION

**Changement** : dans `_signal()`, remplacer le bloc `if sandbox ==
'WATCHLIST': ...` par une simple utilisation de `base_signal` (comme
AI_BLUECHIP/AI_PENNY le font déjà).

- ✅ Aligné avec l'intention originale de WATCHLIST ("n'importe quel type d'opportunité").
- ✅ Aucune perte de capacité (survente déjà incluse dans FUSION, en mieux).
- ✅ Zéro nouvelle pipeline — le calcul tourne déjà à chaque cycle.
- ✅ Corrige gratuitement le bug de réentraînement `model_name` découvert ci-dessus.
- ✅ Interprétabilité partielle conservée (`explanations`, top-5 features).
- ⚠️ WATCHLIST utiliserait EXACTEMENT le même modèle entraîné que AI_BLUECHIP (même fichier `.pkl`) — si les deux sandboxes gardent des tickers/stratégies différents, un seul modèle partagé pourrait ne pas être idéalement calibré pour les deux à la fois (question à trancher séparément : faut-il un modèle WATCHLIST dédié à terme ?).
- ⚠️ Perte de la lisibilité "RSI = X, donc survente" en un coup d'œil — remplacée par une explication top-5 features moins immédiate à lire.
- ⚠️ Change le comportement réel de trading (nouveaux seuils buy/sell à re-calibrer probablement, la distribution de `base_model_signal` n'a aucune raison de ressembler à celle de `_mean_reversion_score`).

### Option B — Garder `_mean_reversion_score`, corriger son calibrage

Pertinent **seulement si** un vrai bug de formule est trouvé (pas juste un
seuil trop strict) — non identifié dans cette session pour la formule
elle-même (RSI + Bollinger, logique standard, pas d'erreur de calcul
repérée). Le bug distinct de `model_name`/réentraînement (voir ci-dessus)
resterait à corriger séparément dans cette option, indépendamment du
calibrage RSI.

- ✅ Change le moins de comportement — reste un filtre de survente pur, prévisible.
- ✅ Lisibilité maximale (RSI, Bollinger — deux chiffres à vérifier).
- ❌ Ne répond pas à la question de fond posée par Eric : WATCHLIST resterait une stratégie de survente étroite, pas "n'importe quel type d'opportunité".
- ❌ Le "toujours 0.0 sur 79 échantillons" n'a pas de cause identifiée ici — corriger un calibrage sans en connaître la cause reviendrait à deviner (nécessiterait une investigation dédiée, non faite cette session).
- ❌ Ne corrige pas le bug de réentraînement `model_name` (à traiter en plus, séparément).

### Option C — Score composite (FUSION en priorité + bonus survente confirmée)

Ex. `signal = base_model_signal + bonus si RSI < 30 et fermeture <= bande basse`.

- ✅ Garde le signal de survente comme un boost explicite et lisible en plus du signal large.
- ✅ Compromis entre les deux — utile si Eric veut spécifiquement continuer à privilégier les entrées en survente confirmée, tout en élargissant la détection.
- ⚠️ Plus de travail (nouvelle formule à calibrer, pas juste un branchement existant) — et vu que FUSION capture déjà la survente en interne (9 features), ce bonus recouperait en partie un signal que le modèle voit déjà, avec un risque de double-comptage à gérer soigneusement.
- ⚠️ Ajoute un paramètre de calibrage de plus (poids du bonus) à maintenir dans le temps.
- Recommandation : à ne considérer que si Eric tient explicitement à préserver un filtre de survente dédié en plus du signal large — pas justifié uniquement par ce que révèle l'étape 1 (FUSION couvre déjà largement la survente).

## Résumé pour décision rapide

| | Option A (FUSION) | Option B (RSI corrigé) | Option C (composite) |
|---|---|---|---|
| Aligné à l'intention "n'importe quelle opportunité" | ✅ Oui | ❌ Non | ✅ Oui |
| Perd la capacité de survente actuelle | ❌ Non (gagne 8 dimensions de plus) | — | ❌ Non |
| Travail d'implémentation | Faible (branchement existant) | Moyen (diagnostic root-cause requis) | Élevé (nouvelle formule + calibrage) |
| Corrige le bug `model_name`/réentraînement en bonus | ✅ Oui | ❌ Non (à faire séparément) | ✅ Oui (même mécanisme) |
| Lisibilité du signal | Moyenne (top-5 features) | Élevée (RSI/Bollinger) | Moyenne-Élevée |
| Recalibrage des seuils buy/sell nécessaire | Oui, probable | Non (si juste un fix de bug) | Oui, probable |

## Addendum (2026-08-14, suite à la demande d'Eric) — le "toujours 0.0" et le "dupliqué à 16 décimales" partagent-ils une cause racine ?

**Réponse courte : non, ce sont deux problèmes distincts, vérifiés séparément
en rejouant le calcul en direct sur la production.** Aucun changement de
code effectué — diagnostic seul, ~30 minutes.

### Preuve 1 — Le signal dupliqué NVDA/AMZN à 16 décimales, confirmé dans les vraies données

```
AMZN  WATCHLIST     2026-02-17  entry_signal=0.6592413017098232  model_name=BLUECHIP
AMZN  AI_BLUECHIP   2026-02-17  entry_signal=0.6592413017098232  model_name=BLUECHIP
NVDA  WATCHLIST     2026-02-20  entry_signal=0.6592413017098232  model_name=BLUECHIP
```

Confirmé : **3 trades, 2 tickers différents, 2 sandboxes différentes, 3
jours d'écart — un signal identique au bit près.** C'est bien
`base_model_signal` (le chemin FUSION/RandomForest), pas
`_mean_reversion_score` : `model_name='BLUECHIP'` sur les 3, et ces dates
(17-20 février) précèdent l'introduction de `_mean_reversion_score` pour
WATCHLIST — à ce moment-là, WATCHLIST utilisait encore directement le
signal FUSION brut, exactement comme AI_BLUECHIP aujourd'hui.

**Rejoué en direct aujourd'hui** (`_model_signal` appelé pour de vrai,
même modèle `.pkl`, même code) :

```
NVDA  base_model_signal = 0.31523266584743603
AMZN  base_model_signal = 0.31352662438312484
AAPL  base_model_signal = 0.30978636806943716
MSFT  base_model_signal = 0.3111979847241962
```

**Ne se reproduit plus** — 4 valeurs différentes aujourd'hui, alors qu'elles
étaient identiques au bit près en février. Cause la plus probable (pas
prouvée formellement, mais cohérente avec toutes les preuves disponibles) :
le modèle `.pkl` en février était très jeune / entraîné sur très peu
d'échantillons (le pipeline paper-trading démarrait à peine à ce
moment-là) — un `RandomForestClassifier` sous-entraîné sur un jeu quasi
vide peut dégénérer en prédicteur quasi-constant, peu importe l'input. Le
modèle a été ré-entraîné quotidiennement depuis (`retrain_from_paper_trades_daily`)
avec des mois de données réelles accumulées, ce qui explique naturellement
la différenciation retrouvée aujourd'hui.

**Nuance à garder en tête pour l'Option A** : les 4 valeurs actuelles
(0.310-0.315) restent assez resserrées entre elles vu à quel point NVDA
(RSI=75, sur-acheté) et AAPL (RSI=26, survendu) sont dans des régimes
opposés — pas une preuve de bug, mais un signal que le modèle FUSION
partagé ne différencie peut-être pas encore fortement selon les tickers.
À surveiller si l'Option A est choisie, pas un bloquant en soi.

### Preuve 2 — `_mean_reversion_score` fonctionne correctement aujourd'hui

Rejoué en direct sur des tickers connus, même run que ci-dessus :

```
NVDA  RSI=75.4  score=0.0     (RSI >= 30, normal)
AMZN  RSI=66.4  score=0.0     (RSI >= 30, normal)
MSFT  RSI=84.8  score=0.0     (RSI >= 30, normal)
AAPL  RSI=26.0  score=0.134   (RSI < 30 -- SCORE POSITIF, formule fonctionne)
```

La formule réagit correctement à un vrai cas survendu (AAPL) dans le même
lot de test que les cas non-survendus. **Pas un bug de calcul.**

### Preuve 3 — Rejoué sur les 10 tickers RÉELS de WATCHLIST aujourd'hui

```
BTO.TO   RSI=81.3   score=0.0
FLT.TO   RSI=51.9   score=0.0
BMBL     RSI=40.9   score=0.0
AIFF     RSI=48.7   score=0.0
DSP      RSI=64.4   score=0.0
ALVO     RSI=75.1   score=0.0
HIVE.TO  RSI=41.5   score=0.0
KEEL.TO  RSI=39.0   score=0.0
LUG.TO   RSI=78.4   score=0.0
NVDA     RSI=75.4   score=0.0
```

Le "toujours 0.0" **se reproduit bel et bien aujourd'hui, en direct** — mais
le RSI varie normalement (39 à 81) et n'est simplement JAMAIS sous 30 pour
ces 10 tickers précis, aujourd'hui. Confirmé : formule correcte, pas de bug
caché ; c'est un problème de calibrage/de nature du signal (un déclencheur
"survente sévère uniquement" appliqué à une liste de tickers qui n'entre
tout simplement pas souvent en survente sévère), exactement le diagnostic
déjà posé à l'étape 2 de ce document.

### Preuve 4 — Précision sur la provenance du chiffre "79 échantillons"

Tracé jusqu'à sa source réelle : `bluechip_dip_scanner` (`portfolio/tasks.py:7561`),
qui appelle aussi `_mean_reversion_score` et dont le commentaire cite
explicitement "WATCHLIST's buy_threshold" comme raison d'être de ce
logging — presque certainement la source du chiffre "79" évoqué la semaine
dernière.

**Correction importante** : interrogé les 237 runs de ce diagnostic
enregistrés depuis le 27 février 2026 — `score_distribution` est **`None`
(vide) dans 100% des cas, sans exception**. Ce n'est pas "0.0 dans 79
échantillons" mais littéralement **zéro échantillon n'a jamais atteint
`_mean_reversion_score`** dans ce chemin précis, parce qu'un filtre en
amont (`_rsi_divergence`, un test de divergence intrajournalière) rejette
~75-80% des candidats et le filtre de fourchette de prix rejette le reste
— `funnel['passed'] = 0` sur les 8 derniers runs consultés (12-14 août),
`evaluated` oscillant justement autour de 75-79 (probable origine du
chiffre "79").

**Ce chemin (`bluechip_dip_scanner`) N'EST PAS le chemin de décision réel
de WATCHLIST** (`_execute_paper_trades_for_sandbox`) — celui-ci appelle
`_mean_reversion_score` directement, sans le filtre `_rsi_divergence`. Les
deux chemins partagent la même fonction mais pas les mêmes filtres
amont. La Preuve 3 ci-dessus (rejoué directement sur le vrai chemin
WATCHLIST) reste la mesure la plus fiable de son comportement réel.

**Fait confirmé en bonus** : les 4 `PaperTrade` jamais créés pour WATCHLIST
dans toute l'histoire du système (AMZN 17 fév, NVDA 20 fév, HIVE 24 fév,
HIVE 28 fév) n'ont **aucun** `mean_reversion_score`/`rsi14` dans
`entry_features` — `_mean_reversion_score` n'a littéralement jamais produit
un seul trade WATCHLIST, à aucun moment de l'historique.

### Conclusion de l'addendum

**Les deux symptômes sont confirmés distincts, sans cause commune :**

1. Le signal dupliqué NVDA/AMZN était dans `base_model_signal` (chemin
   FUSION), causé très probablement par un modèle sous-entraîné début
   février — **ne se reproduit plus aujourd'hui**, rien à corriger pour le
   débloquer.
2. Le "toujours 0.0" de WATCHLIST est dans `_mean_reversion_score` (RSI +
   Bollinger) — **la formule est correcte**, vérifiée trois fois en direct
   (formule pure, cas AAPL survendu, 10 vrais tickers WATCHLIST). C'est un
   problème de calibrage/nature du signal, pas un bug à corriger.

**Aucun bug commun à corriger qui débloquerait WATCHLIST gratuitement.**
Corriger un "bug de features réutilisées entre appels" n'est pas
applicable ici — aucun des deux symptômes n'est causé par ça. L'**Option A
(basculer vers FUSION) reste le choix le mieux soutenu par les faits**,
pour trois raisons désormais confirmées :
- FUSION ne montre pas le problème de duplication aujourd'hui (testé en direct).
- Le "toujours 0.0" est structurel à la formule RSI-seule, pas un accident réparable par un simple fix de bug.
- Basculer corrige en prime le bug `model_name`/réentraînement documenté plus haut.

La seule réserve à garder en tête (Preuve 1) : les valeurs FUSION actuelles
sont resserrées entre tickers très différents — pas un blocage, mais un
point à surveiller après bascule, pas avant.

## Décision et implémentation (2026-08-15)

Eric a choisi **Option A**, avec le recalibrage des seuils buy/sell déjà
anticipé comme nécessaire, et le fix `model_name` (mentionné en passant
dans la description de l'Option B) explicitement inclus.

### Changements de code

1. **`_signal()` dans `_execute_paper_trades_for_sandbox`** (`portfolio/tasks.py`) :
   suppression complète du bloc `if sandbox == 'WATCHLIST': ...` qui
   écrasait `payload['signal']` avec `_mean_reversion_score` et
   `payload['model_name']` avec `'MEAN_REVERSION'`. WATCHLIST tombe
   maintenant sur le même `return payload` par défaut qu'AI_BLUECHIP/
   AI_PENNY, utilisant `base_signal` (sortie `predict_proba` du
   `RandomForestClassifier` sur les 54 features FUSION) tel quel. Ce
   changement corrige **gratuitement** le bug `model_name` documenté à
   l'étape 2 : `payload['model_name']` reste `universe` = `'BLUECHIP'`
   (valeur renvoyée par `_model_signal`), qui matche désormais ce que
   `retrain_from_paper_trades_daily::_train_for_sandbox` filtre pour
   WATCHLIST — plus besoin d'un fix séparé.

2. **Seuils buy/sell recalibrés, chemin SIM** (`_execute_paper_trades_for_sandbox`) :
   `buy_threshold`/`sell_threshold` passent de `0.55`/`0.25` (calibrés pour
   l'ancienne échelle RSI+Bollinger) à **`0.50`/`0.30`**, basés sur la
   distribution réelle de `base_model_signal` déjà loggée pour WATCHLIST
   (`SystemLog`, `signal_distribution.base_model_signal`, échantillons du
   2026-08-10 au 2026-08-14) : min ~0.17-0.25, médiane ~0.38-0.47, max
   ~0.54-0.58. `0.50` reste sélectif (au-dessus de la médiane typique) tout
   en étant atteignable (sous le max typique) ; `0.30` se situe près du bas
   de la fourchette observée. Écart de 0.20, cohérent avec le pattern
   AI_BLUECHIP (0.80/0.60).

3. **Seuils buy/sell recalibrés, chemin Alpaca** (`_execute_alpaca_paper_trades_for_sandbox`) :
   ce chemin utilisait déjà `base_signal` brut (pas de bloc
   `_mean_reversion_score` ici) mais retombait sur le défaut `0.82`
   (calibré pour AI_BLUECHIP/AI_PENNY), beaucoup trop strict pour
   l'échelle réelle observée — confirmé en direct : **1 seul trade
   Alpaca WATCHLIST créé dans toute l'historique du système** sous ce
   seuil. Mêmes valeurs que le chemin SIM (`0.50`/`0.30`) pour cohérence
   (même modèle partagé, même échelle de signal).

4. **Fix connexe découvert nécessaire, item 17 (TECH_DEBT_NOTES.md)** :
   en préparant le déploiement, un second bug indépendant a été trouvé
   dans le code déjà en cours (diagnostiqué en direct le 2026-08-14, avant
   ce lot) : `confidence_factor`/`min_confidence` utilisaient un plancher
   codé en dur (`0.65`/`0.7`), complètement indépendant de `buy_threshold`.
   **Sans ce fix, le passage sur FUSION n'aurait produit AUCUN trade
   supplémentaire en pratique** — la distribution réelle de
   `base_model_signal` (médiane ~0.40, max ~0.54-0.58) tombe presque
   entièrement dans l'ancienne zone morte 0.50-0.65 (clear `buy_threshold`
   mais rejeté ensuite pour `confidence_too_low`). Fix : le plancher est
   désormais plafonné par `buy_threshold`
   (`confidence_floor = min(0.65, buy_threshold)`), sans impact sur
   AI_BLUECHIP/AI_PENNY (seuils normaux au-dessus de 0.65, comportement
   inchangé). Détail complet dans `TECH_DEBT_NOTES.md` item 17.

### Tests

- `WatchlistBaseModelSignalRegressionTests` (`tests_paper_trade_decision_stats.py`) :
  `_model_signal` et `_mean_reversion_score` mockés pour DISAGREE (l'un
  clear le seuil, l'autre non) — confirme que le trade créé vient bien de
  `base_signal`, et que `model_name='BLUECHIP'` (jamais `'MEAN_REVERSION'`).
- `ConfidenceFloorCappedAtBuyThresholdTests` : un signal dans l'ancienne
  zone morte (0.55) crée bien un trade désormais ; un signal sous
  `buy_threshold` reste bloqué au bon endroit.
- Suite complète `portfolio.tests_paper_trade_decision_stats` (11/11 avant
  ce lot + les nouveaux) exécutée sur le vrai backend avant déploiement.
- Chemin Alpaca : pas de suite de tests automatisée dans ce repo (aucune
  avant ce fix) — vérifié manuellement en rejouant le calcul en direct sur
  la production (voir item 17).

### Suivi recommandé (pas fait dans ce lot, à surveiller)

- Les seuils `0.50`/`0.30` sont un calibrage raisonné à partir de la
  distribution observée, pas un optimum empirique (aucun trade WATCHLIST
  réel n'existe encore sous ce nouveau chemin pour calibrer autrement). À
  réévaluer une fois quelques semaines de trades réels accumulées.
- La réserve notée dans l'addendum (valeurs `base_model_signal` resserrées
  entre tickers très différents, 0.31-0.32) reste valable — à surveiller,
  pas un bloquant.
