# FUNDAMENTAL_LAB — plan pour un futur modèle ML entraîné sur les résultats réels

Statut : **pas construit** — ce document structure la collecte de données dès
maintenant pour ne pas avoir à combler des trous plusieurs mois plus tard,
une fois qu'il y aura assez de volume pour entraîner quoi que ce soit. Aucun
modèle n'est entraîné ni déployé par ce lot (2026-08-14).

Contexte : `docs/TECH_DEBT_NOTES.md` item lié au cas HAIN (garde-fou de
sécurité, voir `analysis/services/sanity_check.py`) — un LLM sans mémoire
d'un appel à l'autre ne peut pas "apprendre" d'un mauvais verdict passé. Un
modèle léger entraîné sur les résultats réels de FUNDAMENTAL_LAB (une fois
assez de volume) serait une 4e couche capable, elle, d'apprendre de
l'historique réel des positions — sans remplacer les 3 couches existantes.

## Partie 2.1 — Audit de ce qui est déjà collecté

**`ScanResult.quant_details`** (JSONField) : contient déjà tous les ratios
bruts utilisés à la décision, pas seulement le score composite — confirmé en
lisant `analysis/services/quant_filter.py::fetch_ticker_data` /
`evaluate_ticker` : `pe_ratio`, `peg_ratio`, `pb_ratio`, `ev_ebitda`, `roe`,
`debt_to_ebitda`, `fcf_yield`, `num_analysts`, `gross_margin_trend`,
`market_cap`, `margin_inflection_detected`, et `thresholds_used` (les seuils
sectoriels effectivement appliqués ce jour-là, utile pour normaliser un ratio
par rapport à son secteur plus tard). **Complet, aucun manque trouvé ici.**

**`FundamentalLabPosition`** (une fois fermée) : AVANT ce lot, seul
`is_open` passait à `False` — le rendement réel, le prix de sortie et la
raison de fermeture n'existaient qu'indirectement via `paper_trade`
(`PaperTrade.exit_price`/`exit_date`/`pnl`), et la raison de fermeture
n'existait qu'en texte libre dans `PaperTrade.notes` (`"Stop-loss
déclenché..."`), pas machine-lisible. **Manque comblé** (Partie 2.2
ci-dessous).

**Verdict LLM + confiance** : `ScanResult.claude_verdict`/`claude_reasoning`
étaient déjà stockés, mais `verify_ticker()` (`claude_verifier.py`) retourne
aussi `confidence` (`low`/`medium`/`high`) dans `ClaudeResult` sans que ce
soit jamais persisté nulle part — utilisé une fois (dans le raisonnement du
prompt) puis perdu. **Manque comblé** (Partie 2.2 ci-dessous).

## Partie 2.2 — Manques comblés dans ce lot

1. **`ScanResult.claude_confidence`** (`CharField`, `low`/`medium`/`high`) —
   renseigné dans `analysis/tasks.py::_process_candidate` au même moment que
   `claude_verdict`/`claude_reasoning`.
2. **`ScanResult.anomaly_reason`** (`TextField`) — raison précise du
   garde-fou de sécurité quand `final_verdict == 'flagged_anomaly'` (Partie
   1). Pas une feature ML au sens strict, mais nécessaire pour ne pas
   confondre un rejet-garde-fou avec un vrai signal négatif si on entraîne
   un jour sur `final_verdict` comme label.
3. **`FundamentalLabPosition.exit_price` / `exit_date` / `close_reason`
   (`stop_loss`/`manual`) / `realized_return_pct`** — renseignés dans
   `analysis/tasks.py::monitor_lab_positions` au moment de la fermeture
   stop-loss. Dupliqués depuis `PaperTrade` plutôt que de forcer une
   jointure à chaque requête ML future : cette table est le futur
   feature/label set, elle doit être auto-suffisante.

**Gap restant, noté mais pas comblé (hors scope de ce lot)** :
`monitor_lab_positions` (stop-loss) est aujourd'hui la SEULE voie de
fermeture d'une position FUNDAMENTAL_LAB — aucun mécanisme de fermeture
manuelle n'existe côté FUNDAMENTAL_LAB (`LabPositionMarkManualView` ne fait
que poser un flag `manually_selected`, il ne ferme jamais rien). Le champ
`close_reason='manual'` existe donc dans le schéma mais ne sera jamais
peuplé tant qu'un tel mécanisme n'est pas construit — à faire dans une
session dédiée si Eric veut un jour clôturer une position à la main.

## Partie 2.3 — Plan pour plus tard

**Volume minimum avant d'entraîner : 50-100 trades fermés.** Cohérent avec
le principe déjà appliqué à AI_PENNY sur ce projet (même logique : pas assez
de données en dessous, un modèle entraîné sur trop peu d'exemples
mémorise du bruit plutôt que d'apprendre un vrai signal).

**Type de modèle envisagé : léger.** Régression logistique ou gradient
boosting (ex. `LightGBM`/`XGBoost` avec peu d'arbres) — pas de deep
learning. Cohérent avec le volume de données attendu (quelques centaines
d'exemples au mieux à moyen terme) : un modèle profond surapprendrait
immédiatement sur un jeu aussi petit.

**Feature set probable** (déjà collecté intégralement dès aujourd'hui,
voir Partie 2.1/2.2) :
- Ratios quant bruts : `pe_ratio`, `peg_ratio`, `pb_ratio`, `ev_ebitda`,
  `roe`, `debt_to_ebitda`, `fcf_yield`, `num_analysts`,
  `margin_inflection_detected`.
- Secteur (`ScanResult.sector`) — probablement en one-hot ou embedding
  simple vu le nombre limité de secteurs GICS.
- Verdict LLM (`claude_verdict`) et sa confiance (`claude_confidence`).
- `verdict_source` (`single_call`/`llm_consensus`/`llm_disagreement`) —
  signal indirect de la fiabilité du verdict ce jour-là.
- Label : `FundamentalLabPosition.realized_return_pct` (régression) ou
  `close_reason`/signe du rendement (classification win/loss).

**Position dans le pipeline : 4e couche, pas un remplacement.** Comme
discuté avec Eric — quant (étape 1) → DeepSeek (étape 2) → Claude (étape 3)
→ **modèle ML (étape 4, future)**, qui n'interviendrait qu'APRÈS les 3
étapes existantes, sur les candidats qui les ont déjà passées (confirmed ou
uncertain). Rôle envisagé : un filtre de probabilité supplémentaire ou un
signal de conviction additionnel, jamais un remplacement du jugement
Claude/DeepSeek ni du garde-fou de sécurité (`sanity_check.py`, Partie 1) —
qui reste, lui, un filtre à seuils déterministes, actif dès aujourd'hui et
indépendant de tout modèle futur.

**Pas de date cible.** Ce document sera repris (nouvelle session dédiée)
une fois le seuil de 50-100 trades fermés atteint — pas avant, pour éviter
la même erreur circulaire déjà rencontrée ailleurs sur ce projet (voir
`docs/ML_PIPELINE_AUDIT.md`, garde-fou de promotion circulaire) : ne pas
construire un modèle sur un volume de données insuffisant juste parce que
l'infrastructure est prête.
