# Notes techniques en attente (non corrigées intentionnellement)

Fichier de suivi pour les bugs/chantiers identifiés mais volontairement
**pas corrigés sur le coup**, pour reprise sans avoir à reconstituer le
contexte. Chaque entrée : ce qu'on sait avec certitude (vérifié), ce
qu'on suppose (hypothèse), et ce qui reste à faire.

---

## 1. `LabPositionMarkManualView` écrase `manual_note` sans garde
**Statut : ouvert, déféré au 2026-08-10 (soir précédent), toujours pas corrigé.**

`position.manual_note = request.data.get("note", "")` est inconditionnel.
Cliquer "Je crois en celle-là" sur une position qui a déjà une note de
traçabilité (ex. les 4 positions BRCC/BTO.TO/NVDA/SLQT nettoyées après le
bug du scan bugué du 2026-08-09) écrase silencieusement cette note.
Pas de dommage financier (`is_open`/`PaperTrade` non touchés par cet
endpoint), mais perte de traçabilité pour un futur post-mortem.

**À faire** : garder l'ancienne note si la nouvelle est vide, ou
concaténer avec un séparateur/horodatage plutôt qu'écraser.

Fichier : `portfolio_backend/analysis/views.py` (classe `LabPositionMarkManualView`).

---

## 2. `monitor_my_portfolio` échoue à 100% depuis au moins ce soir
**Statut : ouvert, découvert 2026-08-10 ~20h50 ET pendant l'audit Celery.**

**Vérifié** (TaskRunLog, 10 dernières exécutions consécutives, toutes
entre 16h50 et 20h37 ET aujourd'hui) : **échec systématique, message
identique à chaque fois** :
```
Second or Minute units can only be used with amounts between 1-59.
```

**Vérifié — ce n'est PAS le crontab Beat** (hypothèse initiale écartée) :
`settings.py` ligne ~497, entrée `'monitor-my-portfolio-hourly'` :
```python
'schedule': crontab(minute=0, hour='9-16', day_of_week='mon-fri'),
```
Syntaxe parfaitement valide (`minute=0`, pas de valeur hors 1-59). Le
crash a donc lieu **pendant l'exécution de la tâche**, pas au niveau de
sa planification.

**Hypothèse non vérifiée (piste à suivre demain)** : dans
`portfolio_backend/portfolio/tasks.py`, fonction `monitor_my_portfolio`
(~ligne 12542), la ligne :
```python
interval = '60m' if is_core else '15m'
...
hist = yf.Ticker(symbol).history(period=period, interval=interval)
```
`'60m'` est un interval yfinance documenté comme valide, mais le message
d'erreur ressemble à une validation interne pandas/yfinance sur un
offset `Minute`/`Second` — pas confirmé, pas de traceback complet stocké
dans `TaskRunLog.error` (seulement le message, pas la stack). À
reproduire en isolant l'appel `yf.Ticker(...).history(interval='60m')`
pour un symbole `is_core` avant de toucher au code.

**Ne pas corriger sans reproduire** — le message pourrait aussi venir
d'ailleurs dans la fonction (ex. un des indicateurs `_compute_*` plus
loin, ou `_portfolio_news_sentiment`), le `interval='60m'` n'est qu'une
suspicion basée sur la proximité du message pandas connu.

---

## 3. `monitor_hive_trade` — crash DataFrame ambigu, semble résolu de lui-même
**Statut : probablement clos, à surveiller, pas d'action immédiate.**

**Vérifié** : ~10 échecs en rafale entre 19h58:34 et 19h59:32 UTC
(15h58-15h59 ET) aujourd'hui, message :
```
The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(),
a.item(), a.any() or a.all().
```
Signature classique d'un `if df:` (ou équivalent) au lieu de
`if not df.empty:`/`if df is not None and not df.empty:` quelque part
dans le code de la tâche.

**Vérifié** : depuis 19h59:32 UTC, **20+ exécutions consécutives en
SUCCESS**, aucune récidive au moment de la rédaction (20h51 ET). Semble
intermittent et lié à des données spécifiques à un ticker/moment
(ex. DataFrame vide ou multi-colonnes retourné pour un cas particulier)
plutôt qu'à un bug systématique — mais pas creusé plus loin, comme
convenu.

**À faire (si ça revient)** : chercher un test booléen direct sur un
DataFrame dans `portfolio.tasks.monitor_hive_trade` et le remplacer par
un test explicite sur `.empty`.

---

## 4. `tsx-guardian-30s` — nom trompeur, tourne 24/7 pas juste en heures de marché
**Statut : noté en passant, pas prioritaire, pas vérifié en détail.**

`settings.py` ligne ~375 :
```python
'tsx-guardian-30s': {
    'task': 'portfolio.tasks.monitor_active_trade',
    'schedule': timedelta(seconds=120),
},
```
Le nom dit "30s" mais le vrai intervalle est `timedelta(seconds=120)` =
2 minutes, et contrairement aux autres tâches de surveillance (qui ont
un `crontab(..., hour='9-16', day_of_week='mon-fri')`), celle-ci n'a
aucune restriction d'heure/jour — tourne en continu, marché fermé
compris. À clarifier : est-ce voulu (ex. surveillance de positions
overnight) ou un oubli de restriction horaire copié d'un autre
template ? Pas d'impact connu, juste une incohérence de nommage/config
relevée en marge de l'audit du 2026-08-10.

---

## 5. [FAIT le 2026-08-10] Queue Celery dédiée pour le module analysis
**Statut : implémenté et déployé ce soir, à confirmer demain 9h45 ET.**

Contexte : `run_daily_scan` risquait de rester coincé indéfiniment
derrière le flot de tâches haute-fréquence (penny_sniper_alert,
monitor_hive_trade, monitor_active_trade — confirmé ~26 500 messages
Redis accumulés le 2026-08-10, worker `--pool=solo` mono-thread,
penny_sniper_alert seul prenant 12-14s et revenant toutes les 20-40s).

Solution additive (aucune tâche existante reroutée ou modifiée) :
- `CELERY_TASK_ROUTES` dans `settings.py` : `analysis.tasks.run_daily_scan`
  et `analysis.tasks.check_rejection_outcomes` → queue `analysis_queue`.
- Nouveau service `celery_worker_analysis` dans `docker-compose.yml`,
  ne consomme que `analysis_queue`, `--pool=solo --concurrency=1`
  (même profil que le worker principal, par cohérence). Le worker
  existant (`celery_worker`) est inchangé, non redémarré, non affecté
  (confirmé : uptime continu, aucune tâche haute-fréquence interrompue
  pendant/après le déploiement).
- Testé en direct : `run_daily_scan.delay(...)` reçu et traité par
  `celery_worker_analysis` (logs confirmés), jamais vu par
  `celery_worker` (grep confirmé à 0 occurrence).

**Reste à confirmer demain matin** : que le scan Beat de 9h45 ET
(`morning-fundamental-scan`) utilise bien la nouvelle queue et se
termine sans être bloqué par le backlog du worker principal — c'est la
vraie validation en conditions réelles (Beat, pas un déclenchement
manuel).

**Chantier plus large, toujours différé** (ne pas faire sans le même
niveau de rigueur que le reste de la session) : le worker principal
lui-même reste mono-thread (`--pool=solo`) et le backlog sur la queue
par défaut (~26 500 messages au 2026-08-10) n'a pas été réduit — cette
solution résout uniquement l'isolation du scan analysis, pas la
capacité globale du pipeline de trading. Revisiter la concurrence/
routage du worker principal seulement après la fenêtre d'observation
24-48h du module analysis, comme décidé explicitement.

---

## 6. Volume `TaskRunLog` — rétention déjà en place, verbosité pas touchée ce soir
**Statut : évalué, PAS implémenté (portée trop large pour ce soir).**

**Rétention** : vérifiée, existe déjà — `cleanup_task_run_logs`
(`portfolio/tasks.py:2140`), planifiée tous les jours à 2h15
(`cleanup-taskrunlog-daily` dans `CELERY_BEAT_SCHEDULE`), purge tout ce
qui a plus de `TASKRUNLOG_RETENTION_DAYS` (défaut 30j, configurable via
env var). Rien à faire ici — pas de purge illimitée comme supposé au
départ.

**Verbosité** : mesurée, pas corrigée. Au 2026-08-10 20h55 ET :
213 348 lignes au total, 8 044 sur les dernières 24h, dont **39.8%
(3 202) sont des succès triviaux <1s**. La rétention 30j empêche une
croissance illimitée, mais le volume quotidien reste élevé pour un
diagnostic manuel (ce qui a effectivement ralenti l'audit de ce soir).

Piste identifiée mais **pas implémentée** : `_task_log_start`/
`_task_log_finish` (`portfolio/tasks.py:2122-2136`) sont les 2 fonctions
qui écrivent dans `TaskRunLog`, appelées identiquement par les ~50
tâches du pipeline (trading ET analysis). On pourrait sauter l'écriture
en base pour un succès rapide et routinier (ex. `duration_ms < 1000` et
`status == 'SUCCESS'`), tout en gardant l'écriture systématique pour
tout échec et toute tâche lente (>1s). **Non fait ce soir** parce que
ces 2 fonctions sont un point de passage partagé par l'ensemble du
pipeline de trading — même un changement "pur logging" y touche le même
chemin de code que tout le reste, ce qui sort du cadre "additif sans
risque" fixé pour ce soir. À reprendre avec la même rigueur que le
reste (tests, vérification qu'aucune tâche ne dépend d'un comportement
d'écriture systématique) lors du chantier Celery plus large (item 5).
