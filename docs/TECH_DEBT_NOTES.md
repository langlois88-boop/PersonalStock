# Notes techniques en attente (non corrigées intentionnellement)

Fichier de suivi pour les bugs/chantiers identifiés mais volontairement
**pas corrigés sur le coup**, pour reprise sans avoir à reconstituer le
contexte. Chaque entrée : ce qu'on sait avec certitude (vérifié), ce
qu'on suppose (hypothèse), et ce qui reste à faire.

---

## 1. [CORRIGÉ le 2026-08-10 ~21h40 UTC] `LabPositionMarkManualView` écrase `manual_note` sans garde
**Statut : fermé. Commit `4547f7c6`, déployé, testé, 0 régression.**

`position.manual_note = request.data.get("note", "")` était inconditionnel.
Cliquer "Je crois en celle-là" sur une position qui avait déjà une note de
traçabilité (ex. les 4 positions BRCC/BTO.TO/NVDA/SLQT nettoyées après le
bug du scan bugué du 2026-08-09) écrasait silencieusement cette note.

**Fix** : ne remplace `manual_note` que si une nouvelle note non vide
(après `.strip()`) est fournie ; sinon la note existante est conservée.
2 tests de régression ajoutés (`analysis/tests.py`) : ré-appel sans
`"note"`, et avec une note blanche (espaces seulement) — les deux
préservent la note existante. Suite complète (`manage.py test`) : 0
régression, seulement les 4 erreurs pytest préexistantes et sans
rapport (`ml_engine`, module jamais installé).

Fichier : `portfolio_backend/analysis/views.py` (classe `LabPositionMarkManualView`).

---

## 2. [CORRIGÉ le 2026-08-10 ~22h00 UTC] `monitor_my_portfolio` échouait à 100%
**Statut : fermé. Cause confirmée en direct contre le vrai SDK, pas
supposée. Commits `e1fa9f45` + `e6cf3a45` (fix test), déployé, 0 régression.**

Cause réelle, reproduite en isolant l'appel fautif (`docker compose exec
backend python manage.py shell`) contre le **vrai SDK Alpaca**, pas
juste par lecture de code :
```python
>>> from portfolio.market_data import _timeframe_from_interval
>>> _timeframe_from_interval('60m')
ValueError: Second or Minute units can only be used with amounts between 1-59.
```
Message identique caractère pour caractère à celui de `TaskRunLog`.
**Ce n'était pas yfinance** (l'hypothèse d'hier pointait dans la bonne
direction sur `'60m'` mais visait la mauvaise librairie) : `monitor_my_
portfolio` utilise le wrapper `portfolio.market_data.Ticker` (aliasé
`yf` dans `tasks.py`), qui construit un `TimeFrame(60, TimeFrameUnit.
Minute)` pour l'API Alpaca — rejeté par construction, l'unité Minute
n'acceptant que 1-59. La ligne `interval = '60m' if is_core else '15m'`
(tasks.py ~12561) était bien le déclencheur, mais le vrai bug était dans
`_timeframe_from_interval()` (`market_data.py:122`), qui ne gérait pas
le cas 60 minutes = 1 heure — contrairement à la branche `'h'` juste en
dessous qui gère déjà les heures correctement.

**Fix** : quand le nombre de minutes est un multiple propre de 60,
`_timeframe_from_interval` retourne `TimeFrame(N // 60, TimeFrameUnit.
Hour)` au lieu de `TimeFrame(N, TimeFrameUnit.Minute)`. Grep sur tout le
fichier : `'60m'` n'a qu'un seul site d'appel (`monitor_my_portfolio`)
et `_timeframe_from_interval` qu'un seul appelant — fix contenu, pas de
risque sur d'autres usages (1m/5m/15m/30m/1h/2h non affectés, testés).

5 tests de régression (`tests_market_data_timeframe.py`) : `60m`→1h,
`120m`→2h, intervalles minute/heure ordinaires inchangés, défaut→Day.
Vérifié rigoureusement que ces tests échouent contre l'ancien code avec
la traceback exacte de production (`alpaca/data/timeframe.py:86
validate_timeframe`) avant de confirmer le fix.

---

## 3. [CORRIGÉ le 2026-08-10 ~21h50 UTC] `monitor_hive_trade` — crash DataFrame ambigu
**Statut : fermé. Commit `1a71df9e`, déployé, testé, 0 régression.**

Cause confirmée par lecture directe du code (pas une hypothèse) :
`latest_price` était réutilisée pour porter à la fois le DataFrame
intermédiaire de `yf.Ticker(...).history(...)` et le prix final (float).
Quand `get_latest_trade_price()` renvoyait `None` ET que le DataFrame
yfinance était vide, `latest_price` restait un DataFrame vide au lieu de
retomber sur `None` — le check `if latest_price is None:` ne l'attrapait
donc pas, et plus loin `latest_price >= limit_price` levait "The truth
value of a DataFrame is ambiguous" (message identique à celui vu en
rafale le 2026-08-10 entre 19h58:34-19h59:32 UTC, puis plus revu — cause
de l'intermittence : ne se déclenche que quand Alpaca ET yfinance
échouent tous les deux pour le même appel).

**Fix** : variable dédiée (`hist_1m`) pour le DataFrame intermédiaire,
`latest_price` ne finit plus jamais que comme float ou `None`. Grep sur
tout le fichier : aucun autre site avec ce pattern de réutilisation de
variable, ni aucun autre test booléen direct sur un DataFrame sans
`.empty`/`is None` — cas isolé, pas copié-collé ailleurs.

2 tests de régression (`tests_monitor_hive_trade.py`) : DataFrame vide
→ `'no_price'` (au lieu du crash), DataFrame non vide → prix extrait
normalement. Vérifié rigoureusement que ces tests échouent contre
l'ancien code (`status='failed'` au lieu de `'no_price'`) avant de
confirmer le fix.

---

## 4. `tsx-guardian-30s` — nom trompeur, tourne 24/7 pas juste en heures de marché
**Statut : diagnostiqué (voir item 7), correction EXPLICITEMENT mise en
pause — ne pas resserrer l'horaire ni renommer avant que l'item 7 soit
clarifié.**

Mise à jour 2026-08-10 ~21h35 UTC : en creusant ce point (via un
diagnostic hors-marché plus large sur `penny_sniper_alert`/
`monitor_active_trade`/`monitor_hive_trade`), découverte d'un bug bien
plus profond et pré-existant touchant `monitor_active_trade` (et
d'autres tâches) — voir **item 7**. Un simple resserrement d'horaire ou
renommage ici masquerait le symptôme sans corriger la cause, donc
explicitement pas fait tant que l'item 7 n'est pas clarifié.

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

## 5. [FAIT ET VALIDÉ le 2026-08-10] Queue Celery dédiée pour le module analysis
**Statut : implémenté, corrigé (voir "Suite"), et testé en conditions Beat
réelles (voir "Validation finale" tout en bas) — plus rien d'ouvert ici.**

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

### Suite (même soir, ~21h05-21h12 UTC) : `celery_beat` n'avait pas été redémarré

**Bug réel trouvé en vérifiant le scan de fermeture (16h30 ET) du jour
même** : le conteneur `celery_beat` tournait depuis 04h05 UTC (avant le
changement de `settings.py`) et n'avait **jamais été recréé**. Vérifié
directement en interrogeant ses settings en mémoire :
`CELERY_TASK_ROUTES` → `MISSING`. Conséquence concrète : **ni le scan
du matin (9h45 ET) ni celui de fermeture (16h30 ET) de ce jour n'ont
utilisé `analysis_queue`** — les deux ont été publiés sur la queue par
défaut (`celery`) et sont restés coincés derrière le backlog, exactement
comme avant le fix de ce soir. Retrouvés dans Redis : 3× message
`run_daily_scan` (2 avec `kwargs={}`, signature exacte d'un dispatch
Beat — très probablement les scans 9h45 et 16h30 d'aujourd'hui) + 1×
`check_rejection_outcomes`, tous coincés dans la queue `celery`
(~26 100 messages).

**Correctif** : `docker compose up -d --force-recreate celery_beat`
(le même piège que pour le frontend plus tôt cette session : `--build`
seul ne recrée pas le conteneur avec cette version de Docker Compose —
`up -d --build` avait "buildé" la nouvelle image sans recréer le
conteneur, `--force-recreate` était nécessaire). Vérifié après coup :
nouvelle image chargée, `CELERY_TASK_ROUTES` bien présent, Beat a
recommencé à dispatcher normalement (`Sending due task tsx-guardian-30s`
confirmé ~2 min après le redémarrage). Worker principal non affecté
(uptime continu 17h+, aucune tâche haute-fréquence interrompue).

**Message bénin observé au redémarrage** : `BDB0210
celerybeat-schedule.db: metadata page checksum error` au démarrage —
pas de crash, pas de boucle de redémarrage (`RestartCount=0`),
processus actif (CPU non nul). Piste : un fichier
`portfolio_backend/celerybeat-schedule.db` non versionné (16 384
octets, daté du 23 février) traîne dans le contexte de build sur le NAS
et se fait copier dans chaque image via `COPY . /app` — il n'est pas le
fichier réellement utilisé par Celery (qui écrit/lit
`celerybeat-schedule` sans extension, recréé à chaque démarrage), donc
probablement un résidu mort et inoffensif plutôt que la cause du
message. Pas creusé plus loin — à surveiller si le message réapparaît
ou si un vrai souci de planification est observé.

**Non traité, laissé tel quel** : les 4 messages déjà coincés dans la
queue `celery` par défaut (voir ci-dessus) n'ont **pas** été extraits ou
redirigés manuellement — manipuler des messages Redis au milieu de
~26 000 messages du pipeline trading sort du cadre "additif sans
risque" de ce soir. Ils finiront par être traités (en retard) par le
worker principal, ou resteront en attente indéfiniment si jamais
purgés/expirés — sans impact connu au-delà du délai, puisque le scan de
fermeture d'aujourd'hui a de toute façon déjà un résultat complet et à
jour via le déclenchement manuel de ce soir (`ScanRun id=3`,
6/6 candidats traités avec succès).

### Validation finale (même soir, 21h13-21h27 UTC) : test via un vrai déclenchement Beat, pas un `.delay()` manuel

**Pourquoi ce test était nécessaire** : tout ce qui précède prouvait que
Beat *avait la bonne config chargée*, pas qu'un scan déclenché *par Beat
lui-même* passait réellement par `analysis_queue` — la distinction
compte, puisque c'est exactement ce qui avait été raté deux fois de
suite le même jour (voir "Suite" ci-dessus).

**Méthode** : ajout d'un créneau `CELERY_BEAT_SCHEDULE` temporaire,
clairement marqué `TEMPORARY-test-analysis-queue-routing` dans le code
et le message de commit, fixé à 17h25 ET (~11 min dans le futur au
moment du commit, le temps de rebuild+déployer) :
- Commit d'ajout : `ac8bb493`
- Déployé avec `docker compose up -d --force-recreate celery_beat`
  (confirmé cette fois : `Container ... Recreate/Recreated` dans les
  logs, pas juste `Built`)
- Baseline avant déclenchement : `LLEN analysis_queue` = 0,
  `ScanRun` max id = 3, worker principal `Up 17 hours`.

**Résultat, chronométré à la seconde près** :
| Horodatage (UTC) | Événement |
|---|---|
| 21:25:00,000 | `celery_beat` : `Scheduler: Sending due task TEMPORARY-test-analysis-queue-routing` |
| 21:25:00,004 | `celery_worker_analysis` : `Task analysis.tasks.run_daily_scan[d5711eba-...] received` |
| 21:25:00,019 | Nouveau `ScanRun` id=4 créé en base |

Vérifications croisées :
- Worker principal (`celery_worker`) : **0 occurrence** de la tâche
  dans ses logs sur la fenêtre du test — jamais reçue.
- Queue par défaut (`celery`) : toujours 3 messages `run_daily_scan`
  résiduels (les mêmes qu'avant, voir "Suite" ci-dessus) — aucun
  nouveau message n'y a atterri, la tâche est bien passée par
  `analysis_queue` de bout en bout.

**Nettoyage** : créneau temporaire retiré immédiatement après
confirmation, planning remis à l'état réel (`morning-fundamental-scan`
9h45 ET, `daily-fundamental-scan` 16h30 ET, `weekly-rejection-
outcome-check` lundi 6h00) — commit de retrait : `1b60142d`. Redéployé
avec `--force-recreate` explicite (re-vérifié via `docker inspect` :
nouveau `StartedAt`, nouvelle image) pour repartir sur une base propre
avant le vrai scan de demain.

**À retenir si un problème similaire ressurgit** : après tout changement
touchant `CELERY_TASK_ROUTES` ou `CELERY_BEAT_SCHEDULE`, la première
chose à vérifier n'est pas "le code est-il correct" (il l'était dès le
départ ici) mais **"quel conteneur a réellement rechargé cette
config ?"** — interroger `settings.CELERY_TASK_ROUTES` directement
depuis l'intérieur de `celery_beat` (pas juste `celery_worker`) et
vérifier `docker inspect --format '{{.State.StartedAt}}'` contre l'heure
du dernier commit pertinent. Un `.delay()` manuel ou un test depuis le
worker ne suffit pas à couvrir ce cas — seul un vrai déclenchement Beat
(même via un créneau temporaire jetable comme ici) le prouve.

**Chantier plus large, toujours différé** (ne pas faire sans le même
niveau de rigueur que le reste de la session) : le worker principal
lui-même reste mono-thread (`--pool=solo`) et le backlog sur la queue
par défaut (~26 500 messages au 2026-08-10) n'a pas été réduit — cette
solution résout uniquement l'isolation du scan analysis, pas la
capacité globale du pipeline de trading. Revisiter la concurrence/
routage du worker principal seulement après la fenêtre d'observation
24-48h du module analysis, comme décidé explicitement.

---

## 6. [CORRIGÉ le 2026-08-10 ~22h20 UTC] Volume `TaskRunLog` — élagage des succès rapides
**Statut : fermé. Commit `78a94d6d`, déployé, testé, vérifié en conditions
réelles sur le NAS (10 306 lignes élaguées lors du premier run manuel).**

**Plan initial révisé avant d'implémenter** : l'idée d'hier (sauter/
supprimer l'écriture `TaskRunLog` pour les succès <1s directement dans
`_task_log_finish`) aurait cassé le dashboard de statut système
(`portfolio/views.py` ~8804, `.order_by('-started_at').first()` par
tâche) — vérifié avec les vraies durées avant de coder : 5 des 11
tâches surveillées sont dominées par des exécutions <1s
(`fetch_finnhub_news_daily` avg 5.6ms/182 runs, etc.), donc tout
supprimer aurait fait passer leur carte de statut à `UNKNOWN`.

**Design retenu** : `_prune_fast_success_task_run_logs()`, appelée
depuis le job de nettoyage quotidien existant (`cleanup_task_run_logs`),
PAS depuis le chemin chaud partagé par ~50 tâches. Garde les
`TASKRUNLOG_FAST_SUCCESS_KEEP_RECENT` (défaut 3) lignes les plus
récentes par tâche parmi les succès `< TASKRUNLOG_FAST_SUCCESS_MS`
(défaut 1000ms) ; les échecs et tâches lentes ne sont jamais touchés.
5 tests de régression, vérifiés en échouant contre le code d'avant
(`AttributeError`, fonction inexistante) avant de confirmer le fix.

**Découverte en vérifiant en conditions réelles (pas un bug introduit
ce soir)** : en déclenchant `cleanup_task_run_logs()` manuellement sur
le NAS pour valider, la rétention 30j *déjà existante* (jamais touchée
ce soir) a supprimé 193 569 lignes d'un coup — signe qu'elle n'avait
probablement jamais tourné via Beat. Cause trouvée : `settings.py`
ligne 581, `if not AFTER_HOURS_TASKS_ENABLED:` retire
`cleanup-taskrunlog-daily` (et 19 autres tâches : retraining nocturne,
rollback modèle, rapports hebdo...) de `CELERY_BEAT_SCHEDULE`.
`AFTER_HOURS_TASKS_ENABLED` n'est pas défini dans `deploy/.env` sur le
NAS → reste à `false` par défaut → ce lot de 20 tâches est désactivé
**depuis toujours** sur cette instance, pas cassé par du travail
récent. Le fix de ce soir est déployé et fonctionne (vérifié par
déclenchement manuel), mais **ne tournera pas tout seul** tant que ce
flag reste à `false` ou qu'une entrée de planning dédiée n'est pas
créée en dehors de ce bloc conditionnel — à décider : activer tout le
lot (retraining ML compris, pas juste le nettoyage de logs) ou sortir
`cleanup-taskrunlog-daily` de la liste conditionnelle pour qu'il tourne
indépendamment.

### Suite (même soir, ~22h20-22h27 UTC) : décision prise, `cleanup-taskrunlog-daily` + `cleanup-system-logs-weekly` sortis du flag

**Décision explicite** : ne pas activer `AFTER_HOURS_TASKS_ENABLED` en
bloc. En listant les 21 tâches derrière ce flag pour éclairer la
décision, la répartition réelle est apparue :
- **Ménage/logs** (aucun impact trading) : `cleanup_task_run_logs`,
  `cleanup_system_logs` — juste des `DELETE` sur de vieilles lignes.
- **Rapports/notifications** (informationnel, Telegram/PDF/email) :
  journal quotidien, briefing dominical, calendrier économique, etc.
- **Gouvernance des modèles ML** (impact réel sur le trading live) :
  `auto_rollback_models_daily` (revient à une version antérieure du
  modèle), `auto_retrain_on_drift_daily` / `backtest_retrain_guard` /
  `retrain_from_paper_trades_daily` (peuvent promouvoir un nouveau
  modèle). Activer le flag en bloc aurait réveillé tout un système
  autonome de retraining/rollback dormant depuis toujours sur cette
  instance — pas un simple "activer le nettoyage de logs".

**Bug trouvé en chemin, noté mais PAS corrigé** : `detect_model_drift`
(`model-drift-check-daily`) est définie **imbriquée à l'intérieur de
`_alpaca_position_avg_price`** (`portfolio/tasks.py:6225-6280`,
copier-coller mal indenté) au lieu d'être une fonction de niveau
module. Son enregistrement Celery dépend de si `_alpaca_position_avg_
price` a déjà été appelée dans le worker avant que Beat ne dispatche
la tâche — fragile indépendamment du flag. Laissé tel quel, décision
explicite de ne toucher que le carve-out du ménage ce soir.

**Fait** : `cleanup-taskrunlog-daily` et `cleanup-system-logs-weekly`
sortis de la liste conditionnelle dans `settings.py` (commit
`e8034210`), tournent maintenant indépendamment du flag. Le reste des
19 tâches (rapports + gouvernance ML) reste désactivé, inchangé.
Déployé avec `--force-recreate` sur `backend`+`celery_worker`+
`celery_beat` (les 3 en ont besoin : Beat pour le nouveau planning, le
worker pour exécuter le code, backend pour cohérence). Vérifié dans le
conteneur `celery_beat` lui-même : les 2 tâches de ménage présentes,
`model-rollback-daily` bien absent. 2 tests de régression
(`tests_after_hours_schedule_gate.py`).

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

---

## 7. [PRIORITÉ ÉLEVÉE, découvert le 2026-08-10 ~21h35 UTC] Beat déclenche plusieurs tâches en rafale toutes les ~3-3.5h — pré-existant, pas causé par les changements de ce soir

**Statut : diagnostiqué, cause probable identifiée mais PAS confirmée,
volontairement PAS touché ce soir (bug plus profond que prévu — arrêt
et documentation, comme convenu).**

**Découvert en vérifiant** pourquoi `monitor_active_trade` apparaissait
~12 fois à la même seconde dans les logs. En creusant sur une fenêtre de
40 min : **520 exécutions au lieu des ~20 attendues** (cycle configuré
120s), avec une rafale de **372 exécutions en 9 secondes** (21h23:24-
21h23:33 UTC), 100% `monitor_active_trade`, aucune autre tâche mêlée.

**Confirmé pré-existant, pas lié à mes redémarrages de `celery_beat` ce
soir** : le même pattern de rafales (~90-100 exécutions en 30-60s)
apparaît à **03h28, 07h04, et 10h16 UTC** aujourd'hui — sur une période
où le conteneur `celery_beat` tournait sans interruption depuis 04h05
(donc *avant* même son propre premier démarrage pour 03h28, ce qui
suggère que le pattern existait déjà la veille aussi, pas vérifié plus
loin).

**Ça ne touche pas qu'une seule tâche** : `send_morning_log_email`
(censée tourner 1×/jour à 8h05 ET) et `nightly_summary_report` (1×/jour
à 8h00 UTC) se sont déclenchées **7 et 9 fois en 48h respectivement**, à
des horaires qui collent exactement aux mêmes créneaux de rafale
(03h29, 07h05, 10h17, 14h5x, 18h0x, 21h2x UTC) — pas du tout leurs
heures configurées. C'est donc un dysfonctionnement de Beat qui affecte
plusieurs entrées de `CELERY_BEAT_SCHEDULE` simultanément (au moins un
`timedelta` et deux `crontab`), pas un bug spécifique à une tâche.

**Hypothèse la plus probable (pas confirmée)** : le message `BDB0210
celerybeat-schedule.db: metadata page checksum error`, observé à chaque
redémarrage de `celery_beat` ce soir et noté hier comme "probablement
bénin" (item 5, section Suite) — ne l'est vraisemblablement **pas**.
Le fichier de planning persistant (backend BerkeleyDB/shelve) semble se
corrompre ou perdre son état périodiquement (~toutes les 3-3.5h, cause
du cycle exacte inconnue), ce qui ferait "oublier" à Beat le
`last_run_at` de plusieurs entrées à la fois et les redéclare toutes
"dues" en même temps — sans que le conteneur lui-même redémarre (les
rafales de 07h04/10h16 ont eu lieu sans aucun restart du conteneur).

**Impact potentiel** : si ce pattern se répète ~7-8×/jour sur
plusieurs tâches haute-fréquence (`tsx-guardian-30s` notamment, qui n'a
aucune restriction horaire dans Beat et peut donc rafaler 24h/24),
c'est un contributeur plausible et significatif au backlog de ~26 000
messages Redis documenté dans l'item 5 — potentiellement plus important
que le simple volume de dispatch normal des tâches haute-fréquence.

**Pourquoi ce n'est PAS corrigé ce soir** : la cause exacte n'est pas
confirmée (juste une forte corrélation temporelle avec le message
checksum, pas une preuve). Toucher au mécanisme de persistance de Beat
sans diagnostic confirmé risquerait d'affecter le scheduler partagé par
l'ensemble du pipeline de trading — exactement le genre de changement
qui demande le même niveau de rigueur que le reste de la session,
pas une correction à l'aveugle en fin de soirée.

**À faire pour la prochaine session dédiée à ce sujet** :
1. Confirmer si le fichier `celerybeat-schedule` (backend shelve/bsddb,
   éphémère — pas sur un volume Docker persistant, voir item 5) est la
   vraie cause, ou si c'est autre chose (charge mémoire, comportement du
   `PersistentScheduler` avec beaucoup d'entrées, etc.).
2. Vérifier si passer à un scheduler différent (ex. sans état persisté
   sur disque, ou avec un volume Docker dédié pour que l'état survive
   aux redémarrages) résout le problème.
3. Une fois la vraie cause confirmée, mesurer l'impact réel sur le
   volume de la queue par défaut avant de décider si ça mérite d'être
   traité en priorité par rapport au chantier plus large de l'item 5
   (concurrence du worker principal).
4. **Ne pas resserrer les horaires des tâches individuelles
   (`tsx-guardian-30s` y compris) tant que ce point n'est pas
   clarifié** — un resserrement d'horaire masquerait le symptôme sans
   corriger la cause, et compliquerait le diagnostic futur.

---

## 8. `_create_lab_position` ne vérifie jamais une position déjà ouverte pour le même ticker
**Statut : ouvert, découvert le 2026-08-10 ~22h40 UTC. Symptôme
nettoyé, cause PAS corrigée (décision explicite de l'utilisateur).**

**Confirmé** (`analysis/tasks.py:203`, fonction `_create_lab_position`) :
aucun check `FundamentalLabPosition.objects.filter(ticker=..., preset=...,
is_open=True).exists()` avant de créer une nouvelle position + un nouveau
`PaperTrade`. Chaque appel de `run_daily_scan` qui reconfirme un ticker
déjà détenu crée une position **en plus**, pas une mise à jour/renfort de
l'existante.

**Découvert concrètement** : 3 scans manuels lancés le même soir (20h26,
20h59, 21h28 UTC — tests de validation du routage Celery Beat, voir
item 5) ont chacun re-confirmé BTO.TO, créant 3 `FundamentalLabPosition`
+ 3 `PaperTrade` OPEN distincts (14 actions @ 7.04$ chacun) au lieu d'une
seule — ~296$ de capital papier engagé au lieu de ~99$ prévus
(`FUNDAMENTAL_LAB_POSITION_SIZE=100` par défaut).

**Nettoyé** (pas la cause, juste le symptôme, sur décision explicite) :
les 2 positions en trop (`PaperTrade` id 16 et 19, `FundamentalLabPosition`
id 7 et 10) fermées à prix plat (pnl=0, outcome=None) avec note de
traçabilité expliquant l'origine. La position réelle conservée :
`PaperTrade` id=22 / `FundamentalLabPosition` id=13.

**Portée réelle du bug, au-delà de ce soir** : ce n'est pas spécifique
aux tests de ce soir — n'importe quel jour où `run_daily_scan` tourne
plus d'une fois (Beat matin + Beat fermeture, tous les deux configurés
sur `analysis.tasks.run_daily_scan` sans argument, donc TOUS les
presets actifs à chaque fois) et qu'un même ticker reste "confirmed"
d'un scan à l'autre, une nouvelle position se crée. Avec les 2 scans/
jour actuels (9h45 + 16h30 ET), c'est un risque quotidien, pas un cas
isolé de test.

**À faire** : dans `_create_lab_position` (ou juste avant l'appel),
vérifier s'il existe déjà une `FundamentalLabPosition` ouverte pour ce
`ticker` + `preset` avant de créer une nouvelle ; si oui, soit ignorer
(pas de doublon), soit lier le nouveau `ScanResult` à la position
existante comme reconfirmation plutôt que d'ouvrir un nouvel ordre.
Décision de design à prendre avec l'utilisateur (quel comportement
voulu exactement) avant de coder.

---

## 9. Circuit breaker (`_daily_equity_circuit_breaker`) utilise le cache Django par défaut (non persistant, pas Redis)
**Statut : ouvert, deux faux déclenchements confirmés ce soir (un 3e
touchant AI_PENNY en plus d'AI_BLUECHIP), cause structurelle toujours
pas corrigée, portée revue à la hausse après le 2e épisode.**

**Confirmé** : pas de bloc `CACHES` dans `settings.py` → Django utilise
`LocMemCache` par défaut (cache en mémoire du process, PAS Redis) pour
`_daily_equity_circuit_breaker` (`portfolio/tasks.py:4575`), qui stocke
la référence d'équité du jour (`daily_equity_base:{sandbox}:{date}`) et
l'état de déclenchement (`daily_equity_trip:{sandbox}:{date}`) dans ce
cache.

**Épisode 1** : "Circuit breaker triggered for AI_BLUECHIP" à 22h21:25
UTC, capital loggé = 23803.36$. Recalculé la formule exacte
(`initial_capital + closed_pnl`) : résultat identique — le capital
n'avait pas bougé (0 position ouverte, 0 trade fermé). Coïncide avec
mon redémarrage de `celery_worker` à 21h59:35 UTC (~22 min avant).
Hypothèse initiale : lecture transitoire pendant le redémarrage du
process.

**Épisode 2, ~18 min plus tard** : "Circuit breaker triggered for
AI_BLUECHIP" (encore, 22h39:52 UTC, même capital 23803.36$) **et**
"Circuit breaker triggered for AI_PENNY" (22h40:14 UTC, capital
10038.11$). Recalculé les deux : identiques aux valeurs loguées, 0
position ouverte dans les deux sandboxes — encore un faux positif
confirmé. **Important : `docker inspect` confirme qu'aucun redémarrage
de `celery_worker` n'a eu lieu entre les deux épisodes** (`RestartCount:
0`, `StartedAt` inchangé depuis 22h27:17 UTC) — donc l'hypothèse
"uniquement causé par un redémarrage" de l'épisode 1 est **incomplète**.
Le cache s'est vidé une seconde fois (`baseline`/`trigger` à `None` au
moment de vérifier, ~23h01 UTC) sans redémarrage.

**Hypothèse révisée** : `LocMemCache` de Django effectue par défaut une
éviction aléatoire ("culling") quand il dépasse son nombre d'entrées
max (défaut 300, `CULL_FREQUENCY=3`) — avec ~50 tâches qui écrivent des
clés de cache diverses toute la journée (sentiment, prix, cooldowns
d'alertes...), ce plafond peut être atteint et faire disparaître
n'importe quelle clé, y compris celles du circuit breaker, **à tout
moment de la journée, pas seulement après un redémarrage**. Pas
confirmé formellement (pas creusé le code source de `LocMemCache` ni
compté les clés actives ce soir), mais cohérent avec l'observation.

**Risque réel, révisé à la hausse** : ce mécanisme de sécurité (censé
arrêter les nouvelles entrées d'un sandbox après -3% de drawdown
journalier) peut se déclencher faussement (comme ce soir, 2 fois) OU
perdre silencieusement sa protection (le flag de déclenchement
disparaît, permettant de nouvelles entrées alors qu'un vrai -3% avait
été détecté plus tôt) — et ce **n'importe quand dans la journée**, pas
seulement lors d'un déploiement. Un vrai déclenchement pourrait donc ne
protéger que quelques minutes avant d'être oublié.

**Impact ce soir** : aucun, marché fermé (aucune nouvelle entrée
n'aurait été tentée de toute façon), aucune position réelle affectée
(paper trading), et la clé de cache est datée du jour
(`daily_equity_trip:{sandbox}:{date}`) donc **ne se reporte pas sur
demain** même sans reset manuel.

**À faire** : migrer ce cache (et tout autre état journalier critique
au risque similaire — à auditer) vers le backend Redis déjà utilisé
comme broker Celery, pour qu'il survive aux redémarrages ET à l'éviction
`LocMemCache`. Vérifier aussi si un `CACHES` explicite avec
`max_entries` plus élevé suffirait en attendant, ou si un cache Redis
est vraiment nécessaire. Pas une urgence pour ce soir (aucun trade réel
en jeu), mais à traiter avant que ce mécanisme ait besoin de protéger
un vrai drawdown pendant les heures de marché.

---

## 10. Demande : FUNDAMENTAL_LAB avec son propre compte Alpaca (pas juste son propre sandbox interne)
**Statut : clés reçues et validées le 2026-08-11 ~20h UTC (plus bloqué
sur ce point), implémentation toujours pas commencée.**

**Clés stockées** : `ALPACA_API_KEY_FUNDAMENTAL_LAB` /
`ALPACA_SECRET_KEY_FUNDAMENTAL_LAB` ajoutées dans `deploy/.env` sur le
NAS uniquement (jamais commitées, `deploy/.env` confirmé dans
`.gitignore`). Vérifiées en direct contre le vrai SDK Alpaca :
compte paper actif (`account_number=PA34ZP3JNN2T`), $50 000 de capital
de départ, complètement séparé du compte partagé existant
(`ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, utilisé par WATCHLIST/AI_
BLUECHIP/AI_PENNY). Rien câblé dans le code encore — reste tel que
décrit ci-dessous, à faire lors d'une session dédiée avec la même
rigueur que le reste.

**Ce qui existe déjà** : `FUNDAMENTAL_LAB` a déjà son propre sandbox
(`PaperTrade.sandbox='FUNDAMENTAL_LAB'`, isolé des autres en base), mais
**aucun sandbox du projet n'a de compte Alpaca séparé** — `WATCHLIST`,
`AI_BLUECHIP`, `AI_PENNY`, et `FUNDAMENTAL_LAB` (s'il y allait) partagent
tous la même paire `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`
(`portfolio/alpaca_data.py::_alpaca_trading_client()`, lit ces 2 env
vars sans jamais de variante par sandbox). Actuellement,
`FUNDAMENTAL_LAB` ne va même pas jusqu'à Alpaca du tout : `_create_lab_
position` (`analysis/tasks.py:203`) fixe `broker="SIM"` en dur —
positions purement simulées en base, jamais d'ordre réel envoyé (voir
aussi la note sur le commentaire trompeur à ce sujet, discuté juste
avant ce soir).

**Objectif réel de la demande (le "pourquoi", important pour la suite)** :
pas juste isoler `FUNDAMENTAL_LAB` pour isoler — l'objectif est de le
rendre **directement comparable** aux modèles ML qui tradent déjà
(BLUECHIP/PENNY), avec le même genre de suivi réel (compte Alpaca
dédié, comme les autres sandboxes actifs) plutôt qu'une simulation
séparée. À terme (pas pour cette itération), l'idée est aussi
d'utiliser ce que l'analyse fondamentale (Claude/DeepSeek) détecte
comme **signal supplémentaire pour les modèles ML eux-mêmes** — le but
final étant de rendre les modèles ML "plus intelligents", pas juste
d'avoir un screener fondamental à côté. Phrase de l'utilisateur coupée
en fin de message ("...peut-être même un jour ajouter des...") — à
clarifier/compléter à la reprise, ne pas supposer le détail exact.

**Ce qui bloque pour commencer** : l'utilisateur doit d'abord créer un
**second compte/environnement paper trading Alpaca** sur son dashboard
(possible d'avoir plusieurs comptes paper sous un même login, ou un
second signup) et récupérer la nouvelle paire de clés — impossible pour
moi de créer ce compte. Décidé explicitement de ne pas commencer
l'implémentation avant d'avoir ces clés en main, et même une fois
prêtes, de le faire "à tête reposée" (pas en fin de très longue
session) vu que ça touche l'exécution d'ordres réels (même en paper
trading), pas une simulation en base comme avant.

**Ce qu'il faudra faire une fois les clés en main** (esquisse, à affiner
à la reprise) :
1. Nouvelles env vars pour la paire de clés dédiée (ex.
   `ALPACA_API_KEY_FUNDAMENTAL_LAB` / `ALPACA_SECRET_KEY_FUNDAMENTAL_LAB`),
   sans toucher aux env vars partagées existantes.
2. Un client Alpaca séparé (pattern à ajouter dans
   `portfolio/alpaca_data.py`, ou un module dédié) instancié avec ces
   clés, distinct de `_alpaca_trading_client()` partagé.
3. Remplacer (ou compléter, à décider) le `broker="SIM"` codé en dur
   dans `_create_lab_position` par un vrai envoi d'ordre via ce nouveau
   client, en suivant le pattern déjà existant pour WATCHLIST/AI_
   BLUECHIP/AI_PENNY (`_execute_alpaca_paper_trades_for_sandbox`,
   `portfolio/tasks.py:~7100`) plutôt que d'inventer un nouveau
   mécanisme de synchronisation d'ordres.
4. Tests + vérification manuelle en dry-run avant tout déploiement réel
   — même rigueur que tout le reste de cette session (backup, dry-run,
   vérification avant/après).

## 11. [FAIT le 2026-08-11] Partie A du balayage large S&P500+TSX — implémentation + revue + corrections

**Contexte** : demande explicite de l'utilisateur (2026-08-10 soir) —
balayage hebdomadaire strict (ROE/ROIC/Dette-EBITDA/marge/P/E/PEG/FCF
Yield/Piotroski/Altman, tous obligatoires) sur S&P500+TSX, alimentant
un nouveau modèle `CustomWatchlistTicker`, avec une option
`universe_source="combined"` sur le screener quotidien — explicitement
PAS activée avant un premier cycle réel validé. Voir aussi [[10]] pour
le contexte stratégique complet (comparabilité avec les modèles ML,
signal futur).

**A1-A3** (2026-08-10/11) : source SP500/TSX réparée (3 bugs en chaîne —
`lxml` manquant, Wikipedia 403 sans User-Agent, `pd.read_html` sur
string brut ambigu avec un chemin de fichier — voir requirements.txt et
`portfolio/tasks.py::_read_wikipedia_tables`), 500 SP500 + 221 TSX
réels ; modèle `CustomWatchlistTicker` + migration appliquée ; filtre
strict (`analysis/services/broad_scan_filter.py`) avec Piotroski
F-Score et Altman Z-Score formules standard (1968/2000, pas improvisées).

**A4** : tâche `run_broad_index_scan` (`analysis/tasks.py`), Beat
hebdomadaire dimanche 20h ET (`weekly-broad-index-scan`, queue
`analysis_queue`).

**Revue externe avant premier déploiement (2026-08-11)** — 6 points
soulevés, traités comme suit :

1. **Contention de capital sur un compte Alpaca partagé** — non
   applicable : voir [[10]], FUNDAMENTAL_LAB aura son propre compte
   Alpaca dédié (`ALPACA_API_KEY_FUNDAMENTAL_LAB`, déjà reçu et validé,
   `account_number=PA34ZP3JNN2T`), complètement séparé du compte
   partagé WATCHLIST/AI_BLUECHIP/AI_PENNY. Ce point du prompt original
   était déjà résolu par une décision d'architecture antérieure, pas un
   trou dans le plan de ce soir.
2. **CORRIGÉ** : `consecutive_weeks_missed` s'incrémentait pour tout
   ticker absent des survivants, sans distinguer "recalé légitimement"
   de "scan planté/dégradé avant évaluation". Ajout de
   `MIN_SCAN_COVERAGE_RATIO=0.7` : sous ce seuil de couverture, le run
   est marqué `status='degraded'` et la décroissance entière est sautée
   pour ce cycle (les survivants trouvés sont quand même enregistrés —
   information positive individuellement fiable même sur un run
   dégradé).
3. **CORRIGÉ** : le garde-fou budget (>50 tickers actifs) n'était qu'un
   `logger.warning`, jamais bloquant. Remplacé par un plafond DUR
   (`MAX_ACTIVE_CUSTOM_TICKERS=50`) : au-delà, seuls les 50 meilleurs
   par score composite (Piotroski + Altman Z) restent actifs, le reste
   désactivé (`is_active=False`, jamais supprimé).
4. **Noté, non corrigé (pas urgent)** : redondance — un survivant du
   filtre strict repasse ensuite par `evaluate_ticker` (seuils plus
   souples) une fois dans l'univers combiné quotidien. Double sécurité
   intentionnelle, coût réseau/calcul en double mais pas incohérent.
5. **Validé** : Piotroski/Altman comparés à GuruFocus (2026-08-11).
   AAPL : Piotroski calculé=8 (réf. 9), Altman Z calculé=11.62 (réf.
   13.87) — même zone qualitative ("excellent" 7-9, très au-dessus de
   3.0). Ford : Altman Z calculé=0.80 (réf. 0.88) — même zone
   "détresse" (<1.8). Écarts expliqués par données annuelles
   fiscal-year-end (yfinance) vs mix trimestriel plus récent chez
   GuruFocus, et différences mineures de définition de lignes entre
   fournisseurs (normal pour ces 2 ratios, documenté dans la
   littérature) — classification qualitative correcte dans les 2 cas
   testés (extrême haut et extrême bas), pas une correspondance exacte
   au chiffre près.
6. **À reconfirmer avant Partie B** (voir aussi [[10]]) : maintenant que
   Partie B connecte FUNDAMENTAL_LAB à du vrai capital paper Alpaca (pas
   juste une ligne DB), vaut la peine de reconfirmer avec l'utilisateur
   si un plafond sur le nombre de positions ouvertes automatiquement par
   semaine est souhaité, en connaissance de cause de ce changement de
   nature (simulation → exécution réelle même en paper).

**Déploiement** : `docker compose build` + `--force-recreate` sur
`backend`, `celery_worker_analysis`, `celery_beat`, `celery_worker`
(les 4 partagent le même Dockerfile/contexte mais des images taguées
séparément — rebuilder seulement `backend` ne suffit PAS, leçon déjà
documentée ailleurs dans ce fichier, reconfirmée ce soir : le premier
`.delay()` de test a été rejeté par `celery_worker_analysis` avec
`KeyError` / "Received unregistered task" car son image n'avait pas
encore été reconstruite).

**A-test (déclenchement manuel réel)** : lancé le 2026-08-11 ~22h06 UTC
sur le NAS (`run_broad_index_scan.delay()`), univers confirmé = 720
tickers (500 SP500 + 221 TSX). Résultat en attente (tâche longue,
potentiellement plusieurs heures — voir mise à jour ci-dessous une fois
terminée).

`universe_source="combined"` reste **NON activé** sur aucun preset
jusqu'à validation du résultat de ce cycle.

**Nuances ajoutées par l'utilisateur en relecture (2026-08-11, même
soir)** :
- Point 2 (couverture 70%) : si le scan reste dégradé **plusieurs
  semaines de suite** (pas juste un incident isolé), les tickers déjà
  actifs ne décroissent jamais MAIS ne se rafraîchissent pas non plus —
  ils restent figés sur leur dernière vraie confirmation
  (`piotroski_score`/`altman_z_score`/`last_confirmed` obsolètes). Pas
  un problème dans l'immédiat (le comportement voulu reste "ne pas
  pénaliser à tort"), mais si un ticker actif semble ne plus se mettre
  à jour depuis longtemps, c'est la première piste à vérifier (regarder
  l'historique des `status` retournés par les runs récents, pas
  supposer un bug ailleurs).
- Point 5 (écarts Piotroski/Altman vs GuruFocus) : la cause la plus
  probable est la fraîcheur des données (annuel yfinance vs mix
  trimestriel plus récent), pas un bug de formule — confirmé par le
  fait que la classification qualitative tient dans les 2 cas testés.
  Vigilance à avoir : un ticker **limite** (score proche du seuil
  `STRICT_PIOTROSKI_MIN=7`, ex. 6 ou 7 pile) est le cas où ce même
  écart de fraîcheur pourrait faire basculer un vrai verdict (passe/ne
  passe pas) — si un faux négatif/positif suspect apparaît un jour sur
  un titre proche du seuil, vérifier la fraîcheur des états financiers
  annuels avant de suspecter la formule elle-même.
- Point 6 : confirmé rester **ouvert, pas résolu** — correctement
  séparé pour une vraie discussion avant Partie B (capital réel qui
  bouge automatiquement), pas oublié.
