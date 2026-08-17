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

## 4. [CORRIGÉ le 2026-08-13] `tsx-guardian-30s` — nom trompeur, tournait 24/7 pas juste en heures de marché
**Statut : fermé. Item 7 clarifié et corrigé le 2026-08-12 (débloquait
celui-ci) ; confirmé avec l'utilisateur le 2026-08-13 que le 24/7 était
un oubli, pas voulu (pas de surveillance overnight intentionnelle).**

**Fix** : renommé `tsx-guardian-30s` → `tsx-guardian-120s` (nom qui
reflète enfin le vrai intervalle), et remplacé `timedelta(seconds=120)`
par `crontab(minute='*/2', hour='9-16', day_of_week='mon-fri')` — même
cadence de 2 minutes, mais restreinte aux heures de marché comme
`active-signal-monitor-1min` juste au-dessus dans `settings.py`
(pattern déjà établi pour ce type de tâche, réutilisé plutôt
qu'inventé). Ne tourne plus inutilement marché fermé/weekend.

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

### Suite (même soir, 2026-08-10 21h47 EDT) : `detect_model_drift` corrigé quelques heures plus tard — jamais reflété ici avant le 2026-08-17

**Cette note a induit en erreur pendant une semaine** : relue le
2026-08-17 (demande "petit et rapide" pour corriger l'indentation), la
lecture du code en cours (`portfolio/tasks.py:6436` et suivantes) a
montré que `detect_model_drift` est **déjà** une fonction `@shared_task`
de niveau module, correctement indentée, avec un commentaire interne
daté du 2026-08-10 expliquant le fix. `git log` confirme : commit
`6375a25c` ("Fix detect_model_drift accidentally nested inside a dead
function"), le même soir que ce paragraphe, à 21h47 EDT — donc
**quelques heures après** que cette note ait été écrite (~18h20-18h27
EDT). Le fix a bien dédenté `detect_model_drift` au niveau module et
restauré `_alpaca_position_avg_price` à son corps d'origine, sans rien
changer à la logique interne de `detect_model_drift`.

Ce paragraphe n'a simplement jamais été mis à jour pour le refléter —
la tâche est restée listée comme un problème ouvert dans ce document
pendant une semaine alors qu'elle était déjà réglée en production.

**Revérifié le 2026-08-17** avant de clore ce point :
- `grep` sur `portfolio/tasks.py` confirme `detect_model_drift` définie
  seule, au niveau module (ligne 6437), séparée de
  `_alpaca_position_avg_price` (ligne 6426).
- `settings.py` confirme `'model-drift-check-daily'` bien présent dans
  `CELERY_BEAT_SCHEDULE` (ligne 413) — toujours dans le lot retiré par
  `AFTER_HOURS_TASKS_ENABLED` (décision item 12, inchangée : ne pas
  activer la gouvernance ML en bloc). La tâche ne tourne donc toujours
  pas automatiquement aujourd'hui, mais **si elle tournait**, elle
  s'enregistrerait et s'exécuterait correctement — ce n'est plus une
  mine dormante.
- `portfolio/tests_detect_model_drift.py` (créé par le fix du
  2026-08-10) rejoué : 6/6 OK.

**Aucun changement de code nécessaire aujourd'hui** — uniquement cette
correction de documentation, pour que ce fichier reste une source
fiable de "qu'est-ce qui reste à faire".

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

## 7. [CORRIGÉ le 2026-08-12 ~20h42 UTC] Beat déclenche plusieurs tâches en rafale toutes les ~3-3.5h — pré-existant, pas causé par les changements du 2026-08-10

**Statut : fermé. Hypothèse du 2026-08-10 (BDB0210 checksum error /
fichier de planning éphémère) CONFIRMÉE avec une preuve concrète, corrigée,
vérifiée en direct (persistance survivant à un `--force-recreate`).**

**Preuve concrète qui manquait le 2026-08-10** : la nuit du 2026-08-11/12,
`celery_beat` a été recréé 4 fois (déploiements Partie A/B) — chaque fois
avec le message `BDB0210 celerybeat-schedule.db: metadata page checksum
error`. Résultat mesuré : **3 exécutions hors-calendrier de
`run_daily_scan`** (03h31, 04h58, 06h43 UTC — aucune ne correspond aux
créneaux 9h45/16h30 ET configurés), soit exactement le mécanisme
suspecté : Beat "oublie" `last_run_at` pour plusieurs entrées à la fois
et les redéclare "dues" simultanément. Impact réel de ces 3 exécutions :
aucune position dupliquée (le garde-fou de l'item 8 a bloqué les 21
évaluations), mais ~65 000 tokens Claude gaspillés sur 13 appels.

**Cause racine exacte** (plus précise que l'hypothèse du 2026-08-10) :
le fichier persistant de Beat vivait dans le CWD du conteneur
(`/app/celerybeat-schedule.db`), jamais monté sur un volume Docker —
donc *toujours* perdu à la recréation du conteneur, pas seulement en
cas de corruption occasionnelle. Une variable d'env `CELERYBEAT_SCHEDULE`
préexistante dans `docker-compose.yml` laissait croire qu'un chemin
`/tmp/...` était déjà configuré, mais elle n'était lue par **aucun**
code du projet (pas une variable standard Celery) — donc sans effet
réel, le fichier restait toujours au chemin par défaut non monté.

**Fix appliqué** (`docker-compose.yml`) :
1. Nouveau volume nommé `celerybeat_schedule` monté sur `/celerybeat`.
2. `command:` du service `celery_beat` : ajout du flag CLI réel
   `--schedule=/celerybeat/celerybeat-schedule` (le seul mécanisme qui
   fonctionne, vérifié en direct — la variable d'env supprimée).

**Vérifié en direct sur le NAS avant de considérer le point clos** :
log de démarrage confirme `db -> /celerybeat/celerybeat-schedule` (plus
le chemin par défaut) ; fichier `celerybeat-schedule` (524 288 octets,
dbm réel) présent avec le bon timestamp ; un second `--force-recreate`
délibéré montre le **même** fichier, même timestamp, aucune erreur
`BDB0210` — persistance confirmée, pas juste supposée.

**Conséquence pratique** : les futurs redéploiements touchant
`celery_beat` (fréquents dans ce projet) n'entraîneront plus ce
comportement. `tsx-guardian-30s` (item 4) et le volume de la queue par
défaut (item 5/9) restent des chantiers séparés, non résolus par ce fix
— seule la cause des rafales/re-déclenchements hors-calendrier est
corrigée ici.

---

### Diagnostic original (2026-08-10, conservé pour l'historique du raisonnement)

**Statut original : diagnostiqué, cause probable identifiée mais PAS
confirmée, volontairement PAS touché ce soir-là (bug plus profond que
prévu — arrêt et documentation, comme convenu).**

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

## 8. [CORRIGÉ, découvert le 2026-08-10, fix jamais documenté ici avant le 2026-08-13] `_create_lab_position` ne vérifiait jamais une position déjà ouverte pour le même ticker
**Statut : fermé. Doc mise à jour le 2026-08-13 après vérification
directe du code (pas juste supposé) : le check anti-doublon est bien
présent et actif dans `_create_lab_position` (`analysis/tasks.py:277`).
Le fix a été appliqué à un moment de cette session sans que cette entrée
ne soit mise à jour en conséquence — corrigé maintenant pour que la doc
reflète l'état réel.**

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

## 9. [CORRIGÉ le 2026-08-13] Circuit breaker (`_daily_equity_circuit_breaker`) utilise le cache Django par défaut (non persistant, pas Redis)
**Statut : fermé. Troisième épisode réel le 2026-08-13 (redémarrage
complet du NAS) a fourni la confirmation manquante, corrigé le soir
même — état déplacé vers Redis, vérifié en conditions réelles.**

### Correctif (2026-08-13)

**Nouvelle preuve avant le fix** : le 2026-08-13, un redémarrage complet
du NAS (tous les containers, pas seulement PersonalStock) a coïncidé
avec un second déclenchement du breaker pour AI_BLUECHIP ET AI_PENNY,
~8h après leur premier déclenchement de la journée, avec un `capital`
**identique au chiffre près** aux deux occasions (23803.36$/10038.11$ —
`capital = initial_capital + closed_pnl`, inchangé car 0 trade fermé
entre les deux). Mathématiquement, une baseline correctement conservée
ne peut pas re-déclencher sur une valeur d'équité inchangée
(`equity_now >= baseline` reste vrai) — confirme que l'état a bien été
perdu, pas qu'un vrai second drawdown s'est produit.

**Fix** : `_daily_equity_circuit_breaker` et `reset_daily_equity_breaker`
(`portfolio/tasks.py`) réécrites pour utiliser un client Redis dédié
(`_circuit_breaker_redis_client()`, lazy-init, `REDIS_URL` déjà utilisé
comme broker Celery dans ce projet) au lieu du framework `cache` de
Django par défaut. Changement volontairement **surgical** : seul cet
état précis change de backend, pas tous les usages de `cache.*` ailleurs
dans `tasks.py` (portée plus large, plus risquée, pas nécessaire pour
corriger ce problème précis).

**Vérifié, pas juste supposé corrigé** :
- 5 tests de régression contre le **vrai** Redis (pas un mock — un mock
  en mémoire aurait masqué exactement le bug corrigé), dont un qui
  simule explicitement un redémarrage de worker (recrée le client Redis
  module-level) et confirme que la baseline puis le déclenchement
  survivent.
- Vérification en direct sur le NAS après déploiement, container
  `backend` fraîchement recréé (donc cache en mémoire vidé pour de
  vrai, pas simulé) : baseline établie à 10000$, déclenchement correct
  à -4% après un redémarrage simulé additionnel du client Redis lui-même.

**Note** : les clés de déclenchement d'AI_BLUECHIP/AI_PENNY du
2026-08-13 (avant le fix) ne sont pas récupérables rétroactivement
(écrites dans l'ancien LocMemCache, disparu avec le redémarrage) — sans
impact réel, la clé est datée du jour et ne se reporte pas sur demain de
toute façon.

### Diagnostic original (2026-08-11, conservé pour l'historique)

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

### Résultat du A-test réel (2026-08-11, 22h06-22h36 UTC, ~30 min)

```
{'status': 'ok', 'universe_scanned': 720, 'basic_data_ok': 690,
 'coverage_ratio': 0.958, 'survivors_this_week': 4,
 'total_active_now': 4, 'newly_deactivated_by_miss': 0,
 'newly_deactivated_by_cap': 0}
```

**4 survivants** sur 690 tickers évalués (690/720 = 95.8% de
couverture, run non-dégradé) :

| Ticker | Indice | Piotroski | Altman Z |
|---|---|---|---|
| FVI.TO (Fortuna Mining) | TSX | 9 | 5.82 |
| K.TO (Kinross Gold) | TSX | 9 | 8.21 |
| ADBE (Adobe) | SP500 | 7 | 7.48 |
| OGC.TO (OceanaGold) | TSX | 7 | 8.1 |

Résultat directionnellement sensé : très peu de survivants (cohérent
avec "tous les seuils obligatoires, ET pas OU" -- l'objectif était une
poignée de titres, pas un tri large), tous largement au-dessus des
seuils Altman Z (>3.0) et Piotroski (>=7). Concentration inattendue
mais explicable sur les mines d'or/métaux précieux TSX (FVI/K/OGC) --
plausible dans le contexte actuel (marges/bilans robustes typiques du
secteur en période de prix élevés), pas un signe de bug.

**Bug découvert pendant ce run, corrigé séparément avant le prochain
cycle** : `_fetch_tsx_symbols` produisait des tickers invalides pour
les titres à catégorie multiple/fiducie de revenu (point interne au
lieu du tiret attendu par Yahoo -- ex. `RCI.B.TO` au lieu de
`RCI-B.TO`), ~25 titres TSX skippés ce cycle (dont Rogers, Teck,
CCL Industries, CGI...). Corrigé (`.replace('.', '-')` avant le
suffixe `.TO`), vérifié en direct sur 5 tickers affectés, déployé
séparément (commit `49b42f65`) -- prendra effet au prochain cycle
hebdomadaire (dimanche prochain), pas relancé rétroactivement sur ce
run de test.

**Décision** : `universe_source="combined"` reste **NON activé**.
Un seul cycle réel ne suffit pas pour juger de la stabilité
semaine-après-semaine (le but de `consecutive_weeks_missed` est
justement d'observer plusieurs semaines avant de faire confiance à la
liste) -- à réévaluer après 2-3 cycles hebdomadaires réels (donc pas
avant début septembre 2026, le prochain tournant dimanche prochain
avec le fix TSX inclus).

## 12. [FAIT le 2026-08-11] Partie B — FUNDAMENTAL_LAB connecté à un vrai compte Alpaca paper dédié

**B1 — Diagnostic (aucun changement de code)** : code d'ordre existant
lu en détail (`_execute_alpaca_paper_trades_for_sandbox`,
`sync_alpaca_paper_trades`, `_journal_paper_trades`,
`portfolio/alpaca_data.py`). Confirmé : `_alpaca_trading_client()` et
toutes les fonctions qui en dépendent sont câblées en dur sur
`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` (compte partagé), sans paramètre,
impossible à réutiliser tel quel pour un second compte.

Risque réel confirmé (précisément celui suspecté par l'utilisateur, via
un mécanisme différent de celui deviné) : `sync_alpaca_paper_trades` et
`_journal_paper_trades` filtrent `PaperTrade.objects.filter(broker='ALPACA', ...)`
**sans jamais scoper par sandbox**. Si FUNDAMENTAL_LAB avait réutilisé
`broker='ALPACA'`, `sync_alpaca_paper_trades` aurait tenté de
synchroniser ses lignes via le mauvais client (compte partagé) ->
`get_order_by_id` toujours `None` sur le mauvais compte -> ces lignes ne
se seraient **jamais synchronisées** (statut/PnL figés pour toujours,
silencieusement). `monitor_hive_trade`/`monitor_active_trade` vérifiés
spécifiquement (nommés dans l'inquiétude initiale) : **pas concernés du
tout**, aucun des deux ne touche `PaperTrade`/`broker`.

Grep complet du repo (pas juste les 2 call sites de B1) fait avant B2 :
5 autres occurrences dans `portfolio/views.py`, **toutes déjà protégées**
(whitelist de sandboxes n'incluant jamais FUNDAMENTAL_LAB, ou
`.exclude(sandbox='FUNDAMENTAL_LAB')` explicite déjà présent avant ce
soir). Confirmé : seuls les 2 call sites de B1 sont un vrai risque.

**Découverte additionnelle en cours de B1/B2** : aucun mécanisme de
sortie n'existait pour FUNDAMENTAL_LAB, ni automatique ni manuel
(`LabPositionMarkManualView` ne fait que poser un flag "j'y crois",
jamais fermer). Acceptable en pure simulation DB, plus une fois du vrai
capital paper engagé -- décision utilisateur : ajouter un stop-loss
minimal (voir `monitor_lab_positions` ci-dessous) avant tout ordre réel.

**B2 — Implémentation** :
- `PaperTrade.BROKER_CHOICES` : nouvelle valeur `'ALPACA_LAB'`, distincte
  de `'ALPACA'` par construction -- élimine le risque à la source plutôt
  que de dépendre d'un filtre sandbox ajouté (et à retenir) dans 2
  fonctions partagées. Migration `0036_alter_papertrade_broker`
  (choix Python seulement, SQL confirmé no-op via `sqlmigrate`), backup
  pg_dump avant application, comme toujours sur ce projet.
- `portfolio/alpaca_data_lab.py` (nouveau fichier, aucune modification de
  `alpaca_data.py`) : client Alpaca séparé lisant
  `ALPACA_API_KEY_FUNDAMENTAL_LAB`/`ALPACA_SECRET_KEY_FUNDAMENTAL_LAB`.
  Fonctions nommées explicitement (`_alpaca_lab_trading_client`,
  `submit_lab_market_order`, etc.) pour qu'un futur copier-coller ne
  réintroduise jamais la confusion entre les deux comptes. Vérifié en
  direct sur le NAS : `account_number=PA34ZP3JNN2T`,
  `equity=buying_power=$50 000`.
- `_create_lab_position` (`analysis/tasks.py`) : tickers TSX (`.TO`)
  restent en simulation pure (`broker='SIM'`) -- Alpaca est un courtier
  américain, ne peut pas trader ces titres, pas d'appel API inutile sur
  un échec prévisible. Tickers US : ordre marché réel via
  `submit_lab_market_order`. Échec de l'ordre -> aucun PaperTrade créé
  (même garde-fou que pour toute autre erreur, `lab_position` conservée
  sans ordre attaché, jamais de position fabriquée pour un ordre qui
  n'existe pas).
- `monitor_lab_positions` (nouvelle tâche, Beat `*/10min` heures de
  marché ET, queue `analysis_queue`) : stop-loss minimal, uniquement sur
  `broker='ALPACA_LAB'` (jamais sur les positions SIM/TSX/échec-d'ordre).
  Vend au marché via le même client dédié si le prix casse
  `PaperTrade.stop_loss` (déjà calculé à l'entrée, jamais utilisé avant
  ce soir). Volontairement minimal : pas de take-profit, pas de
  trailing stop.

Déployé sur le NAS (backend + celery_worker_analysis + celery_beat +
celery_worker reconstruits et recréés, migration appliquée, imports +
Beat schedule + connexion compte vérifiés en direct avant recréation
des containers en prod).

**Pas encore fait (B3, prochaine étape)** : aucun ordre réel n'a encore
été placé -- besoin d'un dry-run contrôlé, confirmé directement sur le
dashboard Alpaca (pas seulement en DB), avant de considérer B2 comme
validé en conditions réelles.

### B3 — Dry-run manuel isolé (2026-08-11 23h53 UTC)

Position unique ADBE (~$264, 1 action) créée manuellement via
`_create_lab_position` sur un vrai `ScanResult` construit pour le test
(verdict marqué explicitement comme test manuel dans `quant_details`,
pas un vrai verdict LLM). Ordre confirmé **directement via l'API Alpaca**
(pas juste en DB), `get_lab_order_by_id` interrogé indépendamment :
`symbol=ADBE qty=1 side=BUY status=ACCEPTED`, `buying_power` du compte
réduit de $50 000 à $49 736.74 -- capital réellement engagé sur le
compte dédié `PA34ZP3JNN2T`.

**Bug trouvé avant qu'il ne compte pour de vrai** : l'ordre a été soumis
hors heures de marché (19h53 ET), donc `filled_qty=0` -- pas encore
rempli. `monitor_lab_positions` (version initiale) comparait le prix
marché au stop_loss pour toute position `status='OPEN'` en DB, **sans
jamais vérifier si l'ordre avait réellement été rempli côté Alpaca**. Le
test du soir est resté propre uniquement parce que le prix ADBE était
loin du stop -- pas parce que le code se protégeait correctement. Un
stop-loss cassé sur un ordre encore `ACCEPTED` aurait déclenché une
tentative de vente réelle sur une position qui n'existe pas encore
côté broker (comportement Alpaca jamais testé dans ce cas).

**Corrigé avant l'ouverture du 2026-08-12** (bloquant, pas juste noté --
demande explicite) : `monitor_lab_positions` interroge maintenant
`get_lab_order_by_id` et vérifie `status == 'filled'` AVANT toute
comparaison prix/stop-loss. Sert aussi de synchro légère : au premier
remplissage détecté, `entry_price`/`broker_avg_price`/`broker_status`
sont recalés sur les vraies données de fill (pas une tâche de synchro
complète, juste ce qu'il faut pour que ce garde-fou soit fiable).
Validé en 2 temps avant déploiement : (1) appel direct de
`get_lab_order_by_id` sur l'ordre ADBE en attente, confirmé
`status='accepted'` → non éligible ; (2) exécution complète de
`monitor_lab_positions()` en conteneur jetable, résultat
`{'checked': 1, 'closed': 0, 'skipped_unfilled': 1}`, log exact
conforme à l'attendu. Déployé (backend + les 3 workers/beat reconstruits
et recréés).

**Reste à vérifier demain matin (2026-08-12), ajouté à la checklist** :
confirmer que l'ordre ADBE se remplit réellement à l'ouverture
(`filled_qty` 0→1), que `monitor_lab_positions` détecte ce remplissage
et recale `entry_price` sur le prix réel, en plus de la vérification
habituelle des scans 9h45/16h30 (et surveiller si le scan de 9h45 ouvre
une nouvelle position FUNDAMENTAL_LAB pendant que l'ordre ADBE d'hier
est encore en transition -- scénario justement cité comme risque
potentiel avant le fix).

## 13. [FAIT le 2026-08-12] Partie C — Diagnostic signal WATCHLIST bloqué à zéro

**Contexte** : `base_model_signal` variait normalement (0.24-0.58 sur les
logs), mais le `final` (celui qui gate vraiment l'achat) était observé à
exactement 0 sur toute une semaine de `SystemLog`. Demande explicite de
diagnostiquer avant de corriger, et de ne pas supposer qu'un fix règle
tout juste parce que le signal n'est plus bloqué à 0 (risque de
calibration masquée derrière le premier bug).

**Ce n'est PAS un bug de code (aucun crash, NaN, ou échec silencieux).**
Deux causes réelles, confirmées avec des données de production
(`SystemLog.metadata['signal_distribution']`, pas une supposition) :

**1. `payload['signal']` pour WATCHLIST n'a JAMAIS été dérivé de
`base_model_signal`** — c'est une confusion de prémisse, pas un bug :
`_signal()` (`portfolio/tasks.py:5460`) remplace entièrement le signal
du modèle ML par `_mean_reversion_score()` (RSI14 + bonus Bollinger)
pour ce sandbox spécifiquement — déjà documenté explicitement en
commentaire dans le code avant ce soir. `base_model_signal` est
seulement stocké à titre diagnostique, jamais utilisé pour la décision
d'achat de WATCHLIST.

**2. Qualité des données : 4 tickers sur 10 radiés/invalides.**
`FLT.V`, `HIVE.V`, `BITF.TO`, `LUG` renvoyaient tous un 404 Yahoo
Finance — pas des radiations réelles mais des **changements de suffixe
d'échange non suivis** (même classe de bug que le fix TSX `RCI.B.TO` de
la nuit dernière) :
- `FLT.V` → `FLT.TO` (Volatus Aerospace, gradué TSXV→TSX)
- `HIVE.V` → `HIVE.TO` (HIVE Digital Technologies, gradué TSXV→TSX)
- `BITF.TO` → `KEEL.TO` (Bitfarms rebrandé "Keel Infrastructure",
  avril 2026, changement de ticker)
- `LUG` → `LUG.TO` (Lundin Gold, suffixe manquant)
Tous vérifiés en direct (prix réels obtenus) avant correction.
**Corrigé** : `SandboxWatchlist(sandbox='WATCHLIST').symbols` mis à
jour en DB sur le NAS (changement de donnée, pas de code). Vérifié
après coup : 10/10 tickers produisent maintenant un signal valide
(avant : 6/10).

**3. Calibration confirmée problématique (la 2e couche que l'utilisateur
anticipait)** : `buy_threshold=0.55` pour WATCHLIST (confirmé via
`SystemLog.metadata['buy_threshold']`, valeur réelle en production).
`rsi_score = max(0.0, (30-rsi)/30)` plafonne à 0 dès que RSI >= 30, et
le bonus Bollinger n'ajoute que +0.15 max — pour atteindre 0.55 il faut
RSI ≲ 18, une survente extrême, pas juste "pas cher". Confirmé sur les
données réelles :
- 7-10 août : `final`=0.0/0.0/0.0 sur des dizaines d'entrées
  consécutives (aucun des 6 tickers valides sous RSI 30 pendant 3 jours).
- 10 août 3h38 → aujourd'hui : `final` devient non-nul (jusqu'à
  0.34-0.49) mais **ne franchit jamais 0.55** — le seuil reste
  structurellement hors de portée même quand le marché bouge.
- Vérifié en direct aujourd'hui : RSI des 6 tickers alors valides =
  44-92, aucun proche de la survente.

**Décision explicite : le nettoyage des tickers est fait, la
calibration (seuil/formule) N'EST PAS touchée ce soir** — c'est une
décision de stratégie de trading, pas un bug à corriger unilatéralement.
À rediscuter séparément si voulu.

**Reste ouvert / à surveiller** :
- `HIVE_MONITOR_SYMBOL=HIVE.V` dans `deploy/.env` (utilisé par la tâche
  séparée `monitor_hive_trade`, alerte de prix sur un seul symbole —
  n'a AUCUN rapport avec le pipeline PaperTrade/WATCHLIST, vérifié en
  B1) a le même symbole périmé. Pas corrigé ce soir (hors scope de
  Partie C), à faire séparément si voulu (`HIVE.V` → `HIVE.TO`).
- Avec le nettoyage des tickers, `final` devrait maintenant produire
  des échantillons sur 10 tickers au lieu de 6 aux prochains scans —
  à confirmer sur les prochains cycles réels (pas supposé).

## 14. [LIMITATION CONNUE, découvert le 2026-08-13] La couverture news du pipeline (Alpaca) rate des vents contraires réels, pas juste des catastrophes

**Statut : documenté, pas corrigé — comportement à connaître pour interpréter
tous les verdicts `confirmed` du pipeline (screener auto ET analyse manuelle),
pas un bug isolé à réparer.**

**Origine** : test manuel sur DAR (bouton "Analyse manuelle complète") —
Claude a rendu `confirmed` en citant surtout le côté haussier (EPS/ventes
au-dessus des attentes, relèvements d'objectifs de cours). Un article
TipRanks trouvé séparément par l'utilisateur mentionnait des vents
contraires réels et documentés : "Near-Term Margin Compression Concerns"
(marge Fuel Specialties en baisse de 1,5 point, pression attendue au
Q3) et "Volume Headwinds" (contraction de volume de 2% dans Performance
Chemicals, contraintes d'approvisionnement) — rien de dramatique (pas de
fraude, pas de perte de contrat), donc probablement pas suffisant pour
faire basculer le verdict, mais absent du raisonnement de Claude.

**Vérifié en direct (`analysis/services/news.py::fetch_recent_news`,
pas une supposition)** — deux causes concrètes, cumulatives :

1. **Couverture de source limitée.** Les 9 articles réellement retournés
   par l'API news Alpaca pour DAR (`days_back=60, limit=10`) sont **tous
   de source `benzinga`** — aucun TipRanks. L'article en question n'était
   donc simplement jamais dans le pool de candidats envoyé à DeepSeek/
   Claude, indépendamment de la fenêtre de 60 jours ou du jugement du LLM.
2. **Résumés fréquemment vides.** 6 des 9 articles reçus avaient
   `summary` = chaîne vide — seul le titre était exploitable (ex.
   "Darling Ingredients Q2 EPS $2.41 Beats $0.97 Estimate" sans aucun
   détail). Même si l'article TipRanks avait été de source Benzinga, un
   résumé vide n'aurait transmis aucun détail de marge/volume de toute
   façon.

**Interprétation à retenir pour l'avenir** (résumé de la discussion) :
un verdict `confirmed` de ce pipeline signifie **"aucune raison
dramatique trouvée dans les sources couvertes"**, pas "aucun vent
contraire n'existe". Le système semble fiable pour écarter les vraies
alertes rouges (fraude, perte de contrat, guidance coupée) mais moins
outillé pour peser des nuances plus subtiles ("pression persistante
attendue", pas une urgence mais un facteur réel) qui n'apparaissent que
dans des sources hors Benzinga ou dans le corps d'un article (pas le titre).

**Pistes non explorées, à évaluer séparément si voulu** (pas fait ici,
hors scope de cette découverte) :
- Élargir `fetch_recent_news` à d'autres sources que Benzinga (si l'API
  Alpaca le permet via un paramètre, pas vérifié).
- `limit=10` : vérifier si un `limit` plus élevé changerait la
  composition (pas testé — les 9 articles reçus couvraient déjà toute la
  fenêtre de 60 jours sans être plafonnés par la limite, donc probablement
  pas le facteur limitant ici, mais à confirmer sur un ticker plus suivi).
- Gérer les résumés vides différemment (ex. skip un article sans résumé
  plutôt que de l'envoyer comme un titre nu, ou fetch le corps complet
  ailleurs) — change le comportement du prompt pour tous les tickers,
  à peser avant de toucher.

---

## 15. [CORRIGÉ le 2026-08-14, leçon opérationnelle] nginx met en cache l'IP de `backend` au démarrage — toute recréation de `backend` sans recréer `frontend` casse TOUTES les routes `/api/`

**Statut : fermé pour cette occurrence (redémarrage de `frontend`), mais
la cause structurelle reste vraie à chaque déploiement futur — à retenir,
pas un bug de code.**

**Symptôme signalé** : "Screener fondamental... Request failed with
status code 502... Aucun résultat pour ce preset." Vérifié : **pas
spécifique au screener** — `/api/analysis/ticker/AAPL/`,
`/api/paper-trades/summary/`, `/api/logs/*` retournaient tous 502 au
même moment.

**Cause confirmée** (`docker logs personalstock-frontend-1`) :
```
connect() failed (111: Connection refused) while connecting to upstream,
upstream: "http://172.16.5.9:8000/..."
```
`nginx.conf` utilise `proxy_pass http://backend:8000;` avec un nom
d'hôte statique — nginx résout ce nom **une seule fois, au démarrage du
process nginx**, et met le résultat en cache pour toute la durée de vie
du container (comportement nginx standard, pas une erreur de config en
soi). Cette session a recréé `backend` plusieurs fois le 2026-08-13 (fix
`tsx-guardian`, entre autres) sans recréer `frontend` à chaque fois —
`backend` a reçu une nouvelle IP Docker interne à chaque recréation,
mais `frontend`/nginx continuait d'essayer l'ancienne IP, devenue
invalide. Confirmé : appel direct à `backend` sur le port hôte mappé
(8001) fonctionnait très bien pendant tout ce temps — seul le chemin
interne nginx → backend était cassé, donc invisible sans vérifier
spécifiquement les logs nginx.

**Fix immédiat** : `docker compose restart frontend` — force nginx à
redémarrer son process et donc à ré-résoudre `backend` vers l'IP
actuelle. Confirmé : les 3 endpoints testés (`ticker/AAPL`,
`paper-trades/summary`, `screener/.../run/`) sont repassés à 200/202
immédiatement après.

**À retenir pour tous les déploiements futurs** : recréer `backend`
(ou n'importe quel service que nginx proxy) **sans** recréer/redémarrer
`frontend` juste après laisse une fenêtre où toutes les routes `/api/`
sont cassées, silencieusement (aucune erreur ne remonte tant que
personne n'utilise l'app ou ne vérifie les logs nginx précisément).
Deux options pour l'avenir :
1. **Discipline manuelle** : toujours inclure `frontend` dans la liste
   des services `--force-recreate`/`restart` quand `backend` est
   recréé, même si le code frontend lui-même n'a pas changé.
2. **Fix structurel possible** (pas fait ce soir, à évaluer séparément) :
   nginx supporte un résolveur dynamique (`resolver 127.0.0.11
   valid=10s;` + variable dans `proxy_pass` au lieu d'un nom d'hôte en
   dur) qui re-résout périodiquement sans jamais nécessiter de
   redémarrage manuel de `frontend`. Éliminerait la classe de bug
   entière plutôt que de compter sur la discipline à chaque fois.

## 16. [CORRIGÉ le 2026-08-14] Circuit breaker partageait une clé de cache entre le chemin SIM (~10-25k$) et le chemin Alpaca réel (~100k$) du même sandbox

**Statut : fermé, déployé et vérifié en production.**

**Découvert pendant** l'investigation "est-ce que baisser les seuils
d'entrée (phase d'exploration) a aidé à générer des trades ?" —
AI_BLUECHIP et AI_PENNY déclenchaient le circuit breaker quasi tous les
jours, ~20-30 min après l'ouverture du marché, quel que soit le
contexte réel de pertes.

**Cause confirmée** : `_daily_equity_circuit_breaker(sandbox, equity_now)`
est appelé depuis deux chemins différents pour le même sandbox :
- `_execute_paper_trades_for_sandbox` (chemin SIM) — passe `capital`, une
  comptabilité interne fictive de l'ordre de 10-25k$.
- `_execute_alpaca_paper_trades_for_sandbox` (chemin Alpaca) — passe
  l'équité réelle du compte Alpaca paper **partagé**, de l'ordre de
  100k$.

Les deux écrivaient dans **la même clé de cache** Redis
(`daily_equity_base:{sandbox}:{date}`). Le premier chemin à s'exécuter
chaque jour fixait la baseline à sa propre échelle ; l'autre chemin,
comparé à cette baseline étrangère, se voyait à chaque appel afficher un
"drawdown" de l'ordre de -76 % à -90 %, largement au-delà du seuil de
3 %, déclenchant le breaker presque tous les jours sans rapport avec une
vraie perte.

**Preuve concrète recueillie le 2026-08-14** :
- Baseline Redis du jour pour AI_BLUECHIP : `100019.01` — quasiment
  identique à l'équité réelle du compte Alpaca partagé interrogée
  indépendamment via `get_account()` : `100005.51`.
- `capital` loggé dans `SystemLog` au moment exact des déclenchements
  AI_BLUECHIP/AI_PENNY : `23803.36` / `10038.11` — correspond à la
  comptabilité SIM, pas à Alpaca.
- Confirme directement la contamination croisée : la baseline vient
  d'Alpaca, la vérification vient de SIM.

Explique très probablement l'essentiel du silence documenté dans
`docs/ML_PIPELINE_AUDIT.md` (0 trade WATCHLIST/AI_BLUECHIP/AI_PENNY sur
90 jours) — un breaker qui se déclenche ~20-30 min après l'ouverture
empêche presque tout trade, indépendamment des seuils d'entrée.

**Fix** : ajout d'un paramètre `broker_path` (`'SIM'` ou `'ALPACA'`,
défaut `'SIM'`) qui namespace les clés de cache :
`daily_equity_base:{sandbox}:{broker_path}:{date}` /
`daily_equity_trip:{sandbox}:{broker_path}:{date}`. Les deux chemins ont
maintenant chacun leur propre baseline et leur propre état de
déclenchement — plus jamais de comparaison entre deux échelles
incompatibles. `reset_daily_equity_breaker` mis à jour pour boucler sur
les deux `broker_path` par sandbox.

**Alternative envisagée et écartée** : unifier sur l'équité Alpaca
réelle partout (y compris pour SIM). Écartée parce que SIM et Alpaca
ferment des ensembles de positions entièrement différents au
déclenchement — les fusionner sous un seuil de protection unique aurait
un sens économique douteux (une perte sur le book Alpaca ne devrait pas
figer les trades SIM, et inversement). Décision validée avec
l'utilisateur avant implémentation.

**Tests** : 3 nouveaux tests dans
`tests_circuit_breaker_persistence.py`
(`DailyEquityCircuitBreakerBrokerPathIsolationTests`) vérifiant
l'indépendance des baselines, qu'un déclenchement SIM n'affecte pas
Alpaca, et que l'appel sans `broker_path` explicite reste sur SIM par
défaut (compatibilité). Suite complète (`tests_circuit_breaker_persistence`
+ `tests_paper_trade_decision_stats`, dont le mock a dû être ajusté pour
accepter le nouveau kwarg) : 11/11 OK avant déploiement.

**Déployé** : `backend`, `celery_worker`, `celery_worker_analysis`,
`celery_beat` recréés ensemble le 2026-08-14 (avec `frontend` en même
temps, par discipline suite à l'item 15) — logs de démarrage propres,
aucune erreur, proxy nginx→backend vérifié fonctionnel après coup.

## 17. [CORRIGÉ le 2026-08-15] `confidence_factor`/`min_confidence` utilisaient un plancher codé en dur, indépendant de `buy_threshold`

**Statut : fermé, déployé avec le passage WATCHLIST sur FUSION (item lié,
voir docs/WATCHLIST_SIGNAL_DIAGNOSTIC_2026-08.md).**

**Découvert le 2026-08-14** en direct sur AI_BLUECHIP : un signal `0.6089`
(> `buy_threshold` 0.55, phase d'exploration active) était quand même
bloqué, avec la raison `confidence_too_low` — alors que le seuil affiché
disait que ce candidat aurait dû qualifier.

**Cause confirmée** : `confidence_factor` (chemin SIM,
`_execute_paper_trades_for_sandbox`) et `min_confidence` (chemin Alpaca,
`_execute_alpaca_paper_trades_for_sandbox`) utilisaient un plancher codé en
dur (`0.65` / `ALPACA_MIN_CONFIDENCE` défaut `0.7`) **complètement
indépendant** de `buy_threshold`, quelques lignes plus haut dans la même
fonction. Sans conséquence tant que `buy_threshold` restait au-dessus de ce
plancher (AI_BLUECHIP normal : 0.80, `min(0.65, 0.80) = 0.65`, comportement
inchangé) — mais dès que `buy_threshold` descendait sous 0.65/0.7 (la phase
d'exploration à 0.55, ou désormais le nouveau seuil WATCHLIST à 0.50, voir
item lié), une zone morte apparaissait : un candidat pouvait passer
`buy_threshold` et se faire quand même rejeter ici, sans qu'aucun trade ne
soit jamais créé malgré un seuil affiché qui disait pourtant "oui".

**Pertinence directe pour WATCHLIST (item lié, 2026-08-15)** : sans ce fix,
le passage de WATCHLIST sur `base_signal` (Option A) n'aurait produit
**aucun trade supplémentaire en pratique** — la distribution réelle de
`base_model_signal` pour WATCHLIST (médiane ~0.40, max ~0.54-0.58, voir
diagnostic) tombe presque entièrement dans l'ancienne zone morte
0.50-0.65.

**Fix** : le plancher est maintenant plafonné par `buy_threshold` --
`confidence_floor = min(0.65, buy_threshold)` (SIM) et `min_confidence =
min(_env_float(prefix, 'ALPACA_MIN_CONFIDENCE', '0.7'), buy_threshold)`
(Alpaca) -- les deux gates restent cohérents entre eux dans tous les modes
(normal, phase d'exploration, ou tout futur seuil abaissé), sans jamais
relever le plancher au-dessus de sa valeur d'origine.

**Tests** : 2 nouveaux tests dans `tests_paper_trade_decision_stats.py`
(`ConfidenceFloorCappedAtBuyThresholdTests`) — un signal dans l'ancienne
zone morte (0.55, entre le nouveau seuil WATCHLIST 0.50 et l'ancien
plancher 0.65) crée bien un trade maintenant ; un signal sous
`buy_threshold` reste bloqué au bon endroit (`blocked_threshold`, pas
`blocked_confidence`). Non-régression AI_BLUECHIP/AI_PENNY (seuils
normaux, au-dessus de 0.65) confirmée par les tests existants
(`PaperTradeDecisionStatsInvariantTests`), inchangés.

**Non testé automatiquement** : le chemin Alpaca (`_execute_alpaca_paper_
trades_for_sandbox`) n'a pas de suite de tests dédiée dans ce repo (aucune
avant ce fix) — construire un harnais complet (compte Alpaca, positions,
ordres, tiers) aurait été disproportionné pour cette correction ponctuelle.
Vérifié manuellement en rejouant le calcul en direct sur la production à la
place (voir item lié).

---

## 18. [CORRIGÉ le 2026-08-16] `LabPerformanceView` comptait et moyennait des lignes FUNDAMENTAL_LAB mortes/dupliquées

**Statut : fermé, déployé et vérifié.**

**Découvert** en révisant les stats affichées par le panneau "Analytics
Lab" (patterns automatiques rendement réel vs facteur — secteur,
confiance Claude, ROE, FCF Yield). `LabPerformanceView.get()` faisait
`FundamentalLabPosition.objects.select_related("scan_result").all()` —
incluait donc des positions fermées par un nettoyage de doublons (même
ticker, même scan, entrées orphelines créées avant qu'une garde anti-
doublon n'existe, cf. item 8) sans jamais avoir été de vraies positions
vivantes. Ces lignes gonflaient artificiellement le nombre d'échantillons
et diluaient les moyennes de rendement par facteur.

**Fix** : `.exclude(is_open=False, exit_date__isnull=True)` — exclut
précisément les positions fermées sans date de sortie (signature des
lignes de nettoyage, distinctes d'une position normalement fermée qui a
toujours un `exit_date`).

**Fix lié, même vue** : `LabPositionSerializer.get_current_return_pct`
retournait un rendement recalculé au prix live même pour les positions
déjà **fermées** — incohérent avec le P&L réellement réalisé à la
sortie. Corrigé pour retourner `round(obj.realized_return_pct, 2)`
quand la position est fermée et qu'un rendement réalisé existe,
et ne retomber sur le calcul au prix live que pour les positions encore
ouvertes.

**Tests** : `LabPerformanceViewTests` (3 tests, `analysis/tests.py`) —
confirme que les lignes de nettoyage sont exclues des stats, que le
rendement réalisé prime sur le calcul live pour une position fermée, et
non-régression sur une position ouverte normale.

---

## 19. [FAIT le 2026-08-16] Code couleur gain/vert perte/rouge + badge marché fermé sur les panneaux de trading

**Statut : fermé, déployé et vérifié.**

**Demande utilisateur** (revue UI complète, transcription de l'app
collée en direct) : rendre les gains/pertes visuellement immédiats
("ajoute les couleurs si gain vert si perte rouge etc, rend ça clair et
facile à comprendre rapidement"), et clarifier pourquoi l'Intraday
affiche parfois un tableau figé (marché fermé, pas un bug de
chargement).

**Fix, plusieurs panneaux** (`LivePaperTrading.js`, `AnalyticsLabPage.js`,
`IntradayAI.js`, `MarketScannerPanel.js`) : couleur conditionnelle
(vert/rouge, plus neutre à zéro) sur le P&L des trades, le détail d'un
trade, la liste Watchlist IA, et les stats Intraday — au lieu d'une
couleur fixe indépendante du signe.

**Bug trouvé en chemin dans `MarketScannerPanel.js`** : `+{item.
change_pct}%` préfixait un "+" en dur et gardait `text-emerald-300`
(vert) même quand `change_pct` était négatif — un titre en baisse de
-4.39 % s'affichait "+-4.39%" en vert. Corrigé avec un signe et une
couleur conditionnels au signe réel de la valeur.

**Fix backend associé** : ajout d'un flag `market_closed` (calculé via
`_market_closed_now()`, déjà utilisé ailleurs dans `tasks.py`) à
`MarketScannerView` et `AlpacaIntradayView`. Le frontend affiche
maintenant un badge explicite "marché fermé" plutôt que de laisser
deviner si un tableau figé est un bug de chargement ou l'état normal
hors-séance.

**Effet de bord découvert et corrigé au passage** : 3 tests
(`ConfidenceFloorSimPathTests`, `WatchlistBaseModelSignalRegressionTests`,
`ConfidenceFloorCappedAtBuyThresholdTests`) sont devenus intermittents
en travaillant dans cette zone du code — `_time_of_day_penalty()`
n'était pas mocké et lisait l'heure réelle, appliquant une vraie
pénalité de -10 % "heure du lunch" (11h30-13h30 ET) selon l'heure à
laquelle la suite tournait. Corrigé en ajoutant
`'_time_of_day_penalty': lambda *a, **k: 1.0` aux dictionnaires de mock
des classes concernées.

---

## 20. [FAIT le 2026-08-16] Risk Control Center — remplacement des curseurs `localStorage` factices par un snapshot réel en lecture seule

**Statut : fermé, déployé et vérifié.**

**Découvert** en répondant à la question directe de l'utilisateur : "le
risk control fonctionne-t-il, si je le change ça va tout changer ?". Le
composant `RiskControlCenter.js` affichait des curseurs (seuils
d'achat/vente, drawdown max, taille de position) entièrement
déconnectés du backend — ils lisaient et écrivaient uniquement dans
`localStorage` du navigateur. Les déplacer n'avait strictement aucun
effet sur le trading réel ; l'app donnait l'illusion d'un panneau de
contrôle fonctionnel.

**Fix** : nouvelle vue `RiskSnapshotView` (`portfolio/views.py`,
exposée sur `/api/risk-snapshot/`) et fonction miroir dédiée
`_effective_sandbox_thresholds(sandbox, prefix)` dans `tasks.py` —
recalcule les mêmes seuils que le chemin de trading réel (buy/sell
threshold, effet de la phase d'exploration, drawdown du circuit
breaker, position min/max) mais **en lecture seule**, sans jamais
toucher au chemin de trading live. `RiskControlCenter.js` réécrit pour
afficher ce snapshot réel au lieu de curseurs éditables.

**Décision explicite** : rendre le panneau **read-only** plutôt que de
lui donner un vrai pouvoir d'écriture — modifier les seuils de trading
depuis l'UI aurait été un changement de portée bien plus large (config
en base vs variables d'environnement, validation, audit trail) hors
cadre d'une session "vérifie et corrige les anomalies".

**Tests** : `tests_risk_snapshot.py` (nouveau fichier,
`RiskSnapshotViewTests`, 3 tests) — seuils en mode normal, override en
phase d'exploration, présence des paramètres globaux
(`daily_drawdown_circuit_breaker_pct`, position min/max).

---

## 21. [FAIT le 2026-08-16] Suite de l'item 11 — univers combiné S&P500+TSX activé sur le scan quotidien, preset Value + Catalyst ajouté

**Statut : fermé, déployé et vérifié.**

**Contexte** : l'item 11 avait construit `universe_source="combined"`
mais l'avait laissé **désactivé** — décision explicite à l'époque, en
attendant de voir le screener tourner un moment sur l'univers restreint
aux sandboxes. Reposé directement par l'utilisateur ("regarde
undervalue without reason, dis-moi si on devrait modifier quelque
chose").

**Fix** : `_run_scan_for_preset(preset)` (`analysis/tasks.py`) passe
maintenant `universe_source="combined"` au lieu de `"sandboxes"` —
le scan quotidien du preset `undervalued-without-reason` couvre
désormais tout l'univers S&P500+TSX au lieu des seuls tickers déjà
présents dans les sandboxes de trading actif.

**Nouveau preset `value-catalyst`** (`ScreenerPreset` id=2,
`thresholds={'require_catalyst': True}`) : ajoute une exigence de
catalyseur (achat d'initié détecté, ou inflexion de marge déjà
existante) en plus du filtre value de base, pour surfacer des tickers
sous-évalués avec un signal d'intérêt futur plus concret qu'une simple
décote statique.

**Choix d'implémentation (discuté avec l'utilisateur)** : le preset
`value-catalyst` **réutilise le même passage du quant filter**
qu'`undervalued-without-reason` plutôt que de déclencher une requête
Claude séparée — `evaluate_ticker(ticker, preset_thresholds)` calcule
`insiders_buying` via un nouvel appel `_fetch_insider_buying(ticker)`
(FMP, même clé de cache `insider_summary:{ticker}` que
`portfolio/views.py::_insider_summary`, donc pas de coût API
supplémentaire si déjà interrogé) uniquement quand
`thresholds.get("require_catalyst")` est vrai, et n'ajoute la
vérification Claude qu'après ce filtre local — pas de requête Claude
dupliquée pour la même analyse de base.

**Bug corrigé au passage** : `_fetch_margin_trend(ticker)` comparait
`rev` et `gp` avec `if rev and gp and rev != 0:` — `bool(float('nan'))`
vaut `True` en Python, donc un Gross Profit `NaN` n'était pas filtré et
se propageait jusqu'à un score `NaN`, faisant planter l'écriture en
base (voir item 22, trouvé le lendemain en observant ce fix tourner en
conditions réelles). Corrigé en réutilisant `_safe()`, déjà défini dans
le même module.

**Tests** : `RunScanUsesCombinedUniverseTests` (1 test,
`analysis/tests.py`), `QuantFilterValueCatalystTests` (4 tests —
insider buying détecté/absent, bonus de score, gate `require_catalyst`
absent par défaut), `QuantFilterMarginTrendTests` étendu avec
`test_margin_trend_skips_nan_gross_profit_instead_of_producing_nan`.

---

## 22. [CORRIGÉ le 2026-08-17] `NaN` Gross Profit faisait planter les scans réels (`OGC.TO`)

**Statut : fermé, déployé et vérifié en production.**

**Découvert** le lendemain de l'activation de l'univers combiné (item
21), en observant le premier scan réel tourner sur le nouvel univers
élargi : `django.db.utils.DataError: invalid input syntax for type
json` dans les logs (`docker logs
personalstock-celery_worker_analysis-1`). Tracé par `grep -B60` jusqu'à
`OGC.TO` (OceanaGold, déjà repéré comme cas limite dans le tableau de
l'item 11) — son Gross Profit le plus récent est `NaN` chez le
fournisseur de données.

**Cause** : bug déjà identifié et corrigé une fois dans ce même module
pour `require_catalyst` (item 21) mais qui existait **aussi** dans
`_fetch_margin_trend` avant ce correctif — `rev = revenue.get(col); gp
= gross_profit.get(col)` suivi de `if rev and gp and rev != 0:` ne
filtre pas `NaN` (`bool(float('nan'))` est `True` en Python). Le score
`NaN` qui en résultait remontait jusqu'à l'écriture du résultat de scan
en base, où le champ JSON refusait la valeur — le scan entier plantait
sur ce ticker au lieu de simplement l'ignorer.

**Fix** : `rev = _safe(revenue.get(col)); gp = _safe(gross_profit.get(col))`
— réutilise le helper `_safe()` déjà présent dans le module (test
d'auto-inégalité `value != value`, la manière standard de détecter
`NaN` sans dépendre de `math.isnan` sur une valeur potentiellement
`None`).

**Tests** : `test_margin_trend_skips_nan_gross_profit_instead_of_producing_nan`
(`QuantFilterMarginTrendTests`, `analysis/tests.py`) — confirme qu'un
Gross Profit `NaN` sur une période ne fait plus planter le calcul et
que la période est simplement ignorée plutôt que de produire un score
invalide.

---

## 23. [CORRIGÉ le 2026-08-17] WATCHLIST écrasée par le scanner momentum after-hours après la fermeture du marché

**Statut : fermé, déployé et vérifié (test de régression écrit avant le fix, échoue sur l'ancien code).**

**Découvert** en auditant systématiquement les tâches Beat utilisant le
pattern `hour='9-16'` (déjà connu pour ne pas s'arrêter exactement à la
fermeture de 16h00 — `9-16` couvre toute l'heure 16, donc jusqu'à
16h59 ET, pas juste jusqu'à la clôture). `market_scanner_task` bascule
vers `_afterhours_market_scan` via `_market_closed_now()` une fois le
marché fermé, et **cette branche tournait donc encore activement**
après la clôture, plusieurs fois de suite. Sur les 3 usages totaux de
`_market_closed_now()` dans tout `portfolio/tasks.py`, celui-ci était
le seul avec un effet de bord destructeur ; les 2 autres ont été
confirmés sûrs (un est déjà corrigé, l'autre est un planning fixe à 2h
du matin sans chevauchement possible).

**Cause** : `_afterhours_market_scan(symbols=None)` faisait un
`SandboxWatchlist.objects.update_or_create(sandbox='WATCHLIST', ...)`
**inconditionnel** quand `AI_SCANNER_UPDATE_WATCHLIST_MAIN=true` —
remplaçait entièrement la liste de tickers évaluée le matin même par
Option A (`base_model_signal`, voir item 13) par le résultat du scan
after-hours, parfois un seul symbole si le scan momentum ne trouvait
presque rien après la clôture.

**Premier essai, abandonné après un test qui l'a pris en défaut** :
réutiliser `_merge_bluechip_watchlist` en généralisant son paramètre
`sandbox` (déjà construit pour AI_BLUECHIP, item lié). Mauvais choix
d'architecture — cette fonction ne préserve que les symboles portant un
`protect_until` explicite dans `symbol_sources` ; les tickers de base
de WATCHLIST n'avaient jamais été écrits avec une telle protection
(mécanisme propre au dip-scanner AI_BLUECHIP), donc ils se faisaient
évincer exactement comme n'importe quel candidat non protégé — même bug
sous une autre forme, capturé par le test de régression écrit avant le
fix plutôt que découvert en production. Généralisation entièrement
annulée (retour à `sandbox='AI_BLUECHIP'` en dur, avec une note dans la
docstring expliquant pourquoi ce mécanisme ne convient pas à
WATCHLIST).

**Fix retenu** : fusion additive dédiée, écrite directement dans
`_afterhours_market_scan` — conserve toujours la liste WATCHLIST
existante, ne comble que la place restante (jusqu'à `main_limit`) avec
les nouveaux tickers repérés after-hours qui n'y sont pas déjà. N'écrit
en base que si la liste fusionnée diffère réellement de l'existante.

**Tests** : `AfterhoursScanMergesIntoWatchlistTests`
(`tests_bluechip_watchlist_merge.py`) —
`test_thin_afterhours_result_does_not_wipe_existing_watchlist` : seed 10
tickers `ML{i}` (simulant l'évaluation Option A du matin), mock d'un
DataFrame de breakout synthétique pour un nouveau symbole, confirme que
les 10 tickers existants **et** le nouveau survivent. Suite complète
(`portfolio` + `analysis`) : 138/138 OK avant déploiement.

**Déployé** : `backend`, `celery_worker`, `celery_worker_analysis`,
`celery_beat`, `frontend` recréés ensemble (frontend inclus par
discipline suite à la leçon de l'item 15).
