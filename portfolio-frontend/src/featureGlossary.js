/**
 * Définitions des noms de features ML bruts, tels qu'ils apparaissent dans
 * "Top explications" / "Features (snapshot)" (LivePaperTrading.js).
 *
 * Ces noms viennent directement du registre backend
 * (portfolio/ml_engine/feature_registry.py) et varient selon le modèle
 * (STABLE/PENNY/BLUECHIP/CRYPTO/FUSION) ET selon la version du modèle qui a
 * pris la décision — un vieux trade peut afficher des noms legacy
 * (ex: "MA20", "VolumeZ") qui ont depuis été renommés dans le registre
 * unifié (ex: "sma_ratio_10_50", "volume_zscore_20"). Les deux conventions
 * sont couvertes ici via ALIASES, pour que le survol marche peu importe
 * l'âge du trade affiché.
 */

const FEATURE_GLOSSARY = {
  // Momentum / oscillateurs
  rsi_14: "Force relative sur 14 jours (RSI). Sous 30 = survendu, au-dessus de 70 = suracheté.",
  rsi_7: "Force relative sur 7 jours — version plus réactive/court terme du RSI.",
  stoch_k_14: "Stochastique %K sur 14 jours : position du prix de clôture dans son range récent, utilisé pour repérer des retournements.",
  stoch_d_14: "Stochastique %D : moyenne lissée du %K, sert de ligne de signal.",
  williams_r_14: "Williams %R sur 14 jours : oscillateur de momentum similaire au stochastique, inversé (0 = suracheté, -100 = survendu).",
  cci_20: "Commodity Channel Index sur 20 jours : mesure l'écart du prix par rapport à sa moyenne, en écarts-types.",

  // Tendance
  sma_ratio_10_50: "Ratio entre la moyenne mobile 10 jours et la moyenne mobile 50 jours. Au-dessus de 1 = tendance court terme au-dessus de la tendance longue (haussier).",
  sma_ratio_20_50: "Ratio entre la moyenne mobile 20 jours et la moyenne mobile 50 jours — même logique que sma_ratio_10_50, fenêtre légèrement plus longue.",
  sma_ratio_10_20: "Ratio entre la moyenne mobile 10 jours et la moyenne mobile 20 jours — mesure de tendance très court terme.",
  ema_ratio_9_20: "Ratio entre la moyenne mobile exponentielle 9 jours et celle à 20 jours — équivalent à sma_ratio mais avec plus de poids sur les prix récents.",
  macd_hist: "Histogramme MACD : écart entre la ligne MACD et sa ligne de signal. Positif = momentum haussier qui s'accélère.",
  macd_line: "Ligne MACD brute : différence entre deux moyennes mobiles exponentielles (12 et 26 jours), indicateur de tendance/momentum.",
  adx_14: "Average Directional Index sur 14 jours : force de la tendance (peu importe la direction). Au-dessus de 25 = tendance forte.",
  adx_di_ratio: "Ratio entre les composantes directionnelles +DI/-DI de l'ADX — indique si la pression est plutôt acheteuse ou vendeuse.",

  // Volatilité & régime
  volatility_20: "Écart-type des rendements sur 20 jours — mesure classique de volatilité récente.",
  vol_regime: "Régime de volatilité : compare la volatilité récente à sa moyenne historique pour dire si le marché est plus ou moins agité que d'habitude.",
  atr_pct_14: "Average True Range sur 14 jours, en % du prix — amplitude moyenne des mouvements récents, utilisée pour calibrer les stops.",
  bb_pct_b_20: "Position du prix dans ses bandes de Bollinger (20 jours) : proche de 1 = près de la bande haute, proche de 0 = près de la bande basse.",
  bb_bandwidth_20: "Largeur des bandes de Bollinger (20 jours) — bandes étroites signalent une phase de faible volatilité, souvent avant un mouvement.",
  volatility_spike: "Détection d'un pic de volatilité soudain et anormal par rapport au niveau récent (utilisé côté crypto).",

  // Volume
  volume_zscore_20: "Écart du volume actuel par rapport à sa moyenne 20 jours, en écarts-types (z-score). Une valeur élevée signale une activité inhabituelle.",
  rvol_20: "Volume relatif sur 20 jours : le volume d'aujourd'hui comparé à la moyenne des 20 derniers jours. Au-dessus de 1 = plus actif que d'habitude.",
  RVOL10: "Volume relatif sur 10 jours — même logique que rvol_20, fenêtre plus courte.",
  obv_zscore: "On-Balance Volume, en z-score : cumul directionnel du volume (ajouté si le prix monte, retiré s'il baisse), utilisé pour détecter une accumulation ou distribution cachée.",
  VPT_roc: "Taux de variation du Volume Price Trend — combine variation de prix et de volume pour jauger la conviction derrière un mouvement.",

  // Momentum prix / structure
  return_20d: "Rendement du titre sur les 20 derniers jours, en %.",
  return_5d: "Rendement du titre sur les 5 derniers jours, en %.",
  return_1: "Rendement de la dernière période (1 jour ou 1 bougie selon le modèle).",
  mom_zscore_20: "Momentum des 20 derniers jours, exprimé en z-score par rapport à son historique.",
  linear_slope_20: "Pente de la régression linéaire du prix sur les 20 derniers jours — direction et intensité de la tendance récente.",
  slope_accel_20: "Accélération de la pente (linear_slope_20) — indique si la tendance récente s'accélère ou ralentit.",
  donchian_pct_20: "Position du prix dans son canal de Donchian (plus haut/plus bas des 20 derniers jours), en %.",
  fib_distance_50: "Distance du prix par rapport au niveau de retracement de Fibonacci le plus proche, sur 50 jours.",
  price_to_vwap_20: "Écart entre le prix actuel et le prix moyen pondéré par le volume (VWAP) sur 20 jours.",
  price_to_vwap: "Écart entre le prix actuel et le VWAP de la session.",
  rubber_band_20: "Indicateur \"élastique\" : mesure à quel point le prix s'est éloigné de sa moyenne — plus il est étiré, plus un retour à la moyenne devient probable.",
  rubber_band_index: "Version crypto de rubber_band_20 — même logique de retour à la moyenne.",

  // Bougie / patterns
  candle_body_pct: "Taille du corps de la dernière bougie par rapport à son range total, en % — une bougie à corps large signale une conviction directionnelle forte.",
  close_pos_in_range: "Position du prix de clôture dans le range (haut-bas) de la période — proche de 1 = a clôturé près du plus haut.",
  gap_pct: "Écart en % entre la clôture précédente et l'ouverture actuelle (gap).",
  pattern_doji: "Figure de chandelier Doji détectée (ouverture ≈ clôture, indécision du marché).",
  pattern_hammer: "Figure de chandelier Marteau détectée (souvent un signal de retournement haussier après une baisse).",
  pattern_engulfing: "Figure de chandelier Engulfing (avalante) détectée — une bougie qui \"avale\" complètement la précédente, signal de retournement.",
  pattern_morning_star: "Figure de chandelier Étoile du matin détectée — motif de retournement haussier sur 3 bougies.",

  // Macro / marché
  spy_correlation: "Corrélation du titre avec l'indice S&P 500 (SPY) — proche de 1 = bouge dans le même sens que le marché.",
  spy_corr_60: "Corrélation avec le S&P 500 (SPY) sur une fenêtre de 60 jours.",
  tsx_corr_60: "Corrélation avec l'indice canadien TSX sur 60 jours.",
  sector_beta: "Sensibilité du titre aux mouvements de son secteur — au-dessus de 1 = amplifie les mouvements du secteur.",
  sector_beta_60: "Beta sectoriel calculé sur une fenêtre de 60 jours.",
  btc_correlation: "Corrélation du titre avec le Bitcoin — pertinent pour les actions liées aux cryptomonnaies.",
  VIXCLS: "Indice de volatilité VIX (\"indice de la peur\") — plus il est élevé, plus le marché anticipe de turbulence.",
  DCOILWTICO: "Prix du pétrole brut WTI (série macro FRED) — pertinent pour les titres énergie ou sensibles aux matières premières.",
  CPIAUCSL: "Indice des prix à la consommation américain (inflation, série macro FRED).",
  fred_rate: "Taux d'intérêt de référence (données macro FRED).",

  // Order book / microstructure
  bid_ask_spread_pct: "Écart entre le meilleur prix d'achat et de vente, en % du prix — un spread large signale un titre moins liquide.",
  order_book_imbalance: "Déséquilibre entre les ordres d'achat et de vente sur le carnet d'ordres — peut signaler une pression acheteuse ou vendeuse à court terme.",
  trade_velocity: "Vitesse à laquelle les transactions s'enchaînent — une accélération peut signaler un regain d'intérêt soudain.",

  // Fondamentaux / sentiment
  sentiment_score: "Score de sentiment des actualités récentes sur ce titre (positif/négatif), calculé à partir des articles de presse.",
  news_sentiment: "Alias de sentiment_score utilisé par le modèle recommender.",
  news_count: "Nombre d'articles de presse récents pris en compte pour ce titre.",
  dividend_yield: "Rendement du dividende annuel, en % du prix de l'action.",
  roe: "Return on Equity — rentabilité de l'entreprise par rapport aux capitaux propres.",
  debt_to_equity: "Ratio dette / capitaux propres — mesure de l'endettement de l'entreprise.",
  frac_diff_close: "Prix de clôture après différenciation fractionnaire — une transformation statistique qui rend la série plus stable tout en gardant un peu de mémoire des niveaux de prix (technique avancée pour l'entraînement du modèle).",
  vol_zscore: "Alias de volume_zscore_20 utilisé par le modèle recommender.",
};

// Alias vers les noms legacy encore visibles sur d'anciens trades
// (modèles entraînés avant l'unification du registre de features).
const ALIASES = {
  MA20: FEATURE_GLOSSARY.sma_ratio_10_50,
  VolumeZ: FEATURE_GLOSSARY.volume_zscore_20,
  Sentiment: FEATURE_GLOSSARY.sentiment_score,
  RSI: FEATURE_GLOSSARY.rsi_14,
};

/**
 * Retourne la définition d'une feature par son nom brut, tel qu'affiché par
 * le backend (insensible à la casse, avec repli sur les alias legacy).
 */
export function getFeatureExplanation(name) {
  if (!name) return null;
  if (FEATURE_GLOSSARY[name]) return FEATURE_GLOSSARY[name];
  if (ALIASES[name]) return ALIASES[name];
  const lower = String(name).toLowerCase();
  const match = Object.keys(FEATURE_GLOSSARY).find((k) => k.toLowerCase() === lower);
  if (match) return FEATURE_GLOSSARY[match];
  const aliasMatch = Object.keys(ALIASES).find((k) => k.toLowerCase() === lower);
  if (aliasMatch) return ALIASES[aliasMatch];
  return "Indicateur technique utilisé par le modèle — pas encore documenté dans le glossaire.";
}

export default FEATURE_GLOSSARY;
