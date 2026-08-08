/**
 * Définitions courtes des termes ML/trading utilisés dans l'interface.
 * Utilisées par <Term> / <InfoTooltip> (components/ui/Tooltip.js) pour
 * expliquer le jargon directement où il apparaît, sans forcer l'utilisateur
 * à chercher ailleurs.
 */
export const GLOSSARY = {
  sharpeRatio:
    "Rendement ajusté au risque : plus il est élevé, meilleur est le gain pour le niveau de risque pris. Au-dessus de 1, c'est généralement considéré bon.",
  maxDrawdown:
    "La plus grosse perte enregistrée entre un sommet et un creux du portefeuille, en %. Indique le pire scénario vécu jusqu'ici.",
  winRate:
    "Pourcentage de trades gagnants sur l'ensemble des trades fermés.",
  totalReturn:
    "Le gain ou la perte total(e), en %, depuis le début de la période affichée.",
  featureImportance:
    "À quel point chaque donnée d'entrée (prix, volume, sentiment...) influence les décisions du modèle ML. Plus la barre est longue, plus cette donnée pèse dans la prédiction.",
  confidence:
    "Le niveau de certitude du modèle sur sa prédiction, en %. Une confidence élevée ne garantit pas un résultat correct — c'est une estimation du modèle, pas une garantie.",
  sentiment:
    "Score qui résume le ton (positif/négatif) des actualités récentes sur ce titre, calculé à partir des articles de presse des derniers jours.",
  imbalance:
    "Déséquilibre entre les ordres d'achat et de vente sur le carnet d'ordres — peut signaler une pression acheteuse ou vendeuse à court terme.",
  rvol:
    "Volume relatif : le volume échangé aujourd'hui comparé à la moyenne des jours précédents. Au-dessus de 1, ça se négocie plus que d'habitude.",
  volumeZ:
    "Écart du volume actuel par rapport à sa moyenne récente, exprimé en écarts-types (z-score). Une valeur élevée signale une activité inhabituelle.",
  scoreGemini:
    "Note donnée par le modèle de langage (Gemini), qui commente et évalue l'opportunité identifiée par les signaux techniques.",
  patternSignal:
    "Figure de chandelier (candlestick) détectée automatiquement sur le graphique récent (ex : engulfing, doji) et utilisée comme signal d'entrée.",
  stopLossSuggested:
    "Le prix auquel le bot recommande de fermer la position pour limiter la perte si le marché va dans le mauvais sens.",
  targetSuggested:
    "Le prix de sortie visé si le trade évolue favorablement — l'objectif de gain.",
  dailyStopLoss:
    "Limite de perte maximale autorisée pour la journée : au-delà, le bot arrête d'ouvrir de nouvelles positions jusqu'au lendemain.",
  pnlRealized:
    "Profit & Loss réalisé — le gain ou la perte déjà encaissé(e) sur les positions fermées (par opposition aux positions encore ouvertes).",
  sandbox:
    "Environnement de test où le bot exécute des trades simulés (« paper trading ») avec de l'argent fictif, pour valider sa stratégie sans risque réel.",
  topExplanations:
    "Les données d'entrée qui ont le plus influencé la décision du modèle pour ce trade précis, classées par contribution.",
  featuresSnapshot:
    "Les valeurs exactes de chaque donnée d'entrée utilisées par le modèle au moment où il a ouvert cette position.",
  finalBalance:
    "Le solde du portefeuille simulé à la fin de la période affichée, en dollars.",
};

export default GLOSSARY;
