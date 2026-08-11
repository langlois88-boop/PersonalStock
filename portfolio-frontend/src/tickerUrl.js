/**
 * Encode le point en %2E plutôt que de le laisser littéral dans une URL de
 * ticker -- trouvé le 2026-08-10 : un filtre réseau/logiciel de sécurité
 * côté client peut bloquer/altérer toute URL contenant ".to" (domaine
 * national des Tonga, associé au piratage), même dans un simple segment de
 * chemin comme "BTO.TO" qui n'a rien d'un vrai nom de domaine.
 * encodeURIComponent ne suffit pas ici : le point est un caractère
 * "unreserved" qu'il laisse toujours tel quel.
 *
 * Centralisé ici (au lieu de dupliqué dans chaque composant qui lie vers
 * une fiche ticker) pour que le contournement ne parte pas en drift si un
 * seul des call sites est mis à jour -- utilisé par AnalysisScreener.js et
 * toute future liste de tickers cliquables (ex. AnalyticsLabPage.js).
 * Voir TickerDetail.js pour le décodage côté réception.
 */
export const encodeTickerForUrl = (ticker) => String(ticker || '').replace(/\./g, '%2E');

export const tickerDetailPath = (ticker) => `/analysis/ticker/${encodeTickerForUrl(ticker)}`;
