using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using MathNet.Numerics.Statistics;

namespace ArimaGarch
{
    /// <summary>
    /// Implémentation complète du modèle ARIMA-GARCH en C#.
    /// ARIMA (AutoRegressive Integrated Moving Average) modélise la moyenne conditionnelle.
    /// GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) modélise la variance conditionnelle.
    /// Ce modèle combiné est très utilisé en finance pour les séries temporelles avec volatilité variable.
    /// </summary>
    public class ArimaGarch
    {
        // ─── Paramètres ARIMA ───────────────────────────────────────────────────────

        /// <summary>Ordre AR (p) : nombre de termes autorégressifs dans ARIMA.</summary>
        private readonly int _p;

        /// <summary>Ordre d'intégration (d) : nombre de différenciations pour stationnariser la série.</summary>
        private readonly int _d;

        /// <summary>Ordre MA (q) : nombre de termes de moyenne mobile dans ARIMA.</summary>
        private readonly int _q;

        // ─── Paramètres GARCH ───────────────────────────────────────────────────────

        /// <summary>Ordre GARCH (P) : nombre de termes de variance conditionnelle retardée.</summary>
        private readonly int _garchP;

        /// <summary>Ordre ARCH (Q) : nombre de termes d'erreurs au carré retardées.</summary>
        private readonly int _archQ;

        // ─── Paramètres estimés ARIMA ───────────────────────────────────────────────

        /// <summary>Constante (mu) dans l'équation ARIMA.</summary>
        private double _mu;

        /// <summary>Coefficients AR [phi_1, ..., phi_p].</summary>
        private double[] _arCoeffs;

        /// <summary>Coefficients MA [theta_1, ..., theta_q].</summary>
        private double[] _maCoeffs;

        // ─── Paramètres estimés GARCH ───────────────────────────────────────────────

        /// <summary>Constante omega dans l'équation de variance GARCH (omega > 0).</summary>
        private double _omega;

        /// <summary>Coefficients ARCH [alpha_1, ..., alpha_Q] (effets des chocs passés sur la variance).</summary>
        private double[] _archCoeffs;

        /// <summary>Coefficients GARCH [beta_1, ..., beta_P] (persistance de la variance).</summary>
        private double[] _garchCoeffs;

        // ─── Données internes ────────────────────────────────────────────────────────

        /// <summary>Série originale fournie par l'utilisateur.</summary>
        private double[] _originalSeries;

        /// <summary>Série différenciée d fois (stationnaire).</summary>
        private double[] _differencedSeries;

        /// <summary>Résidus de l'équation ARIMA après estimation.</summary>
        private double[] _residuals;

        /// <summary>Variances conditionnelles estimées par le modèle GARCH.</summary>
        private double[] _conditionalVariances;

        /// <summary>
        /// Constructeur principal du modèle ARIMA-GARCH.
        /// </summary>
        /// <param name="p">Ordre AR de l'ARIMA (nombre de lags autorégressifs).</param>
        /// <param name="d">Ordre d'intégration (nombre de différenciations).</param>
        /// <param name="q">Ordre MA de l'ARIMA (nombre de lags de moyenne mobile).</param>
        /// <param name="garchP">Ordre P du GARCH (lags de variance conditionnelle).</param>
        /// <param name="archQ">Ordre Q de l'ARCH (lags des erreurs au carré).</param>
        public ArimaGarch(int p, int d, int q, int garchP = 1, int archQ = 1)
        {
            _p = p;
            _d = d;
            _q = q;
            _garchP = garchP;
            _archQ = archQ;

            // Initialisation des tableaux de coefficients
            _arCoeffs   = new double[p];
            _maCoeffs   = new double[q];
            _archCoeffs = new double[archQ];
            _garchCoeffs= new double[garchP];
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  1. DIFFÉRENCIATION
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Différencie la série <paramref name="times"/> exactement <paramref name="times"/> fois.
        /// Une différenciation d'ordre 1 : y'_t = y_t - y_{t-1}.
        /// Une différenciation d'ordre 2 : y''_t = y'_t - y'_{t-1}, etc.
        /// </summary>
        /// <param name="series">Série temporelle d'entrée.</param>
        /// <param name="times">Nombre de fois à différencier (= d dans ARIMA).</param>
        /// <returns>Série différenciée (longueur = série.Length - times).</returns>
        private static double[] Difference(double[] series, int times)
        {
            double[] result = (double[])series.Clone();

            for (int i = 0; i < times; i++)
            {
                // À chaque itération on remplace result par sa différence première
                double[] diff = new double[result.Length - 1];
                for (int t = 1; t < result.Length; t++)
                    diff[t - 1] = result[t] - result[t - 1];
                result = diff;
            }

            return result;
        }

        /// <summary>
        /// Inverse la différenciation pour repasser dans l'espace original.
        /// Utilisé pour reconstruire les prévisions à partir de la série différenciée.
        /// </summary>
        /// <param name="diffSeries">Série différenciée.</param>
        /// <param name="originalSeries">Série originale (pour récupérer les valeurs initiales).</param>
        /// <param name="d">Nombre de différenciations à inverser.</param>
        /// <returns>Série reconstituée dans l'espace original.</returns>
        private static double[] InvertDifference(double[] diffSeries, double[] originalSeries, int d)
        {
            double[] result = (double[])diffSeries.Clone();

            for (int i = 0; i < d; i++)
            {
                // La valeur initiale de la série avant la i-ème différenciation
                // correspond à originalSeries[d - 1 - i]
                double[] integrated = new double[result.Length + 1];
                integrated[0] = originalSeries[d - 1 - i];

                for (int t = 1; t < integrated.Length; t++)
                    integrated[t] = integrated[t - 1] + result[t - 1];

                result = integrated;
            }

            return result;
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  2. ESTIMATION ARIMA (OLS simplifié sur la partie AR+MA)
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Estime les paramètres ARIMA par moindres carrés non linéaires (approximation).
        /// On initialise les résidus à zéro puis on les met à jour itérativement.
        ///
        /// L'équation ARIMA(p,d,q) sur la série différenciée y_t :
        ///   y_t = mu
        ///       + phi_1*y_{t-1} + ... + phi_p*y_{t-p}   (partie AR)
        ///       + theta_1*e_{t-1} + ... + theta_q*e_{t-q} (partie MA)
        ///       + e_t
        /// </summary>
        /// <param name="series">Série différenciée (stationnaire).</param>
        private void EstimateArima(double[] series)
        {
            int n     = series.Length;
            int start = Math.Max(_p, _q); // Indice de départ après les lags nécessaires

            // Nombre total de paramètres : 1 (mu) + p (AR) + q (MA)
            int numParams = 1 + _p + _q;

            // Initialisation des paramètres à zéro
            double[] theta = new double[numParams];
            theta[0] = series.Average(); // Initialiser mu à la moyenne

            // Résidus courants (initialement zéro)
            double[] residuals = new double[n];

            // Descente de gradient simple (gradient numérique) pour minimiser SSR
            double learningRate = 1e-4;
            int maxIter = 500;
            double tol  = 1e-8;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // ── Calcul des résidus avec les paramètres courants ──────────────
                double[] newResiduals = ComputeArimaResiduals(series, theta, residuals, start, n);

                // SSR (Sum of Squared Residuals)
                double ssr = newResiduals.Skip(start).Sum(r => r * r);

                // ── Gradient numérique pour chaque paramètre ─────────────────────
                double[] gradient = new double[numParams];
                double eps = 1e-6;

                for (int k = 0; k < numParams; k++)
                {
                    double[] thetaPlus = (double[])theta.Clone();
                    thetaPlus[k] += eps;
                    double[] resPlus = ComputeArimaResiduals(series, thetaPlus, residuals, start, n);
                    double ssrPlus = resPlus.Skip(start).Sum(r => r * r);

                    gradient[k] = (ssrPlus - ssr) / eps;
                }

                // ── Mise à jour des paramètres ────────────────────────────────────
                double gradNorm = Math.Sqrt(gradient.Sum(g => g * g));
                if (gradNorm < tol) break;

                for (int k = 0; k < numParams; k++)
                    theta[k] -= learningRate * gradient[k] / (gradNorm + 1e-10);

                residuals = newResiduals;
            }

            // ── Extraction des paramètres estimés ────────────────────────────────
            _mu = theta[0];
            for (int i = 0; i < _p; i++) _arCoeffs[i] = theta[1 + i];
            for (int i = 0; i < _q; i++) _maCoeffs[i] = theta[1 + _p + i];

            // Calcul final des résidus
            _residuals = ComputeArimaResiduals(series, theta, residuals, start, n);
        }

        /// <summary>
        /// Calcule les résidus de l'équation ARIMA pour un jeu de paramètres donné.
        /// </summary>
        /// <param name="series">Série différenciée.</param>
        /// <param name="theta">Vecteur de paramètres [mu, AR..., MA...].</param>
        /// <param name="prevResiduals">Résidus de l'itération précédente (pour la partie MA).</param>
        /// <param name="start">Indice de départ.</param>
        /// <param name="n">Longueur de la série.</param>
        /// <returns>Nouveau vecteur de résidus.</returns>
        private double[] ComputeArimaResiduals(
            double[] series, double[] theta,
            double[] prevResiduals, int start, int n)
        {
            double mu = theta[0];
            double[] ar = new double[_p];
            double[] ma = new double[_q];
            for (int i = 0; i < _p; i++) ar[i] = theta[1 + i];
            for (int i = 0; i < _q; i++) ma[i] = theta[1 + _p + i];

            double[] res = new double[n];

            for (int t = start; t < n; t++)
            {
                // Valeur prédite : mu + partie AR + partie MA
                double predicted = mu;

                for (int i = 0; i < _p; i++)
                    predicted += ar[i] * series[t - 1 - i];

                for (int i = 0; i < _q; i++)
                    predicted += ma[i] * prevResiduals[t - 1 - i];

                // Résidu = valeur réelle - valeur prédite
                res[t] = series[t] - predicted;
            }

            return res;
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  3. ESTIMATION GARCH (MLE simplifié par log-vraisemblance gaussienne)
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Estime les paramètres GARCH(P,Q) sur les résidus ARIMA par maximisation
        /// de la log-vraisemblance sous hypothèse gaussienne conditionnelle.
        ///
        /// Équation de variance GARCH(P,Q) :
        ///   h_t = omega
        ///       + alpha_1*e²_{t-1} + ... + alpha_Q*e²_{t-Q}   (partie ARCH)
        ///       + beta_1*h_{t-1}   + ... + beta_P*h_{t-P}     (partie GARCH)
        ///
        /// Condition de stationnarité : sum(alpha) + sum(beta) < 1
        /// </summary>
        /// <param name="residuals">Résidus de l'équation ARIMA.</param>
        private void EstimateGarch(double[] residuals)
        {
            int n     = residuals.Length;
            int start = Math.Max(_archQ, _garchP);

            // Variance inconditionnelle (initialisation de h_t)
            double unconditionalVar = residuals.Select(r => r * r).Average();

            // Nombre de paramètres GARCH : 1 (omega) + Q (alpha) + P (beta)
            int numParams = 1 + _archQ + _garchP;

            // Initialisation : omega petit, alpha et beta uniformément répartis
            // sous contrainte sum(alpha)+sum(beta) < 1
            double[] garchTheta = new double[numParams];
            garchTheta[0] = unconditionalVar * 0.1; // omega
            for (int i = 0; i < _archQ; i++) garchTheta[1 + i]         = 0.1 / _archQ;
            for (int i = 0; i < _garchP; i++) garchTheta[1 + _archQ + i] = 0.8 / _garchP;

            // Optimisation par gradient numérique (maximisation de log-vraisemblance)
            double learningRate = 1e-5;
            int maxIter = 1000;
            double tol  = 1e-10;

            for (int iter = 0; iter < maxIter; iter++)
            {
                // Log-vraisemblance courante
                double ll = ComputeGarchLogLikelihood(residuals, garchTheta, unconditionalVar, start, n);

                // Gradient numérique
                double[] gradient = new double[numParams];
                double eps = 1e-7;

                for (int k = 0; k < numParams; k++)
                {
                    double[] thetaPlus = (double[])garchTheta.Clone();
                    thetaPlus[k] += eps;

                    // On ne perturbe que si la contrainte de stationnarité est respectée
                    if (IsGarchStationary(thetaPlus))
                    {
                        double llPlus = ComputeGarchLogLikelihood(residuals, thetaPlus, unconditionalVar, start, n);
                        gradient[k] = (llPlus - ll) / eps;
                    }
                }

                double gradNorm = Math.Sqrt(gradient.Sum(g => g * g));
                if (gradNorm < tol) break;

                // Mise à jour avec projection sur le domaine admissible
                double[] proposed = new double[numParams];
                for (int k = 0; k < numParams; k++)
                    proposed[k] = garchTheta[k] + learningRate * gradient[k] / (gradNorm + 1e-10);

                // Projection : tous les paramètres > 0 et stationnarité
                ProjectGarchParams(proposed);
                if (IsGarchStationary(proposed))
                    garchTheta = proposed;
            }

            // ── Extraction des paramètres estimés ────────────────────────────────
            _omega = garchTheta[0];
            for (int i = 0; i < _archQ; i++)  _archCoeffs[i]  = garchTheta[1 + i];
            for (int i = 0; i < _garchP; i++) _garchCoeffs[i] = garchTheta[1 + _archQ + i];

            // Calcul final des variances conditionnelles
            _conditionalVariances = ComputeConditionalVariances(residuals, garchTheta, unconditionalVar, n);
        }

        /// <summary>
        /// Calcule la log-vraisemblance gaussienne du modèle GARCH.
        /// LL = -0.5 * sum_t [ log(2π) + log(h_t) + e²_t / h_t ]
        /// </summary>
        private double ComputeGarchLogLikelihood(
            double[] residuals, double[] theta,
            double unconditionalVar, int start, int n)
        {
            double[] h = ComputeConditionalVariances(residuals, theta, unconditionalVar, n);

            double ll = 0.0;
            for (int t = start; t < n; t++)
            {
                if (h[t] <= 0) return double.NegativeInfinity;
                ll -= 0.5 * (Math.Log(2 * Math.PI) + Math.Log(h[t]) + residuals[t] * residuals[t] / h[t]);
            }

            return ll;
        }

        /// <summary>
        /// Calcule les variances conditionnelles h_t pour un jeu de paramètres GARCH.
        /// h_t = omega + sum_i(alpha_i * e²_{t-i}) + sum_j(beta_j * h_{t-j})
        /// Les valeurs initiales h_t pour t < start sont fixées à la variance inconditionnelle.
        /// </summary>
        private double[] ComputeConditionalVariances(
            double[] residuals, double[] theta,
            double unconditionalVar, int n)
        {
            double omega = theta[0];
            double[] alpha = new double[_archQ];
            double[] beta  = new double[_garchP];
            for (int i = 0; i < _archQ; i++)  alpha[i] = theta[1 + i];
            for (int i = 0; i < _garchP; i++) beta[i]  = theta[1 + _archQ + i];

            double[] h = new double[n];

            // Initialisation : variance inconditionnelle
            for (int t = 0; t < n; t++)
                h[t] = unconditionalVar;

            int start = Math.Max(_archQ, _garchP);

            for (int t = start; t < n; t++)
            {
                h[t] = omega;

                // Terme ARCH : effets des chocs passés
                for (int i = 0; i < _archQ; i++)
                    h[t] += alpha[i] * residuals[t - 1 - i] * residuals[t - 1 - i];

                // Terme GARCH : persistance de la variance passée
                for (int j = 0; j < _garchP; j++)
                    h[t] += beta[j] * h[t - 1 - j];

                // Plancher numérique : h_t doit rester positif
                if (h[t] < 1e-10) h[t] = 1e-10;
            }

            return h;
        }

        /// <summary>
        /// Projette les paramètres GARCH dans le domaine admissible :
        /// omega > 0, alpha_i >= 0, beta_j >= 0.
        /// </summary>
        private static void ProjectGarchParams(double[] theta)
        {
            for (int k = 0; k < theta.Length; k++)
                if (theta[k] < 1e-8) theta[k] = 1e-8;
        }

        /// <summary>
        /// Vérifie la condition de stationnarité du GARCH :
        /// sum(alpha) + sum(beta) < 1.
        /// Si cette condition n'est pas respectée, la variance explose.
        /// </summary>
        private bool IsGarchStationary(double[] theta)
        {
            double sumAlpha = 0, sumBeta = 0;
            for (int i = 0; i < _archQ; i++)  sumAlpha += theta[1 + i];
            for (int i = 0; i < _garchP; i++) sumBeta  += theta[1 + _archQ + i];
            return (sumAlpha + sumBeta) < 0.9999 && theta[0] > 0;
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  4. MÉTHODE PRINCIPALE : FIT
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Ajuste le modèle ARIMA-GARCH à la série temporelle fournie.
        /// Étapes :
        ///   1. Différenciation d'ordre d (pour stationnariser la série).
        ///   2. Estimation ARIMA sur la série différenciée.
        ///   3. Estimation GARCH sur les résidus ARIMA.
        /// </summary>
        /// <param name="series">Série temporelle brute (ex. prix d'actions, taux de change…).</param>
        public void Fit(double[] series)
        {
            if (series == null || series.Length < 2)
                throw new ArgumentException("La série doit contenir au moins 2 observations.");

            _originalSeries = (double[])series.Clone();

            // ── Étape 1 : Différenciation ─────────────────────────────────────────
            _differencedSeries = Difference(series, _d);

            Console.WriteLine($"[ARIMA-GARCH] Série originale : {series.Length} pts → " +
                              $"après différenciation d={_d} : {_differencedSeries.Length} pts");

            // ── Étape 2 : Estimation ARIMA ────────────────────────────────────────
            Console.WriteLine("[ARIMA-GARCH] Estimation ARIMA...");
            EstimateArima(_differencedSeries);

            Console.WriteLine($"  mu    = {_mu:F6}");
            for (int i = 0; i < _p; i++) Console.WriteLine($"  phi_{i+1} = {_arCoeffs[i]:F6}");
            for (int i = 0; i < _q; i++) Console.WriteLine($"  theta_{i+1} = {_maCoeffs[i]:F6}");

            // ── Étape 3 : Estimation GARCH ────────────────────────────────────────
            Console.WriteLine("[ARIMA-GARCH] Estimation GARCH...");
            EstimateGarch(_residuals);

            Console.WriteLine($"  omega = {_omega:F6}");
            for (int i = 0; i < _archQ; i++)  Console.WriteLine($"  alpha_{i+1} = {_archCoeffs[i]:F6}");
            for (int i = 0; i < _garchP; i++) Console.WriteLine($"  beta_{i+1}  = {_garchCoeffs[i]:F6}");
            Console.WriteLine("[ARIMA-GARCH] Estimation terminée.");
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  5. PRÉVISION (FORECAST)
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Prédit les <paramref name="steps"/> prochaines valeurs de la série originale
        /// ainsi que la volatilité (écart-type conditionnel) associée.
        ///
        /// Retourne un tableau de tuples (moyenne prévue, volatilité prévue).
        /// </summary>
        /// <param name="steps">Nombre de pas à prévoir.</param>
        /// <returns>Tableau de (forecast_mean, forecast_volatility) de longueur steps.</returns>
        public (double mean, double volatility)[] Forecast(int steps)
        {
            int n = _differencedSeries.Length;
            int resLen = _residuals.Length;

            // ── Prévision ARIMA ───────────────────────────────────────────────────
            // On étend la série différenciée avec les valeurs prévues
            double[] extSeries    = new double[n + steps];
            double[] extResiduals = new double[resLen + steps];
            Array.Copy(_differencedSeries, extSeries, n);
            Array.Copy(_residuals, extResiduals, resLen);

            for (int h = 0; h < steps; h++)
            {
                int t = n + h;  // Indice dans la série étendue

                double predicted = _mu;

                // Partie AR : utilise les valeurs réelles si disponibles, sinon les prévisions
                for (int i = 0; i < _p; i++)
                {
                    int idx = t - 1 - i;
                    predicted += _arCoeffs[i] * (idx < n ? _differencedSeries[idx] : extSeries[idx]);
                }

                // Partie MA : résidus futurs = 0 (espérance des chocs futurs)
                for (int i = 0; i < _q; i++)
                {
                    int idx = t - 1 - i;
                    predicted += _maCoeffs[i] * (idx < resLen ? _residuals[idx] : 0.0);
                }

                extSeries[t]    = predicted;
                extResiduals[t] = 0.0; // Résidus futurs nuls en moyenne
            }

            // ── Prévision GARCH ───────────────────────────────────────────────────
            double[] extVariances = new double[resLen + steps];
            Array.Copy(_conditionalVariances, extVariances, resLen);

            for (int h = 0; h < steps; h++)
            {
                int t = resLen + h;

                double hForecast = _omega;

                // Terme ARCH : chocs futurs = variance conditionnelle prévue (espérance)
                for (int i = 0; i < _archQ; i++)
                {
                    int idx = t - 1 - i;
                    // Pour les chocs futurs on utilise h_{t-i} car E[e²_{t-i}] = h_{t-i}
                    hForecast += _archCoeffs[i] * (idx < resLen
                        ? _residuals[idx] * _residuals[idx]
                        : extVariances[idx]);
                }

                // Terme GARCH : persistance de la variance
                for (int j = 0; j < _garchP; j++)
                {
                    int idx = t - 1 - j;
                    hForecast += _garchCoeffs[j] * extVariances[idx];
                }

                extVariances[t] = Math.Max(hForecast, 1e-10);
            }

            // ── Inversion de la différenciation ──────────────────────────────────
            // Récupérer uniquement les steps prévisions dans l'espace différencié
            double[] diffForecasts = extSeries.Skip(n).Take(steps).ToArray();

            // Inverser la différenciation pour revenir à l'espace original
            double[] originalForecasts = InvertDifference(diffForecasts, _originalSeries, _d);
            // On ne garde que les `steps` dernières valeurs reconstituées
            originalForecasts = originalForecasts.Skip(originalForecasts.Length - steps).Take(steps).ToArray();

            // ── Construction du résultat ──────────────────────────────────────────
            var result = new (double mean, double volatility)[steps];
            for (int h = 0; h < steps; h++)
            {
                result[h] = (
                    mean:       originalForecasts[h],
                    volatility: Math.Sqrt(extVariances[resLen + h])  // Volatilité = écart-type conditionnel
                );
            }

            return result;
        }

        // ════════════════════════════════════════════════════════════════════════════
        //  6. DIAGNOSTICS
        // ════════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Retourne les résidus standardisés : z_t = e_t / sqrt(h_t).
        /// Un bon modèle GARCH produira des z_t proches d'un bruit blanc gaussien N(0,1).
        /// </summary>
        public double[] GetStandardizedResiduals()
        {
            int n = Math.Min(_residuals.Length, _conditionalVariances.Length);
            double[] z = new double[n];
            for (int t = 0; t < n; t++)
            {
                double sigma = Math.Sqrt(_conditionalVariances[t]);
                z[t] = sigma > 1e-10 ? _residuals[t] / sigma : 0.0;
            }
            return z;
        }

        /// <summary>
        /// Calcule le critère d'information d'Akaike (AIC) du modèle GARCH.
        /// AIC = -2 * LL + 2 * k   (k = nombre de paramètres GARCH)
        /// Un AIC plus faible indique un meilleur compromis ajustement/complexité.
        /// </summary>
        public double ComputeAIC()
        {
            int n = _residuals.Length;
            int start = Math.Max(_archQ, _garchP);
            double unconditionalVar = _residuals.Select(r => r * r).Average();

            // Reconstruction du vecteur de paramètres GARCH
            double[] theta = new double[1 + _archQ + _garchP];
            theta[0] = _omega;
            for (int i = 0; i < _archQ; i++)  theta[1 + i]         = _archCoeffs[i];
            for (int i = 0; i < _garchP; i++) theta[1 + _archQ + i] = _garchCoeffs[i];

            double ll = ComputeGarchLogLikelihood(_residuals, theta, unconditionalVar, start, n);
            int k = 1 + _p + _q + 1 + _archQ + _garchP; // Paramètres ARIMA + GARCH

            return -2.0 * ll + 2.0 * k;
        }

        /// <summary>
        /// Calcule le critère d'information bayésien (BIC) du modèle GARCH.
        /// BIC = -2 * LL + k * log(n)
        /// Pénalise davantage la complexité que l'AIC pour des échantillons grands.
        /// </summary>
        public double ComputeBIC()
        {
            int n = _residuals.Length;
            int start = Math.Max(_archQ, _garchP);
            double unconditionalVar = _residuals.Select(r => r * r).Average();

            double[] theta = new double[1 + _archQ + _garchP];
            theta[0] = _omega;
            for (int i = 0; i < _archQ; i++)  theta[1 + i]         = _archCoeffs[i];
            for (int i = 0; i < _garchP; i++) theta[1 + _archQ + i] = _garchCoeffs[i];

            double ll = ComputeGarchLogLikelihood(_residuals, theta, unconditionalVar, start, n);
            int k = 1 + _p + _q + 1 + _archQ + _garchP;

            return -2.0 * ll + k * Math.Log(n);
        }
    }
}
