using System;
using System.Linq;
using ArimaGarch;
using MathNet.Numerics.Statistics;

// ── Génération d'une série synthétique type rendements financiers ─────
// Modèle génératif : ARIMA(1,1,1) + GARCH(1,1)
int seed = 42;
var rng  = new Random(seed);
int T    = 500;

// Paramètres vrais du processus générateur
double trueOmega = 0.00001;
double trueAlpha = 0.10;    // Effet ARCH
double trueBeta  = 0.85;    // Persistance GARCH
double truePhi   = 0.3;     // Coefficient AR
double trueTheta = 0.2;     // Coefficient MA

double[] returns = new double[T];
double   h       = trueOmega / (1 - trueAlpha - trueBeta); // Variance stationnaire
double   prevE   = 0;

for (int t = 0; t < T; t++)
{
    // Tirage standard normal (Box-Muller)
    double u1 = 1 - rng.NextDouble();
    double u2 = 1 - rng.NextDouble();
    double z  = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);

    // Choc GARCH
    double e = Math.Sqrt(h) * z;

    // Partie ARIMA(1,0,1) sur les rendements (d=0 car déjà différencié)
    returns[t] = 0.001 + (t > 0 ? truePhi * returns[t - 1] : 0) + trueTheta * prevE + e;

    // Mise à jour variance GARCH
    h = trueOmega + trueAlpha * e * e + trueBeta * h;
    if (h < 1e-10) h = 1e-10;

    prevE = e;
}

// ── Instanciation et ajustement du modèle ARIMA(1,0,1)-GARCH(1,1) ───
// d=0 car les rendements sont déjà une série stationnaire
var model = new ArimaGarch.ArimaGarch(p: 1, d: 0, q: 1, garchP: 1, archQ: 1);
model.Fit(returns);

// ── Critères d'information ────────────────────────────────────────────
Console.WriteLine($"\nAIC = {model.ComputeAIC():F4}");
Console.WriteLine($"BIC = {model.ComputeBIC():F4}");

// ── Prévision sur 5 pas ───────────────────────────────────────────────
Console.WriteLine("\n── Prévisions (5 pas) ──────────────────────────────");
var forecasts = model.Forecast(steps: 5);
for (int h2 = 0; h2 < forecasts.Length; h2++)
{
    Console.WriteLine($"  t+{h2+1} : mean={forecasts[h2].mean:+0.000000;-0.000000}  " +
                      $"volatility={forecasts[h2].volatility:F6}");
}

// ── Diagnostic : résidus standardisés ────────────────────────────────
double[] stdRes = model.GetStandardizedResiduals();
double   meanZ  = stdRes.Average();
double   stdZ   = Statistics.StandardDeviation(stdRes);
Console.WriteLine($"\nRésidus standardisés : mean={meanZ:F4}  std={stdZ:F4}");
Console.WriteLine("(Idéalement : mean≈0, std≈1 pour un bon modèle GARCH)");
