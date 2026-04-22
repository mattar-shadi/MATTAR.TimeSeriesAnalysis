# MATTAR.TimeSeriesAnalysis

[![CI](https://github.com/mattar-shadi/MATTAR.TimeSeriesAnalysis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mattar-shadi/MATTAR.TimeSeriesAnalysis/actions/workflows/ci.yml)

Time series analysis library in C#/.NET, providing an **ARIMA-GARCH** model implementation for forecasting and volatility estimation.

## Features
- ARIMA(p,d,q) mean model
- GARCH(P,Q) conditional variance model
- Combined forecasting: returns *(mean, volatility)* for future steps
- Diagnostics helpers (standardized residuals, AIC, BIC)

## Requirements
- **.NET SDK 10.0** (project targets `net10.0`)

## Install / Use
### Option A: Reference the project (recommended for now)
Because this repo doesn’t currently include NuGet packaging metadata, the simplest way to use it is to reference the project directly:

```bash
# from your solution folder
git submodule add https://github.com/mattar-shadi/MATTAR.TimeSeriesAnalysis.git
```

Then add a project reference to `src/MATTAR.TimeSeriesAnalysis.csproj` from your app/test project.

### Option B: Copy the source
You can also copy `src/ArimaGarch.cs` into your project and ensure you reference MathNet.Numerics.

## Build
### Build with dotnet
```bash
dotnet restore MATTAR.TimeSeriesAnalysis.sln
dotnet build MATTAR.TimeSeriesAnalysis.sln -c Release
dotnet test  MATTAR.TimeSeriesAnalysis.sln -c Release
```

### Windows build script (build.bat)
If you want a simple Windows batch build, create a `build.bat` file at the repo root with:

```bat
@echo off
setlocal

REM Build + test (Release)
dotnet restore MATTAR.TimeSeriesAnalysis.sln || exit /b 1
dotnet build MATTAR.TimeSeriesAnalysis.sln -c Release --no-restore || exit /b 1
dotnet test  MATTAR.TimeSeriesAnalysis.sln -c Release --no-build --no-restore || exit /b 1

echo Build succeeded.
endlocal
```

## Quick start
> Note: the `ArimaGarch` class currently lives in the `ArimaGarch` namespace.

```csharp
using System;
using ArimaGarch;

var series = new double[] {
    100, 101, 99, 102, 105, 103, 104, 106, 108, 107
};

// Example: ARIMA(1,1,1) + GARCH(1,1)
var model = new ArimaGarch(p: 1, d: 1, q: 1, garchP: 1, archQ: 1);
model.Fit(series);

var forecast = model.Forecast(steps: 5);
foreach (var (mean, volatility) in forecast)
{
    Console.WriteLine($"mean={mean:F4}, vol={volatility:F4}");
}

var standardizedResiduals = model.GetStandardizedResiduals();
var aic = model.ComputeAIC();
var bic = model.ComputeBIC();
```

## Notes / Limitations
- The implementation uses simplified numeric optimization approaches; results should be validated for your use case.
- Consider adding a dedicated `Models` namespace aligned with the project’s root namespace (`MATTAR.TimeSeriesAnalysis`) if you want the public API to match the package name.

## Dependencies
- [MathNet.Numerics](https://www.nuget.org/packages/MathNet.Numerics)

## License
No license file is currently present in the repository. If you intend others to use this library, consider adding a LICENSE (MIT/Apache-2.0/etc.).
