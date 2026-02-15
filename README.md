<br>

<p align="center">
<img width="500" alt="Adelon_logo" src="https://github.com/user-attachments/assets/eb0e4fec-610a-4493-95e4-bc9c1115db60" />
<br>
<br>
  
<p align="center">
  <strong>Bayesian Marketing Mix Model</strong></p>  
<br>
<br>

A Bayesian Marketing Mix Model built with PyMC. This model breaks revenue into the components that drive it — baseline, trend, seasonality, media spend across 5 channels, and external controls — then uses the fitted response curves to optimize budget allocation.

The model runs on **synthetic data with known ground truth**. That means every parameter the model estimates can be checked against the values that actually generated the data. No guessing whether the model is right.
<br>
<br>
<br>


<div align="center">
    <img src="https://custom-icon-badges.demolab.com/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=vsc&logoColor=white">
    <img src="https://img.shields.io/badge/Python-ffffff?logo=python&style=for-the-badge&color=3776AB&logoColor=ffffff">
    <img src="https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white">
    <img src="https://img.shields.io/badge/ArviZ-4DABCF?logo=ArviZ&style=for-the-badge&logoColor=fff">
    <img src="https://img.shields.io/badge/PyTensor-2CA5E0?style=for-the-badge&logo=PyTensor&logoColor=white">
    <img src="https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&style=for-the-badge&logoColor=fff">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=fff">
    <img src="https://img.shields.io/badge/SciPy-%238CAAE6?style=for-the-badge&logo=scipy&logoColor=white">
    <img src="https://img.shields.io/badge/PyMC-%23FF4B4B?style=for-the-badge&logo=PyMC&logoColor=white">
    <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B?style=for-the-badge&logo=plotly&logoColor=white">
    <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white">
</div>
<br>
<br>

## Key Findings

| Metric | Value |
|--------|-------|
| Revenue explained (R²) | 0.976 |
| MAPE | 2.57% |
| Media share of revenue | ~32% (ground truth: 32%) |
| Highest ROAS channel | Search |
| Longest carryover | TV (α = 0.85, ~3-week decay) |
| Shortest carryover | Search (α = 0.20, near-immediate) |
| MCMC convergence | R-hat < 1.005, 36 divergences (0.9%, concentrated in Print/OOH) |

## Tech Stack

- **Inference**: PyMC 5, ArviZ, pytensor
- **Data**: pandas, NumPy, SciPy
- **Visualization**: Plotly
- **Dashboard**: Streamlit
- **Testing**: pytest (80%+ coverage)
- **CI/CD**: GitHub Actions (lint, test, MCMC sampling test)
- **Containerization**: Docker (multi-stage build)

## Quick Start

The fastest way to see the model dashboard is to pull the pre-built Docker image — no local setup required:

```bash
docker pull ghcr.io/<michael-gurule>/adelon_mmm:main
docker run -p 8501:8501 ghcr.io/<michael-gurule>/adelon_mmm:main
# Open http://localhost:8501
```

The image includes a pre-trained model with all evaluation artifacts. The full pipeline (data generation, MCMC training, evaluation) was run at build time in CI to keep container startup instant.

### CLI Commands

| Command | Description |
|---------|-------------|
| `adelon-generate` | Generate synthetic data with known ground truth |
| `adelon-train` | Fit the Bayesian model via NUTS MCMC |
| `adelon-evaluate` | Produce evaluation artifacts (metrics, ROAS, contributions) |
| `adelon-run` | Run full pipeline: generate → train → evaluate |

All commands accept `--log-level` and `--log-file`. Run any command with `--help` for the full list of options.

### Docker (Recommended)

The Docker image ships with a **pre-trained model** — the full pipeline (data generation, MCMC training, evaluation) runs at build time via GitHub Actions so you don't have to. Just pull and run:

```bash
docker pull ghcr.io/<michael-gurule>/adelon_mmm:main
docker run -p 8501:8501 ghcr.io/<michael-gurule>/adelon_mmm:main
# Dashboard available immediately at http://localhost:8501
```

This approach keeps the container startup instant and avoids running MCMC sampling on your local machine.

### Running the Pipeline Locally

If you want to generate fresh data, retrain the model, or modify parameters, clone the repo and run the pipeline directly:

```bash
git clone https://github.com/<michael-gurule>/adelon_mmm.git
cd adelon_mmm
pip install -e ".[bayesian,dashboard,dev]"

# Run the full pipeline: generate -> train -> evaluate
adelon-run

# Launch the dashboard
streamlit run dashboards/app.py
```

## Project Structure

```
adelon_mmm/
├── config/
│   └── model_config.yaml          # Channel priors, MCMC settings
├── data/
│   ├── generate_data.py           # Synthetic data generator (MMMDataGenerator)
│   ├── mmm_daily_data.csv         # 1,095 days × 23 columns (generated)
│   ├── mmm_contributions_truth.csv
│   └── mmm_ground_truth.json      # True parameters for validation
├── src/
│   ├── __init__.py
│   ├── bayesian_mmm.py            # BayesianMMM class (PyMC model)
│   ├── preprocessing.py           # Adstock, saturation, data prep
│   ├── visualization.py           # 12 Plotly chart functions
│   ├── optimization.py            # Budget allocation: greedy marginal ROI + constrained SLSQP
│   ├── exceptions.py              # Domain-specific exception hierarchy
│   ├── logging_config.py          # Centralized logging setup
│   └── pipeline/
│       ├── __init__.py
│       ├── generate.py            # CLI: adelon-generate
│       ├── train.py               # CLI: adelon-train
│       ├── evaluate.py            # CLI: adelon-evaluate
│       └── run_all.py             # CLI: adelon-run (orchestrator)
├── dashboards/
│   └── app.py                     # Streamlit dashboard (6 pages)
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_preprocessing.py
│   ├── test_bayesian_mmm.py
│   ├── test_data_generation.py
│   ├── test_optimization.py
│   ├── test_pipeline.py
│   ├── test_posterior_predictions.py
│   └── test_exceptions.py
├── artifacts/                     # Evaluation outputs (gitignored)
├── traces/                        # Saved MCMC traces (gitignored)
├── .github/workflows/
│   ├── ci.yml                     # Lint + test + MCMC test (every push)
│   └── docker-build.yml           # Build + push image (src/config/Dockerfile changes only)
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Methods

### Revenue Model

Revenue is modeled as a sum of components — following standard MMM practice:

```
revenue[t] = intercept
           + trend * t
           + Σ fourier_coeffs * [sin/cos(2πkt/365.25)]
           + Σ dow_betas * day_of_week[t]
           + Σ beta_c * Hill(Adstock(spend_c[t], α_c), K_c, S_c) * intercept
           + Σ control_betas * controls[t]
           + ε[t]
```

### Adstock (Carryover)

Geometric decay captures the fact that media impact doesn't vanish overnight:

```
adstock[t] = spend[t] + α · adstock[t-1]
```

Implemented with `pytensor.scan` so the recursive computation stays differentiable within the PyMC model graph.

### Hill Saturation (Diminishing Returns)

The Hill function maps spend to a 0–1 saturation level — the classic diminishing returns curve:

```
f(x) = x^S / (K^S + x^S)
```

- **K**: half-saturation point (spend at which effect is 50%)
- **S**: steepness (higher = sharper S-curve)

### Bayesian Inference

Standard NUTS setup. The informative priors do real work here — without them, the multicollinearity between channels makes the posteriors too wide to be useful.

- **Sampler**: NUTS via PyMC (nutpie backend)
- **Chains**: 4 chains, 1,000 draws each, 1,000 tuning steps
- **Priors**: Beta on adstock alpha, HalfNormal on saturation and media coefficients
- **Diagnostics**: R-hat, ESS (bulk + tail), divergences
- **Validation**: Parameter recovery against known ground truth

## Dashboard Pages

1. **Overview** — KPIs, monthly spend/revenue trends, ground truth decomposition
2. **Channel Deep Dive** — Per-channel spend, adstock decay, saturation curves, transform pipeline
3. **Model Results** — Parameter recovery table, revenue waterfall, model fit, residuals
4. **Response Curves** — Marginal spend-response with 94% credible intervals
5. **Budget Optimizer** — Greedy marginal ROI allocator and constrained SLSQP optimizer with per-channel bounds and revenue lift projection
6. **MCMC Diagnostics** — R-hat, ESS, divergences, trace plots

## Testing

Most tests run in under a minute. The MCMC tests are slower, they actually sample, so they're marked `slow` and skipped by default.

```bash
# Fast tests (no MCMC sampling)
pytest tests/ -v -m "not slow"

# Full suite including MCMC sampling
pytest tests/ -v

# With coverage reporting
pytest tests/ -v -m "not slow" --cov=src --cov-report=term-missing
```

## References

- Broadbent, S. (1979). *One Way TV Advertisements Work*. Journal of the Market Research Society, 21(3), 139–166. — Introduced the geometric adstock concept for modeling advertising carryover effects.
- Hill, A.V. (1910). The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. *Journal of Physiology*, 40, iv–vii. — Original Hill equation, adapted here as the saturation (diminishing returns) function.
- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. *Google Inc.* — Bayesian MMM framework combining adstock and Hill saturation with MCMC inference.
- Hoffman, M.D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623. — The NUTS sampler used by PyMC for posterior inference.
- Gelman, A., & Rubin, D.B. (1992). Inference from Iterative Simulation Using Multiple Sequences. *Statistical Science*, 7(4), 457–472. — R-hat convergence diagnostic used for MCMC validation.
- Salvatier, J., Wiecki, T.V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55. — PyMC probabilistic programming framework underlying the model.
- Harvey, A.C., & Shephard, N. (1993). Structural Time Series Models. In *Handbook of Statistics*, Vol. 11. Elsevier. — Fourier-based seasonal decomposition applied to the trend and seasonality components.

## Future Enhancements

Things that would make this more useful on real data:

- Time-varying parameters (regime switching for when channel effectiveness shifts)
- Cross-channel interaction effects
- Bayesian model comparison via WAIC/LOO
- Out-of-sample validation with rolling windows
- Integration with ad platform APIs for live spend data
<br>

<br>

<h1 align="center">LET'S CONNECT!</h1>

<h3 align="center">Michael Gurule</h3>

<p align="center">
  <strong>Data Science | ML Engineering</strong>
</p>
<br>

<div align="center">
  <a href="mailto:michaelgurule1164@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
  
  <a href="michaelgurule.com">
    <img src="https://custom-icon-badges.demolab.com/badge/MICHAELGURULE.COM-150458?style=for-the-badge&logo=browser&logoColor=white"></a>
  
  <a href="www.linkedin.com/in/michael-gurule-447aa2134">
    <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=fff"></a>
  
  <a href="https://medium.com/@michaelgurule1164">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></a>
</div>
<br>

---

<p align="center">
  
<img width="450" alt="Github-Profile-Footer" src="https://github.com/user-attachments/assets/653bdf43-f5e6-4ace-a5ee-502d994bf43a" />

</p>
