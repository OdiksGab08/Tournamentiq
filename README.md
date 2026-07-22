# TournamentIQ

TournamentIQ is a Streamlit application for exploring historical international football data, estimating match outcome probabilities, comparing teams, and running tournament and Monte Carlo simulations for the 2026 FIFA World Cup.

The dashboard is deliberately thin. It presents results from the existing feature-building, prediction, and simulation services; it does not maintain a second set of model logic.

## What the application does

- Predicts a home win, draw, or away win for a selected matchup using the persisted production model and preprocessor.
- Compares teams with the same underlying match-prediction data.
- Simulates a configured tournament bracket and Monte Carlo tournament outcomes.
- Explores verified historical match statistics.
- Surfaces artifact-backed historical coverage metrics on the Home page.

## Architecture

```text
dashboard/app.py
  └─ dashboard/navigation.py              Native Streamlit route registry
      └─ dashboard/views/                 Thin page entry points
          └─ dashboard/components/        Streamlit presentation components
              └─ dashboard/services/      Dashboard adapters and state contracts
                  └─ src/simulator/       Predictor, features, tournament engine
                      └─ models/          Persisted production model and preprocessor

src/
  ├─ ingestion/                           Source-data ingestion
  ├─ data/                                Cleaning, standardization, validation
  ├─ warehouse/                           Warehouse artifact construction
  ├─ features/                            Feature engineering
  ├─ models/                              Training, preprocessing, evaluation
  ├─ pipelines/                           Pipeline orchestration
  └─ simulator/                           Match and tournament simulation
```

The active prediction path is:

```text
Match UI → dashboard.services.match_prediction_service
         → src.simulator.predictor.Predictor
         → FeatureBuilder / historical snapshots / H2H features
         → models/preprocessor.pkl + models/best_model.pkl
```

Tournament and Monte Carlo simulations reuse that same prediction layer.

## Requirements

- Python 3.12 or later
- The tracked `data/` and `models/` artifacts available locally
- Dependencies from `pyproject.toml` / `uv.lock`

## Install

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync --group dev
```

## Run the dashboard

```powershell
uv run streamlit run dashboard\app.py
```

The persisted model is large, so the first real prediction after a process start can take longer than subsequent predictions. The dashboard caches the production predictor for the lifetime of the Streamlit process.

## Validate

Run the focused dashboard and simulation contracts:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_navigation.py `
  tests/test_match_prediction_service.py `
  tests/test_tournament_simulation_service.py `
  tests/test_monte_carlo_service.py `
  tests/test_statistics_service.py
```

Then check source quality:

```powershell
.\.venv\Scripts\python.exe -m compileall -q dashboard src tests
.\.venv\Scripts\ruff.exe check dashboard src tests
```

## Repository conventions

- Do not modify persisted models or datasets while changing the dashboard.
- Keep prediction calls behind `dashboard.services.match_prediction_service`; its result contract is shared by the predictor, comparison, and simulation experiences.
- Keep dashboard state in `st.session_state` and shared resources in Streamlit caches.
- Treat `src/` as the data, ML, and simulation layer; UI work belongs in `dashboard/`.

## License

MIT. See [LICENSE](LICENSE).
