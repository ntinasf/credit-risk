# Run any script through the project virtual environment
uv run python scripts/process_data.py
uv run python -m credit_risk_model.train

# Run tests
uv run pytest tests/ -v

# Run the Streamlit app
uv run streamlit run app/streamlit_app.py

# Add a new runtime dependency (updates pyproject.toml + uv.lock atomically)
uv add some-package

# Add a development-only dependency
uv add --dev pytest-mock

# Regenerate the lock file after manually editing pyproject.toml
uv lock

# Reproduce the exact environment from uv.lock (what CI/CD runs)
uv sync

# Start the full local stack (MLflow + Streamlit app)
docker compose up
# Rebuild after changing source code
docker compose up --build
# Run training inside the same environment as the container
docker compose run --rm app uv run python scripts/train_all.py


git clone https://github.com/ntinasf/credit-risk.git
cd credit-risk/credit-risk-project
uv sync
uv run python scripts/process_data.py    # raw → german_credit.csv
uv run python scripts/split_data.py      # → train + test + app sample
uv run pytest tests/ -v                  # 25 passing ✅
uv run python main.py --no-tune          # train all 4 (fast)
uv run python main.py                    # train all 4 (with BayesSearch)