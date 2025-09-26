# Activate venv
source .venv/bin/activate

# Run FastAPI with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
