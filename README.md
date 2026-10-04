# TRUST-X

Evidence-driven AI integrity assurance for computer-vision pipelines. The repository is the integrated version of the four contributors' work: data forensics, model forensics, reasoning/provenance, controlled benchmarks, and the React frontend.

## Architecture

Data Forensics + Model Forensics + Inference Integrity -> Evidence -> Integrity Graph -> Hypothesis Fusion -> Assurance Decision -> UI.

The deterministic investigator is the security authority. Optional local LLM reporting is writer-only and receives grounded facts; it cannot change the assurance decision.

## Backend

Use Python 3.11+ or another supported Python version for your environment. Create a virtual environment and install:

```bash
python -m venv .venv
# Linux / WSL
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /demo/graph`
- `GET /demo/investigate`
- `GET /graph` and `/graph/node-link`
- `GET /investigate` and `/report`
- model/data ingestion and forensic endpoints under `/models`, `/batches`, `/evidence`
- stateless reasoning under `/api/v1/investigate` and `/api/v1/counterfactual/{node_id}`
- inference passport verification under `/api/v1/inference/verify`

Run the Python test suite:

```bash
pytest -q
```

Run the controlled data benchmark:

```bash
python -m backend.benchmark.run_benchmark
```

Run the controlled inference-integrity benchmark:

```bash
python -m backend.attacks.inference_tampered
```

PyTorch is intentionally optional because its wheels are platform-specific. The model inspection/comparison/trigger endpoints support PyTorch/TorchScript when PyTorch is installed. Only load model artifacts you trust in a controlled environment.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI implements the four TRUST-X screens: Pipeline, Assurance, Investigation, and Evidence. It uses the backend demo endpoints by default. Set `VITE_API_URL` to point at another backend.

## Design notes

Behavioral and trigger thresholds are v0.1 expert heuristics and are not calibrated probabilities. Benchmark-derived detector reliability should replace placeholders before making research claims. Benign distribution shift is treated as a competing explanation, not automatically as an attack.
