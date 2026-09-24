# Digit Recognizer

A containerized MNIST digit recognizer: draw a digit in the browser (or upload an image) and get a prediction back from a ResNet18 model served over a REST API.

This is the Milestone 2 deliverable for SYS-304 (Scalable Algorithms and Infrastructure) — wrapping the Phase 1 model in a web service, a UI, container images, and an automated CI pipeline.

A demo recording is included at [`Digit Recognizer.mp4`](Digit%20Recognizer.mp4).

## Architecture

```
┌────────────────┐        POST /predict        ┌──────────────────┐
│  frontend (nginx)│ ───────────────────────────▶│  backend (FastAPI)│
│  index.html + JS │◀─────────────────────────── │  ResNet18 (timm)  │
└────────────────┘        JSON prediction       └──────────────────┘
     port 8080                                        port 8000
```

- **Backend** ([backend/](backend/)): FastAPI service serving a `timm` ResNet18 (first conv layer swapped for single-channel/grayscale input) trained on MNIST-style digit data. Exposes `GET /health` and `POST /predict` (accepts an image file, returns the predicted digit, confidence, and full per-digit probability distribution).
- **Frontend** ([frontend/](frontend/)): a single static page with a drawable HTML canvas and a file-upload fallback, served by nginx. No build step or framework — plain JS.
- **Containerization**: each service has its own [backend/Dockerfile](backend/Dockerfile) / [frontend/Dockerfile](frontend/Dockerfile), orchestrated together with [docker-compose.yml](docker-compose.yml).

## Running it

```bash
./deploy.sh
```

This builds and starts both containers via `docker compose up --build -d`, waits for the backend's `/health` check to pass, then prints the URLs:

- Frontend: http://localhost:8080
- Backend: http://localhost:8000

To stop everything: `docker compose down`.

## Testing

Four layers of tests, each covering a different part of the stack:

| Layer | Location | What it covers |
|---|---|---|
| Backend unit tests | [backend/tests/test_preprocess.py](backend/tests/test_preprocess.py) | Image preprocessing logic in isolation (resize, grayscale, normalization, background inversion) |
| Backend integration tests | [backend/tests/test_api.py](backend/tests/test_api.py) | The real FastAPI app via `TestClient` — health check, valid/invalid prediction requests (covers the ">= 2 integration tests" requirement on its own) |
| Frontend unit tests | [frontend/tests/app.test.js](frontend/tests/app.test.js) | Pure JS helpers (API URL resolution, percentage formatting, result table rendering) |
| End-to-end / UI tests | [e2e/test_ui_flow.py](e2e/test_ui_flow.py) | Playwright driving a real browser against the full running stack — uploads an image through the actual page and checks the rendered result |

```bash
# backend unit + integration tests
cd backend && pip install -r requirements-dev.txt && pytest

# frontend unit tests
cd frontend && node --test tests/

# end-to-end tests (stack must be running via deploy.sh first)
cd e2e && pip install -r requirements.txt && playwright install && pytest
```

## CI/CD

**Continuous Integration** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on every push and pull request to `main`, as three jobs:

1. `backend` — installs dependencies, lints with `ruff check .`, runs the pytest suite.
2. `frontend` — runs the Node unit tests (`node --test`).
3. `e2e` — gated on the first two passing; builds and starts the real Docker stack (`docker compose up --build -d`), waits for `/health`, runs the Playwright suite against it, then tears the stack down (`if: always()`, so it cleans up even on failure).

**Continuous Delivery** is the local one-command deployment via [deploy.sh](deploy.sh) / `docker-compose.yml` described above — anyone cloning this repo can bring up the full stack with a single command, no manual setup beyond having Docker installed. Cloud deployment was not pursued for this milestone.

## Project structure

```
backend/
  main.py                     FastAPI app: model loading, preprocessing, /health and /predict
  Dockerfile                  backend container image
  resnet18_mnist_baseline.pt  trained model weights (tracked via Git LFS)
  tests/                      unit + integration tests
frontend/
  index.html                  UI, canvas drawing, result rendering
  app.js                      pure helper functions (URL resolution, formatting, HTML building)
  Dockerfile                  frontend container image (nginx)
  tests/                      unit tests for app.js
e2e/
  test_ui_flow.py             Playwright end-to-end tests
.github/workflows/ci.yml      GitHub Actions CI pipeline
docker-compose.yml            two-service orchestration (backend + frontend)
deploy.sh                     build, start, and health-check the stack
```
