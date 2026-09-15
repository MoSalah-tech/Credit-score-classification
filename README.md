# Credit Score Classification 🏦

Multi-class classification pipeline predicting customer credit score bands
(**Good / Standard / Poor**) using XGBoost and LightGBM, tuned with Optuna,
and tracked with MLflow.

## Project Structure

```
credit_score_project/
├── configs/
│   └── config.yaml           # All settings in one place
├── data/
│   └── train.csv             # Download from Kaggle (not committed)
├── src/
│   ├── data/
│   │   ├── loader.py         # Load raw CSV
│   │   └── cleaner.py        # Fix dirty values, impute nulls
│   ├── features/
│   │   ├── encoder.py        # Label encode target + categoricals
│   │   └── splitter.py       # Stratified train/test split
│   ├── models/
│   │   ├── tuner.py          # Optuna objectives for both models
│   │   ├── xgboost_model.py  # XGBoost build + train
│   │   └── lightgbm_model.py # LightGBM build + train
│   ├── evaluation/
│   │   ├── metrics.py        # Accuracy, F1, per-class F1
│   │   └── visualizer.py     # Confusion matrix plots
│   └── utils/
│       ├── logger.py         # Centralised logging
│       └── mlflow_utils.py   # MLflow helpers
├── tests/
│   ├── test_cleaner.py
│   ├── test_encoder.py
│   └── test_models.py
├── outputs/
│   ├── plots/                # Confusion matrix PNGs
│   └── reports/
├── mlflow_runs/              # Local MLflow tracking store
├── main.py                   # ← Run this
├── requirements.txt
└── .gitignore
```

## Dataset :

- **Source:** [Kaggle — Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data- **Source:** 
- **Size:** ~100,000 rows, 28 columns
- **Target:** `Credit_Score` (Good / Standard / Poor)
- **License:** Not specified on Kaggle (check for commercial use)



## XGBoost vs LightGBM

| | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Level-wise | Leaf-wise |
| Speed | Slower | Much faster |
| Overfit risk | Lower | Higher (needs regularization) |
| Extra key param | `max_depth` | `num_leaves` |
| Best for | Safer baseline | Large data (like this 100K dataset) |

## MLflow Tracking

Each run logs:
- ✅ Best hyperparameters (from Optuna)
- ✅ Accuracy + Weighted F1 + per-class F1
- ✅ Confusion matrix PNG as artifact
- ✅ Trained model artifact (loadable with `mlflow.xgboost.load_model`)



## 🚀 How to run :
```shell
pip install -r requirements.txt
# drop train.csv into data/
python main.py --trials 30

# view MLflow
mlflow ui --backend-store-uri mlflow_runs

# run tests
pytest tests/
```

## 📊 Results
```text
Model	Accuracy	F1 Weighted
XGBoost	0.7971	0.7967
LightGBM	0.7961	0.7959
```
XGBoost wins — but honestly it's basically a tie, only 0.001 difference.

### Per-class breakdown (XGBoost)
```text
Class	F1
Standard	0.814 — easiest, most data
Poor	0.798 — solid
Good	0.744 — hardest, least data (class imbalance)
```
![Confusion Matrix XGBoost](outputs/plots/confusion_matrix_xgboost.png)

![Confusion Matrix LightGBM](outputs/plots/confusion_matrix_lightgbm.png)



---

## CI/CD with Jenkins

The project includes a Jenkins pipeline for continuous integration. Every push to `main` triggers an automated build that checks out the code, sets up the Python environment, and runs the test suite.

### Pipeline Stages

| Stage | What it does |
| :--- | :--- |
| **Checkout** | Clones the repo and prints the last commit |
| **Setup Environment** | Creates a virtualenv and installs `requirements.txt` |
| **Lint & Test** | Runs `pytest` against the `tests/` directory |

Model training is **not** run in CI. The dataset (~100 MB) isn't committed to the repo, and training on a CPU-only Jenkins container would take too long for a fast feedback loop. Training is a separate, manual step.

### Jenkins Setup

Jenkins runs in Docker using a custom image that includes Python 3, Git, and the build tools needed to install scientific packages.

**`jenkins/Dockerfile`:**

```dockerfile
FROM jenkins/jenkins:lts

USER root
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*
USER jenkins
```

Run the container:

```shell
docker build -t jenkins-custom:latest jenkins/

docker run -d --name jenkins `
  -p 8080:8080 `
  -p 50000:50000 `
  -v jenkins_home:/var/jenkins_home `
  jenkins-custom:latest

```

#### Jenkins is reachable at http://localhost:8080. The jenkins_home volume persists jobs, plugins, and configuration across container rebuilds.

### Jenkinsfile
The pipeline is a scripted Jenkinsfile at jenkins/Jenkinsfile:

Trigger
The pipeline uses Poll SCM with a 5-minute interval (H/5 * * * *). This avoids the need for GitHub webhooks, which require Jenkins to be publicly reachable — not the case when running locally.

To register the trigger, run Build Now once manually in the Jenkins UI. After that, Jenkins checks GitHub every ~5 minutes and starts a build when it detects a new commit.



## A few notes on the section

**Why I wrote it this way:**

- **Honest about limitations** — "no training in CI" is stated up front, not hidden. A reviewer will trust the project more.
- **Design decisions section** — explains *why* scripted over declarative, *why* no Docker-in-Docker. This is what separates a portfolio project from a tutorial follow-along.
- **Future improvements** — shows you know the "correct" production path even if you haven't taken it yet.
- **Working commands** — someone can copy-paste and reproduce your setup.

