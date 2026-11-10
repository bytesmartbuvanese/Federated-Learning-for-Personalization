# Federated Learning for Personalization

A federated learning system for educational personalization that predicts course-completion success across multiple independent institutions without ever sharing raw student-level data. Local models are trained on each institution's siloed dataset and aggregated using **FedAvg (Federated Averaging)** to produce a single global model.

> ⚠️ **Research artifact.** All data is synthetic and fictional. The model is trained on generated institutional data and should be validated on real institution data before any operational use.


## Project Overview

| Item | Details |
|------|---------|
| **Problem** | Predict whether a student will successfully complete a course (`successful_completion`) using only features available at prediction time, across federated institution silos. |
| **Approach** | Federated Averaging (FedAvg) — local model updates from each institution are weighted by sample count and aggregated into a global model; raw records never leave their institution folder. |
| **Institutions** | 3 independent silos (`institution_A`, `institution_B`, `institution_C`), 500 students each (1,500 total). |
| **Target** | Binary: `student_academic_outcomes.successful_completion` (1 = passed/completed, 0 = failed/withdrawn). |
| **Features** | 18 engineered features spanning student history, course context, assessment activity, and topic practice (see [Feature Schema](#feature-schema)). |
| **Models compared** | MLP (FedAvg), Logistic Regression (FedAvg), Random Forest (local), XGBoost (local). |
| **API** | FastAPI service serving completion predictions, student-performance analysis, and topic/resource recommendations. |
| **Python** | >=3.12 (managed with `uv`). |

## Repository Layout

```
federated_learning_for_personalization/
├── README.md
├── pyproject.toml              # uv project metadata & entry point
├── .python-version             # Python 3.12
├── uv.lock                     # Locked dependencies
├── generate_federated_dataset.py   # Synthetic dataset generator (21 tables × 3 institutions)
├── compare_train.py            # Federated vs. local model comparison script
├── src/
│   └── federated_learning_for_personalization/
│       └── __init__.py         # Package entry point
├── model_and_api/
│   ├── train_federated.py      # Core federated training (CompletionMLP, FedAvg, metrics)
│   ├── api.py                  # FastAPI inference API
│   ├── requirements.txt        # API runtime dependencies
│   └── artifacts/              # Trained model + metrics
│       ├── global_completion_mlp.pt
│       ├── training_metrics.json
│       └── feature_schema.json
├── federated_learning_dataset/
│   ├── institution_A/          # 21 CSV tables (synthetic University of Technology)
│   ├── institution_B/          # 21 CSV tables (synthetic National University)
│   ├── institution_C/          # 21 CSV tables (synthetic Institute of Higher Education)
│   └── shared_documentation/   # Feature definitions, schema relationships, data dictionary, quality reports
└── artifacts/
    └── comparison/             # Model comparison outputs (CSV, JSON, model checkpoints)
```

## Quick Start

### 1. Install dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) (Python >=3.12).

```bash
cd /home/austin/algorithms/federated_learning_for_personalization
uv sync
```

### 2. (Optional) Regenerate the synthetic dataset

The dataset is already included under `federated_learning_dataset/`. To regenerate it:

```bash
python generate_federated_dataset.py
# or with custom output:
python generate_federated_dataset.py --output ./federated_learning_dataset --students-per-institution 500 --seed 42
```

### 3. Train the federated model

Run FedAvg across the three institution silos (default: 15 rounds, 3 local epochs):

```bash
cd model_and_api
python train_federated.py
```

This trains a `CompletionMLP` (18 → 64 → 32 → 1) locally on each institution, aggregates updates with FedAvg, and saves:

- `model_and_api/artifacts/global_completion_mlp.pt` — global model state
- `model_and_api/artifacts/training_metrics.json` — per-round metrics
- `model_and_api/artifacts/feature_schema.json` — ordered feature list

### 4. Run the model comparison

Compare FedAvg MLP, FedAvg Logistic Regression, Random Forest, and XGBoost:

```bash
python compare_train.py
```

Results are written to `artifacts/comparison/` (`model_comparison.csv`, `comparison_summary.json`, round histories).

### 5. Start the API

```bash
cd model_and_api
uvicorn api:app --reload
```

The API is then available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## API Reference

The FastAPI server exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check; reports whether the model artifact is loaded. |
| `/predict/course-completion` | POST | Predicts course-completion probability from 18 input features. |
| `/predict/student-performance` | POST | Full student-performance analysis: subject breakdown, weak topics, risk level, recommendations. |
| `/recommend/{institution}/{student_id}/{course_id}` | GET | Rule-based weak-topic ranking and learning-resource recommendations. |

### Example: course-completion prediction

```bash
curl -X POST "http://localhost:8000/predict/course-completion" \
  -H "Content-Type: application/json" \
  -d '{
    "current_year": 2,
    "education_level_code": 2,
    "course_credit_hours": 3,
    "course_difficulty_code": 1,
    "term_year": 4,
    "prior_courses_count": 5,
    "prior_success_rate": 0.8,
    "prior_mean_percentage": 75.0,
    "prior_mean_attendance": 85.0,
    "assessment_count": 4,
    "assessment_mean_percentage": 70.0,
    "assessment_submission_rate": 1.0,
    "topic_count": 6,
    "topic_mean_accuracy": 72.0,
    "topic_min_accuracy": 55.0,
    "topic_total_attempts": 40,
    "topic_mean_response_time": 45.0
  }'
```

Response:

```json
{
  "successful_completion_probability": 0.7832,
  "predicted_outcome": "successful",
  "threshold": 0.5
}
```

## Feature Schema

The model uses 18 features, constructed point-in-time per completed course outcome (no same-course final outcomes used as predictors):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `current_year` | Student's current academic year |
| 2 | `education_level_code` | Encoded education level (Foundation=0 … Doctoral=4) |
| 3 | `course_credit_hours` | Course credit hours |
| 4 | `course_difficulty_code` | Encoded difficulty (Beginner=0, Intermediate=1, Advanced=2) |
| 5 | `term_year` | Academic term year (centered at 2020) |
| 6 | `prior_courses_count` | Count of previously completed courses |
| 7 | `prior_success_rate` | Fraction of prior courses successfully completed |
| 8 | `prior_mean_percentage` | Mean final percentage across prior courses |
| 9 | `prior_mean_attendance` | Mean attendance across prior courses |
| 10 | `assessment_count` | Number of assessments submitted before cutoff |
| 11 | `assessment_mean_percentage` | Mean assessment percentage |
| 12 | `assessment_submission_rate` | Fraction of assessments submitted |
| 13 | `topic_count` | Number of distinct topics practiced |
| 14 | `topic_mean_accuracy` | Mean topic practice accuracy |
| 15 | `topic_min_accuracy` | Minimum topic practice accuracy |
| 16 | `topic_total_attempts` | Total topic practice attempts |
| 17 | `topic_mean_response_time` | Mean response time in seconds |

Full ordering is stored in `model_and_api/artifacts/feature_schema.json` and `federated_learning_dataset/shared_documentation/feature_definitions.md`.

## Training Results (synthetic data, seed 42)

After 15 federated rounds across 3 institutions (500 students each):

| Model | Accuracy | Precision | Recall | F1 | Brier Score | ROC-AUC |
|-------|----------|-----------|--------|----|-------------|---------|
| MLP_FedAvg | 0.8136 | 0.8242 | 0.7751 | 0.7989 | 0.1275 | 0.9024 |
| Logistic_Regression_FedAvg | 0.8072 | 0.8408 | 0.7358 | 0.7848 | 0.1321 | 0.8981 |
| Random_Forest_Local | 0.8169 | 0.8238 | 0.7846 | 0.8037 | 0.1243 | 0.9060 |
| XGBoost_Local | 0.8177 | 0.8220 | 0.7894 | 0.8053 | 0.1248 | 0.9064 |

Per-round MLP training metrics are recorded in `model_and_api/artifacts/training_metrics.json`. ROC-AUC climbs from ~0.890 at round 1 to ~0.902 at round 15.

## Data

### Dataset generation

`generate_federated_dataset.py` produces 3 independent institutional silos. Each institution contains exactly 21 relational tables (63 CSV files total):

```
institutions, departments, programs, academic_terms,
students, student_enrollments, courses, course_topics,
assessments, student_assessment_results, student_course_history,
learning_sessions, learning_activity_logs, student_topic_performance,
student_academic_outcomes, learning_resources, student_resource_interactions,
federated_clients, federated_training_rounds, federated_client_updates,
personalized_recommendations
```

Key characteristics:
- 500 students per institution, 1,500 total
- Random seed 42; all people, institutions, and records are fictional
- Each institution uses an independent random-number stream
- No central student-level dataset is ever exported — each folder is a local silo
- Validation passed with 0 failures (see `validation_failures.csv`)

### Federated data distribution

Each institution directory is an independent local silo. Federated round IDs are shared across institutions; each local folder records only its own client update metadata. No raw student records are exported to a shared directory.

See `federated_learning_dataset/shared_documentation/` for:
- `feature_definitions.md` — how point-in-time training examples are built
- `schema_relationships.md` — all foreign-key relationships across the 21 tables
- `data_dictionary.csv` — column-level data dictionary
- `data_quality_report.csv` — per-institution quality metrics
- `validation_failures.csv` — validation failures (none)
- `synthetic_data_generation_report.md` — generation methodology
- `federated_data_distribution_report.md` — distribution summary
- `target_variable_documentation.md` — target-variable definitions and caveats
- `generation_summary.json` — row counts and generation parameters

## How It Works

### Federated training (`train_federated.py`)

1. **Build client frames** — For each institution, `build_client_frame()` joins 9 source tables into one point-in-time row per completed course outcome, computing the 18 features from prior history only (no same-course outcome leakage).
2. **Train/test split** — `GroupShuffleSplit` splits by `student_id` (80/20), ensuring a student's outcomes do not appear in both train and test.
3. **Local training** — Each institution trains a `CompletionMLP` locally for `N` epochs on its own training split (Adam optimizer, BCEWithLogitsLoss, gradient clipping).
4. **FedAvg aggregation** — Client state dictionaries are weighted by local sample count and averaged into a new global state.
5. **Evaluation** — After each round, the global model is evaluated on the combined test sets of all institutions (accuracy, precision, recall, F1, Brier score, ROC-AUC).
6. **Artifact save** — The final global model, feature schema, and round history are saved to `model_and_api/artifacts/`.

### Inference API (`api.py`)

- `/predict/course-completion` accepts the 18 features directly and returns a completion probability with a 0.5 threshold.
- `/predict/student-performance` looks up a student by ID or reference within an institution, builds their feature frame, runs the global model across their courses, and returns subject-level analysis, weak topics, academic risk, and recommendations.
- `/recommend/{institution}/{student_id}/{course_id}` performs transparent weak-topic ranking (topics with accuracy < 70% and ≥ 3 attempts) and heuristic resource ranking using interaction completion/feedback as tie-breakers.

### Model comparison (`compare_train.py`)

Runs four model families on the same federated client splits and writes side-by-side results:
1. MLP with FedAvg
2. Logistic Regression with FedAvg
3. Random Forest (trained locally on pooled client data)
4. XGBoost (trained locally on pooled client data)

## Requirements

| Package | Minimum version |
|---------|-----------------|
| fastapi | >=0.110 |
| numpy | >=1.24 |
| pandas | >=2.0 |
| pydantic | >=2.0 |
| scikit-learn | >=1.3 |
| torch | >=2.1 |
| uvicorn[standard] | >=0.27 |
| xgboost | >=3.4.1 |

Python >=3.12 is required. The project is configured with `uv` (`pyproject.toml`, `uv.lock`).

## Caveats & Limitations

- **Synthetic data only.** All students, institutions, courses, and records are fictional. Do not deploy on real student data without re-validation.
- **Single global model.** The federated simulation runs locally in one process; real federated deployment would require secure client-server communication, encryption, and differential privacy considerations.
- **No cross-client scaler.** Features are clipped to fixed ranges but not standardized across clients; each institution's data distribution differs.
- **Target leakage guard.** Same-course final outcomes (`final_percentage`, `final_grade`, `completion_status`, `outcome_date`) are excluded from predictors; only prior history up to the outcome date is used.
- **Recommendations are heuristic.** Topic/resource recommendations use transparent rules (accuracy < 70%, ≥ 3 attempts), not a trained relevance model. Evaluate with held-out relevance labels before production use.

## License

All data and code in this repository are synthetic research artifacts.

## Author

bytesmartbuvanese — bytesmart.buvanese@gmail.com