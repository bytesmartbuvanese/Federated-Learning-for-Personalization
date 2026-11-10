from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

SEED = 42
FEATURES = [
    "current_year", "education_level_code", "course_credit_hours",
    "course_difficulty_code", "term_year",
    "prior_courses_count", "prior_success_rate", "prior_mean_percentage",
    "prior_mean_attendance",
    "assessment_count", "assessment_mean_percentage", "assessment_submission_rate",
    "topic_count", "topic_mean_accuracy", "topic_min_accuracy",
    "topic_total_attempts", "topic_mean_response_time",
]
INSTITUTIONS = ["institution_A", "institution_B", "institution_C"]

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_csv(folder, name):
    p = folder / f"{name}.csv"
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p, low_memory=False)

def safe_numeric(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def build_client_frame(folder: Path):
    """Build one point-in-time row per completed course outcome; no same-course final outcomes as predictors."""
    outcomes = read_csv(folder, "student_academic_outcomes")
    students = read_csv(folder, "students")
    courses = read_csv(folder, "courses")
    terms = read_csv(folder, "academic_terms")
    history = read_csv(folder, "student_course_history")
    assessments = read_csv(folder, "assessments")
    results = read_csv(folder, "student_assessment_results")
    topics = read_csv(folder, "student_topic_performance")

    outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"], errors="coerce")
    outcomes["successful_completion"] = outcomes["successful_completion"].astype(str).str.lower().map(
        {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
    )
    outcomes = outcomes.dropna(subset=["student_id", "course_id", "term_id", "outcome_date", "successful_completion"]).copy()
    outcomes["successful_completion"] = outcomes["successful_completion"].astype(int)
    outcomes = outcomes.drop_duplicates("outcome_id")

    # Static attributes available at prediction time.
    base = outcomes[["outcome_id", "student_id", "course_id", "term_id", "outcome_date", "successful_completion"]].copy()
    s = students[["student_id", "current_year", "education_level"]].drop_duplicates("student_id")
    c = courses[["course_id", "credit_hours", "difficulty_level"]].drop_duplicates("course_id")
    t = terms[["term_id", "academic_year"]].drop_duplicates("term_id")
    base = base.merge(s, on="student_id", how="left").merge(c, on="course_id", how="left").merge(t, on="term_id", how="left")
    base["current_year"] = safe_numeric(base["current_year"])
    base["education_level_code"] = base["education_level"].map({"Foundation": 0, "Diploma": 1, "Undergraduate": 2, "Postgraduate": 3, "Doctoral": 4}).fillna(2)
    base["course_credit_hours"] = safe_numeric(base["credit_hours"])
    base["course_difficulty_code"] = base["difficulty_level"].map({"Beginner": 0, "Intermediate": 1, "Advanced": 2}).fillna(1)
    base["term_year"] = pd.to_numeric(base["academic_year"].astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce").fillna(2024)

    # Historical courses: strictly earlier than the target course outcome and never same course.
    h = history.copy()
    h["completion_date"] = pd.to_datetime(h["completion_date"], errors="coerce")
    h = h.merge(base[["outcome_id", "student_id", "course_id", "outcome_date"]], on="student_id", how="inner", suffixes=("", "_target"))
    h = h[(h["course_id"] != h["course_id_target"]) & h["completion_date"].notna() & (h["completion_date"] < h["outcome_date"])]
    h["success_hist"] = h["completion_status"].astype(str).str.lower().isin(["completed", "passed", "successful"]).astype(float)
    hist_agg = h.groupby("outcome_id").agg(
        prior_courses_count=("history_id", "count"),
        prior_success_rate=("success_hist", "mean"),
        prior_mean_percentage=("final_percentage", "mean"),
        prior_mean_attendance=("attendance_percentage", "mean"),
    ).reset_index()
    base = base.merge(hist_agg, on="outcome_id", how="left")

    # Assessment signals observed before the course outcome date.
    a = assessments[["assessment_id", "course_id", "assessment_date"]].copy()
    a["assessment_date"] = pd.to_datetime(a["assessment_date"], errors="coerce")
    r = results[["student_id", "assessment_id", "percentage", "submission_status", "submission_date"]].copy()
    r["submission_date"] = pd.to_datetime(r["submission_date"], errors="coerce")
    ar = r.merge(a, on="assessment_id", how="inner")
    ar = ar.merge(base[["outcome_id", "student_id", "course_id", "outcome_date"]], on=["student_id", "course_id"], how="inner")
    ar = ar[ar["submission_date"].notna() & (ar["submission_date"] < ar["outcome_date"])]
    ar["submitted_flag"] = ar["submission_status"].astype(str).str.lower().isin(["submitted", "completed", "late"]).astype(float)
    aa = ar.groupby("outcome_id").agg(
        assessment_count=("assessment_id", "count"),
        assessment_mean_percentage=("percentage", "mean"),
        assessment_submission_rate=("submitted_flag", "mean"),
    ).reset_index()
    base = base.merge(aa, on="outcome_id", how="left")

    # Topic practice signals strictly before outcome date.
    tp = topics.copy()
    tp["last_practiced_at"] = pd.to_datetime(tp["last_practiced_at"], errors="coerce")
    tp = tp.merge(base[["outcome_id", "student_id", "course_id", "outcome_date"]], on=["student_id", "course_id"], how="inner")
    tp = tp[tp["last_practiced_at"].notna() & (tp["last_practiced_at"] < tp["outcome_date"])]
    ta = tp.groupby("outcome_id").agg(
        topic_count=("topic_id", "nunique"),
        topic_mean_accuracy=("accuracy_percentage", "mean"),
        topic_min_accuracy=("accuracy_percentage", "min"),
        topic_total_attempts=("practice_attempts", "sum"),
        topic_mean_response_time=("average_response_time", "mean"),
    ).reset_index()
    base = base.merge(ta, on="outcome_id", how="left")

    for col in FEATURES:
        if col not in base: base[col] = 0.0
        base[col] = safe_numeric(base[col])
    # Keep features within reasonable numerical ranges; no cross-client fitted scaler is required.
    base["current_year"] = base["current_year"].clip(0, 10)
    base["course_credit_hours"] = base["course_credit_hours"].clip(0, 12)
    base["term_year"] = (base["term_year"] - 2020).clip(0, 15)
    base["prior_courses_count"] = base["prior_courses_count"].clip(0, 30)
    base["prior_success_rate"] = base["prior_success_rate"].clip(0, 1)
    base["prior_mean_percentage"] = base["prior_mean_percentage"].clip(0, 100)
    base["prior_mean_attendance"] = base["prior_mean_attendance"].clip(0, 100)
    base["assessment_count"] = base["assessment_count"].clip(0, 50)
    base["assessment_mean_percentage"] = base["assessment_mean_percentage"].clip(0, 100)
    base["assessment_submission_rate"] = base["assessment_submission_rate"].clip(0, 1)
    base["topic_count"] = base["topic_count"].clip(0, 100)
    base["topic_mean_accuracy"] = base["topic_mean_accuracy"].clip(0, 100)
    base["topic_min_accuracy"] = base["topic_min_accuracy"].clip(0, 100)
    base["topic_total_attempts"] = base["topic_total_attempts"].clip(0, 500)
    base["topic_mean_response_time"] = base["topic_mean_response_time"].clip(0, 3600)
    X = base[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32").to_numpy()
    y = base["successful_completion"].astype("float32").to_numpy()
    groups = base["student_id"].astype(str).to_numpy()
    return X, y, groups, base

class CompletionMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

def train_local(global_state, X, y, indices, epochs=3, batch_size=256, lr=1e-3, device="cpu"):
    model = CompletionMLP(X.shape[1]).to(device)
    model.load_state_dict(global_state)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, len(indices)

def fedavg(client_states, client_sizes):
    total = float(sum(client_sizes))
    out = {}
    for key in client_states[0]:
        out[key] = sum(state[key] * (n / total) for state, n in zip(client_states, client_sizes))
    return out

@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32).to(device))
    return torch.sigmoid(logits).cpu().numpy()

def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
    }
    result["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="federated_learning_dataset")
    ap.add_argument("--output", default="artifacts")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed_everything(args.seed)
    # Resolve the data directory relative to this script's location so the
    # default path works no matter which directory the command is run from.
    data_root = Path(args.data)
    if not data_root.exists():
        data_root = Path(__file__).resolve().parent.parent / args.data
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clients = {}
    for inst in INSTITUTIONS:
        folder = data_root / inst
        if not folder.exists(): raise FileNotFoundError(f"Missing institution folder: {folder}")
        X, y, groups, frame = build_client_frame(folder)
        if len(np.unique(y)) < 2: raise ValueError(f"{inst} has only one target class.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        clients[inst] = {"X": X, "y": y, "train": train_idx, "test": test_idx}
        print(f"{inst}: rows={len(y):,}, train={len(train_idx):,}, test={len(test_idx):,}, students={len(set(groups)):,}")
    model = CompletionMLP(len(FEATURES)).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    round_history = []
    for rnd in range(1, args.rounds + 1):
        states, sizes = [], []
        for inst, c in clients.items():
            state, size = train_local(global_state, c["X"], c["y"], c["train"], epochs=args.local_epochs, device=device)
            states.append(state); sizes.append(size)
        global_state = fedavg(states, sizes)
        model.load_state_dict(global_state)
        # Aggregate only evaluation predictions/labels in this local simulation.
        ys, ps = [], []
        for c in clients.values():
            ys.append(c["y"][c["test"]])
            ps.append(predict(model, c["X"][c["test"]], device))
        m = metrics(np.concatenate(ys), np.concatenate(ps))
        round_history.append({"round": rnd, **m})
        print(f"Round {rnd:02d}: " + " | ".join(f"{k}={v:.4f}" for k,v in m.items() if v is not None))
    artifact = {
        "feature_names": FEATURES,
        "model_state": {k: v.cpu() for k, v in global_state.items()},
        "architecture": {"input_dim": len(FEATURES), "hidden": [64, 32]},
        "target": "successful_completion",
        "institution_count": len(clients),
        "rounds": args.rounds,
        "warning": "Synthetic-data research artifact; validate on real institution data before operational use."
    }
    torch.save(artifact, out_dir / "global_completion_mlp.pt")
    (out_dir / "training_metrics.json").write_text(json.dumps(round_history, indent=2))
    (out_dir / "feature_schema.json").write_text(json.dumps({"features": FEATURES}, indent=2))
    print(f"\nSaved model and reports to: {out_dir.resolve()}")

if __name__ == "__main__": main()
