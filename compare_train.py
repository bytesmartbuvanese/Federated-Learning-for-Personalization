
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from model_and_api.train_federated import (
    FEATURES,
    INSTITUTIONS,
    CompletionMLP,
    build_client_frame,
    train_local,
    fedavg,
    predict,
    metrics,
    seed_everything,
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --------------------------------------------------
# LOGISTIC REGRESSION MODEL
# --------------------------------------------------

class CompletionLogistic(nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# --------------------------------------------------
# LOCAL FEDERATED LOGISTIC TRAINING
# --------------------------------------------------

def train_local_logistic(
    global_state,
    X,
    y,
    indices,
    epochs=3,
    batch_size=256,
    lr=1e-3,
    device="cpu",
):

    model = CompletionLogistic(X.shape[1]).to(device)
    model.load_state_dict(global_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(
        torch.tensor(X[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()

    for _ in range(epochs):

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            loss = loss_fn(model(xb), yb)

            loss.backward()
            optimizer.step()

    state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    return state, len(indices)


@torch.no_grad()
def predict_logistic(model, X, device="cpu"):

    model.eval()

    x = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    return torch.sigmoid(model(x)).cpu().numpy()


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_predictions(y_true, predictions):

    return metrics(
        np.asarray(y_true),
        np.asarray(predictions),
    )


def evaluate_global_model(model, clients, device, logistic=False):

    all_y = []
    all_predictions = []

    for client in clients.values():

        X_test = client["X"][client["test"]]
        y_test = client["y"][client["test"]]

        if logistic:
            probabilities = predict_logistic(
                model, X_test, device
            )
        else:
            probabilities = predict(
                model, X_test, device
            )

        all_y.extend(y_test)
        all_predictions.extend(probabilities)

    return evaluate_predictions(
        all_y,
        all_predictions,
    )


# --------------------------------------------------
# FEDERATED MLP
# --------------------------------------------------

def run_federated_mlp(
    clients,
    rounds,
    local_epochs,
    device,
):

    print("\nTraining MLP using FedAvg...")

    model = CompletionMLP(len(FEATURES)).to(device)

    global_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    history = []

    for rnd in range(1, rounds + 1):

        client_states = []
        client_sizes = []

        for name, client in clients.items():

            state, size = train_local(
                global_state,
                client["X"],
                client["y"],
                client["train"],
                epochs=local_epochs,
                device=device,
            )

            client_states.append(state)
            client_sizes.append(size)

        global_state = fedavg(
            client_states,
            client_sizes,
        )

        model.load_state_dict(global_state)

        result = evaluate_global_model(
            model,
            clients,
            device,
        )

        history.append({
            "round": rnd,
            **result,
        })

        print(
            f"MLP Round {rnd:02d} | "
            f"Accuracy={result['accuracy']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f}"
        )

    return model, global_state, history


# --------------------------------------------------
# FEDERATED LOGISTIC REGRESSION
# --------------------------------------------------

def run_federated_logistic(
    clients,
    rounds,
    local_epochs,
    device,
):

    print("\nTraining Logistic Regression using FedAvg...")

    model = CompletionLogistic(
        len(FEATURES)
    ).to(device)

    global_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    history = []

    for rnd in range(1, rounds + 1):

        client_states = []
        client_sizes = []

        for name, client in clients.items():

            state, size = train_local_logistic(
                global_state,
                client["X"],
                client["y"],
                client["train"],
                epochs=local_epochs,
                device=device,
            )

            client_states.append(state)
            client_sizes.append(size)

        global_state = fedavg(
            client_states,
            client_sizes,
        )

        model.load_state_dict(global_state)

        result = evaluate_global_model(
            model,
            clients,
            device,
            logistic=True,
        )

        history.append({
            "round": rnd,
            **result,
        })

        print(
            f"Logistic Round {rnd:02d} | "
            f"Accuracy={result['accuracy']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f}"
        )

    return model, global_state, history


# --------------------------------------------------
# RANDOM FOREST LOCAL BASELINE
# --------------------------------------------------

def run_random_forest(clients):

    print("\nTraining Random Forest at each institution...")

    all_y = []
    all_predictions = []

    for name, client in clients.items():

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        X_train = client["X"][client["train"]]
        y_train = client["y"][client["train"]]

        X_test = client["X"][client["test"]]
        y_test = client["y"][client["test"]]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        all_y.extend(y_test)
        all_predictions.extend(probabilities)

        print(f"{name}: Random Forest training completed.")

    return evaluate_predictions(
        all_y,
        all_predictions,
    )


# --------------------------------------------------
# XGBOOST LOCAL BASELINE
# --------------------------------------------------

def run_xgboost(clients):

    if not XGBOOST_AVAILABLE:

        print(
            "\nXGBoost is not installed. "
            "Skipping XGBoost comparison."
        )

        return None

    print("\nTraining XGBoost at each institution...")

    all_y = []
    all_predictions = []

    for name, client in clients.items():

        model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        X_train = client["X"][client["train"]]
        y_train = client["y"][client["train"]]

        X_test = client["X"][client["test"]]
        y_test = client["y"][client["test"]]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        all_y.extend(y_test)
        all_predictions.extend(probabilities)

        print(f"{name}: XGBoost training completed.")

    return evaluate_predictions(
        all_y,
        all_predictions,
    )


# --------------------------------------------------
# SAVE GLOBAL MODEL ARTIFACTS
# --------------------------------------------------

def save_model(model_state, model_name, output_dir):

    artifact = {
        "feature_names": FEATURES,
        "model_state": model_state,
        "target": "successful_completion",
        "institution_count": len(INSTITUTIONS),
    }

    torch.save(
        artifact,
        output_dir / model_name,
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="federated_learning_dataset",
    )

    parser.add_argument(
        "--output",
        default="artifacts/comparison",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_root = Path(args.data)

    if not data_root.exists():

        data_root = (
            Path(__file__).resolve().parent.parent
            / args.data
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------
    # LOAD DATA AND CREATE IDENTICAL SPLITS
    # ----------------------------------------------

    print("\nLoading institutional datasets...")

    clients = {}

    for institution in INSTITUTIONS:

        folder = data_root / institution

        if not folder.exists():
            raise FileNotFoundError(folder)

        X, y, groups, frame = build_client_frame(folder)

        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{institution} has only one target class."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=args.seed,
        )

        train_idx, test_idx = next(
            splitter.split(X, y, groups)
        )

        clients[institution] = {
            "X": X,
            "y": y,
            "train": train_idx,
            "test": test_idx,
        }

        print(
            f"{institution}: "
            f"rows={len(y):,}, "
            f"train={len(train_idx):,}, "
            f"test={len(test_idx):,}"
        )

    # ----------------------------------------------
    # RUN MODEL COMPARISONS
    # ----------------------------------------------

    results = {}

    # 1. Federated MLP
    mlp, mlp_state, mlp_history = run_federated_mlp(
        clients,
        args.rounds,
        args.local_epochs,
        device,
    )

    results["MLP_FedAvg"] = mlp_history[-1]

    save_model(
        mlp_state,
        "global_completion_mlp_comparison.pt",
        output_dir,
    )

    # 2. Federated Logistic Regression
    logistic, logistic_state, logistic_history = (
        run_federated_logistic(
            clients,
            args.rounds,
            args.local_epochs,
            device,
        )
    )

    results["Logistic_Regression_FedAvg"] = (
        logistic_history[-1]
    )

    save_model(
        logistic_state,
        "global_completion_logistic_fedavg.pt",
        output_dir,
    )

    # 3. Random Forest
    results["Random_Forest_Local"] = run_random_forest(
        clients
    )

    # 4. XGBoost
    xgb_result = run_xgboost(clients)

    if xgb_result is not None:
        results["XGBoost_Local"] = xgb_result

    # ----------------------------------------------
    # SAVE COMPARISON RESULTS
    # ----------------------------------------------

    rows = []

    for model_name, result in results.items():

        rows.append({
            "model": model_name,
            **result,
        })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(results, indent=4)
    )

    pd.DataFrame(mlp_history).to_csv(
        output_dir / "mlp_round_history.csv",
        index=False,
    )

    pd.DataFrame(logistic_history).to_csv(
        output_dir / "logistic_round_history.csv",
        index=False,
    )

    # ----------------------------------------------
    # DISPLAY FINAL RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 85)
    print("FINAL MODEL COMPARISON")
    print("=" * 85)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nResults saved to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
