from pathlib import Path
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT = BASE_DIR / "artifacts" / "global_completion_mlp.pt"
DATA_ROOT = BASE_DIR.parent / "federated_learning_dataset"

if __package__:
    from .train_federated import FEATURES, build_client_frame
else:
    from train_federated import FEATURES, build_client_frame


class CompletionMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(1)

app = FastAPI(title="Federated Learning for Personalization API", version="1.0.0")
model = None

@app.on_event("startup")
def load_model():
    global model
    if ARTIFACT.exists():
        artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
        model = CompletionMLP(artifact["architecture"]["input_dim"])
        model.load_state_dict(artifact["model_state"])
        model.eval()

class CompletionInput(BaseModel):
    current_year: float = 1
    education_level_code: float = 2
    course_credit_hours: float = 3
    course_difficulty_code: float = 1
    term_year: float = 4
    prior_courses_count: float = 0
    prior_success_rate: float = 0
    prior_mean_percentage: float = 0
    prior_mean_attendance: float = 0
    assessment_count: float = 0
    assessment_mean_percentage: float = 0
    assessment_submission_rate: float = 0
    topic_count: float = 0
    topic_mean_accuracy: float = 0
    topic_min_accuracy: float = 0
    topic_total_attempts: float = 0
    topic_mean_response_time: float = 0


class StudentPerformanceInput(BaseModel):
    student_id: str = Field(min_length=1)
    institution_id: str = Field(pattern=r"^institution_[ABC]$")


class SubjectAnalysis(BaseModel):
    subject: str
    predicted_score: int = Field(ge=0, le=100)
    weak_topics: list[str]
    priority: str


class StudentPerformanceOutput(BaseModel):
    student_id: str
    institution_id: str
    predicted_performance: str
    performance_probability: float = Field(ge=0, le=1)
    academic_risk: str
    subject_analysis: list[SubjectAnalysis]
    recommendations: list[str]
    overall_recommendation: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict/course-completion")
def predict_completion(payload: CompletionInput):
    if model is None: raise HTTPException(503, "Model artifact not found. Run train_federated.py first.")
    values = payload.model_dump()
    x = torch.tensor([[float(values[k]) for k in FEATURES]], dtype=torch.float32)
    with torch.no_grad(): probability = float(torch.sigmoid(model(x)).item())
    return {
        "successful_completion_probability": round(probability, 4),
        "predicted_outcome": "successful" if probability >= 0.5 else "at_risk",
        "threshold": 0.5,
    }


@app.post(
    "/predict/student-performance",
    response_model=StudentPerformanceOutput,
)
def predict_student_performance(payload: StudentPerformanceInput):
    if model is None:
        raise HTTPException(503, "Model artifact not found. Run train_federated.py first.")

    folder = DATA_ROOT / payload.institution_id
    try:
        students = pd.read_csv(folder / "students.csv", low_memory=False)
        courses = pd.read_csv(folder / "courses.csv", low_memory=False)
        outcomes = pd.read_csv(folder / "student_academic_outcomes.csv", low_memory=False)
        topic_performance = pd.read_csv(
            folder / "student_topic_performance.csv", low_memory=False
        )
        topics = pd.read_csv(folder / "course_topics.csv", low_memory=False)
        features, _, _, feature_frame = build_client_frame(folder)
    except FileNotFoundError as exc:
        raise HTTPException(500, f"Required institutional data is missing: {exc.filename}") from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(500, f"Could not parse institutional data: {exc}") from exc

    requested_student_id = payload.student_id
    students["student_id"] = students["student_id"].astype(str)
    students["student_reference"] = students["student_reference"].astype(str)
    matching_student = students[
        students["student_id"].eq(requested_student_id)
        | students["student_reference"].eq(requested_student_id)
    ]
    if matching_student.empty:
        alias_number = requested_student_id.removeprefix("STU_")
        if alias_number.isdigit():
            institution_number = {
                "institution_A": 1,
                "institution_B": 2,
                "institution_C": 3,
            }[payload.institution_id]
            dataset_reference = (
                f"INST{institution_number}-STU-{int(alias_number):05d}"
            )
            matching_student = students[
                students["student_reference"].eq(dataset_reference)
            ]
    if matching_student.empty:
        raise HTTPException(404, "Student not found in the selected institution.")
    student_id = str(matching_student.iloc[0]["student_id"])

    student_outcomes = outcomes[
        outcomes["student_id"].astype(str).eq(student_id)
    ].copy()
    student_outcomes["final_percentage"] = pd.to_numeric(
        student_outcomes["final_percentage"], errors="coerce"
    )
    student_outcomes["outcome_date"] = pd.to_datetime(
        student_outcomes["outcome_date"], errors="coerce"
    )
    student_outcomes = student_outcomes.dropna(
        subset=["course_id", "outcome_date", "final_percentage"]
    )
    if student_outcomes.empty:
        raise HTTPException(404, "No academic outcome records found for this student.")

    feature_frame = feature_frame.reset_index(drop=True)
    student_features = feature_frame[
        feature_frame["student_id"].astype(str).eq(student_id)
    ].copy()
    if student_features.empty:
        raise HTTPException(404, "No model features found for this student.")

    with torch.no_grad():
        all_probabilities = torch.sigmoid(
            model(torch.tensor(features, dtype=torch.float32))
        ).cpu().numpy()
    student_features["completion_probability"] = all_probabilities[
        student_features.index.to_numpy()
    ]

    course_subjects = courses[["course_id", "subject_area"]].drop_duplicates("course_id")
    student_outcomes = student_outcomes.merge(
        course_subjects, on="course_id", how="left"
    )
    student_features = student_features.merge(
        course_subjects, on="course_id", how="left"
    )
    student_outcomes = student_outcomes.dropna(subset=["subject_area"])
    student_features = student_features.dropna(subset=["subject_area"])
    if student_outcomes.empty:
        raise HTTPException(404, "No subject information found for this student's courses.")

    latest_outcomes = (
        student_outcomes.sort_values("outcome_date")
        .groupby("course_id", as_index=False)
        .tail(1)
    )
    latest_probabilities = (
        student_features.sort_values("outcome_date")
        .groupby("course_id", as_index=False)
        .tail(1)
    )
    if latest_probabilities.empty:
        raise HTTPException(404, "No completion predictions found for this student.")

    student_topics = topic_performance[
        topic_performance["student_id"].astype(str).eq(student_id)
    ].copy()
    student_topics["accuracy_percentage"] = pd.to_numeric(
        student_topics["accuracy_percentage"], errors="coerce"
    )
    student_topics["practice_attempts"] = pd.to_numeric(
        student_topics["practice_attempts"], errors="coerce"
    ).fillna(0)
    student_topics["last_practiced_at"] = pd.to_datetime(
        student_topics["last_practiced_at"], errors="coerce"
    )
    student_topics = student_topics.merge(
        course_subjects, on="course_id", how="inner"
    ).merge(
        topics[["topic_id", "topic_name"]].drop_duplicates("topic_id"),
        on="topic_id",
        how="left",
    )
    student_topics = (
        student_topics.sort_values("last_practiced_at")
        .groupby(["course_id", "topic_id"], as_index=False)
        .tail(1)
    )

    subject_analysis = []
    weak_topic_names = []
    for subject, subject_outcomes in latest_outcomes.groupby("subject_area"):
        recent_scores = (
            subject_outcomes.sort_values("outcome_date")["final_percentage"]
            .tail(3)
        )
        predicted_score = float(recent_scores.mean())
        subject_topics = student_topics[
            student_topics["subject_area"].eq(subject)
        ].dropna(subset=["accuracy_percentage"])
        weak_topics = (
            subject_topics[subject_topics["accuracy_percentage"] < 70]
            .sort_values("accuracy_percentage")
            .head(3)
        )
        weak_names = weak_topics["topic_name"].dropna().astype(str).tolist()
        weak_topic_names.extend(weak_names)

        if predicted_score < 60 or len(weak_names) >= 2:
            priority = "High"
        elif predicted_score < 75 or weak_names:
            priority = "Medium"
        else:
            priority = "Low"

        subject_analysis.append({
            "subject": str(subject),
            "predicted_score": round(predicted_score),
            "weak_topics": weak_names,
            "priority": priority,
        })

    if not subject_analysis:
        raise HTTPException(404, "No subject-level results found for this student.")

    subject_analysis.sort(key=lambda result: result["predicted_score"])
    overall_score = sum(
        result["predicted_score"] for result in subject_analysis
    ) / len(subject_analysis)
    performance_probability = float(
        latest_probabilities["completion_probability"].mean()
    )

    if overall_score < 70:
        predicted_performance = "Needs Improvement"
    elif overall_score < 80:
        predicted_performance = "Satisfactory"
    elif overall_score < 90:
        predicted_performance = "Good"
    else:
        predicted_performance = "Excellent"

    if overall_score < 60:
        academic_risk = "High"
    elif overall_score < 75:
        academic_risk = "Medium"
    else:
        academic_risk = "Low"

    recommendations = [
        f"Practice {topic} exercises"
        for topic in dict.fromkeys(weak_topic_names)
    ][:5]
    if not recommendations:
        recommendations.append("Continue practicing to maintain current performance")

    weakest_subject = subject_analysis[0]["subject"]
    overall_recommendation = (
        f"Prioritize {weakest_subject} improvement"
        if subject_analysis[0]["priority"] != "Low"
        else "Maintain consistent study habits across subjects"
    )

    return {
        "student_id": requested_student_id,
        "institution_id": payload.institution_id,
        "predicted_performance": predicted_performance,
        "performance_probability": round(performance_probability, 2),
        "academic_risk": academic_risk,
        "subject_analysis": subject_analysis,
        "recommendations": recommendations,
        "overall_recommendation": overall_recommendation,
    }


@app.get("/recommend/{institution}/{student_id}/{course_id}")
def recommend(institution: str, student_id: str, course_id: str, top_k: int = 5):
    """Transparent rule-based topic/resource suggestions; not a trained relevance model."""
    if institution not in {"institution_A","institution_B","institution_C"}:
        raise HTTPException(404, "Unknown institution.")
    folder = DATA_ROOT / institution
    try:
        tp = pd.read_csv(folder / "student_topic_performance.csv")
        topics = pd.read_csv(folder / "course_topics.csv")
        resources = pd.read_csv(folder / "learning_resources.csv")
        interactions = pd.read_csv(folder / "student_resource_interactions.csv")
    except Exception as e:
        raise HTTPException(500, f"Could not load institutional data: {e}")
    student_topics = tp[(tp.student_id == student_id) & (tp.course_id == course_id)].copy()
    if student_topics.empty: raise HTTPException(404, "No topic-performance records for this student/course.")
    student_topics["accuracy_percentage"] = pd.to_numeric(student_topics["accuracy_percentage"], errors="coerce")
    student_topics["practice_attempts"] = pd.to_numeric(student_topics["practice_attempts"], errors="coerce").fillna(0)
    student_topics = student_topics[student_topics.practice_attempts >= 3]
    weak = student_topics.sort_values(["accuracy_percentage","practice_attempts"], ascending=[True,False]).head(top_k)
    weak_ids = set(weak.topic_id.astype(str))
    topic_names = topics.set_index("topic_id")["topic_name"].to_dict()
    student_interactions = interactions[interactions.student_id == student_id].copy()
    student_interactions["completion_percentage"] = pd.to_numeric(student_interactions["completion_percentage"], errors="coerce")
    student_interactions["feedback_rating"] = pd.to_numeric(student_interactions["feedback_rating"], errors="coerce")
    interaction_stats = student_interactions.groupby("resource_id").agg(
        mean_completion=("completion_percentage","mean"), mean_rating=("feedback_rating","mean"), count=("resource_id","size")
    ).to_dict("index")
    candidates = resources[(resources.course_id == course_id) & resources.topic_id.astype(str).isin(weak_ids)].copy()
    ranked = []
    for row in candidates.to_dict("records"):
        stats = interaction_stats.get(str(row["resource_id"]), {})
        # Resource is ranked by weak-topic relevance; observed completion/feedback is a modest tie-breaker only.
        score = 1.0 - float(weak.loc[weak.topic_id.astype(str) == str(row["topic_id"]), "accuracy_percentage"].iloc[0] or 0)/100.0
        if stats.get("mean_completion") is not None and pd.notna(stats.get("mean_completion")):
            score += 0.10 * float(stats["mean_completion"])/100.0
        if stats.get("mean_rating") is not None and pd.notna(stats.get("mean_rating")):
            score += 0.05 * float(stats["mean_rating"])/5.0
        ranked.append({
            "resource_id": str(row["resource_id"]), "topic_id": str(row["topic_id"]),
            "topic_name": topic_names.get(row["topic_id"], "Unknown topic"),
            "resource_title": row["resource_title"], "resource_type": row["resource_type"],
            "difficulty_level": row["difficulty_level"], "ranking_score": round(score,4)
        })
    ranked.sort(key=lambda x:x["ranking_score"], reverse=True)
    return {
        "student_id": student_id, "course_id": course_id,
        "academic_improvement_areas": [
            {"topic_id": str(r.topic_id), "topic_name": topic_names.get(r.topic_id,"Unknown topic"),
             "accuracy_percentage": None if pd.isna(r.accuracy_percentage) else float(r.accuracy_percentage),
             "practice_attempts": int(r.practice_attempts)}
            for r in weak.itertuples()
        ],
        "recommended_resources": ranked[:top_k],
        "method": "Transparent weak-topic ranking and heuristic resource ranking; validate with held-out relevance data."
    }
