#!/usr/bin/env python3
"""
Federated Learning for Personalization
Synthetic institutional dataset generator.

Generates 3 independent institutional datasets, each containing the exact
21 tables specified in the project master prompt, plus shared documentation
and validation reports.

Requirements:
    pip install pandas numpy

Usage:
    python generate_federated_dataset.py
    python generate_federated_dataset.py --output ./federated_learning_dataset --seed 42
    python generate_federated_dataset.py --students-per-institution 500

All records are synthetic. No real student identities or private records are used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exact output schemas (column order is part of the contract)
# ---------------------------------------------------------------------------

TABLE_COLUMNS = {
    "institutions": ["institution_id", "institution_name", "institution_type", "country", "created_at", "is_active"],
    "departments": ["department_id", "institution_id", "department_name"],
    "programs": ["program_id", "department_id", "program_name", "education_level", "duration_years"],
    "academic_terms": ["term_id", "institution_id", "term_name", "academic_year", "start_date", "end_date"],
    "students": ["student_id", "institution_id", "program_id", "student_reference", "enrollment_date", "education_level", "current_year", "preferred_language", "created_at"],
    "student_enrollments": ["enrollment_id", "student_id", "course_id", "term_id", "enrollment_date", "enrollment_status"],
    "courses": ["course_id", "institution_id", "department_id", "course_code", "course_name", "subject_area", "credit_hours", "difficulty_level"],
    "course_topics": ["topic_id", "course_id", "parent_topic_id", "topic_name", "topic_description", "sequence_number"],
    "assessments": ["assessment_id", "course_id", "term_id", "assessment_name", "assessment_type", "max_marks", "assessment_date", "topic_id"],
    "student_assessment_results": ["result_id", "student_id", "assessment_id", "marks_obtained", "percentage", "submission_status", "submission_date"],
    "student_course_history": ["history_id", "student_id", "course_id", "term_id", "final_percentage", "final_grade", "gpa_points", "attendance_percentage", "completion_status", "completion_date"],
    "learning_sessions": ["session_id", "student_id", "course_id", "session_start", "session_end", "duration_minutes", "device_type"],
    "learning_activity_logs": ["activity_id", "session_id", "student_id", "course_id", "topic_id", "activity_type", "activity_timestamp", "time_spent_seconds", "completion_status"],
    "student_topic_performance": ["performance_id", "student_id", "course_id", "topic_id", "practice_attempts", "correct_attempts", "accuracy_percentage", "average_response_time", "last_practiced_at"],
    "student_academic_outcomes": ["outcome_id", "student_id", "course_id", "term_id", "successful_completion", "final_grade", "final_percentage", "outcome_date"],
    "learning_resources": ["resource_id", "course_id", "topic_id", "resource_title", "resource_type", "difficulty_level", "resource_url", "learning_objective"],
    "student_resource_interactions": ["interaction_id", "student_id", "resource_id", "interaction_date", "time_spent_seconds", "completion_percentage", "feedback_rating"],
    "federated_clients": ["client_id", "institution_id", "client_name", "client_status", "registered_at"],
    "federated_training_rounds": ["round_id", "global_model_version", "started_at", "completed_at", "aggregation_method", "round_status"],
    "federated_client_updates": ["update_id", "round_id", "client_id", "model_version", "sample_count", "update_status", "submitted_at"],
    "personalized_recommendations": ["recommendation_id", "student_id", "course_id", "recommendation_type", "topic_id", "resource_id", "prediction_score", "recommendation_text", "generated_at"],
}

INSTITUTIONS = [
    ("Synthetic University of Technology", "University", "India"),
    ("Synthetic National University", "University", "India"),
    ("Synthetic Institute of Higher Education", "College", "India"),
]
DEPARTMENT_NAMES = ["Computer Science", "Electrical Engineering", "Mathematics", "Business Administration"]
PROGRAMS = [
    ("B.Tech Computer Science", "Undergraduate", 4, 0),
    ("B.Sc Mathematics", "Undergraduate", 3, 2),
    ("MBA", "Postgraduate", 2, 3),
]
COURSE_CATALOG = [
    ("CS101", "Programming Fundamentals", "Programming", 4, "Beginner", 0, ["Variables", "Data Types", "Conditionals", "Loops", "Functions", "Arrays"]),
    ("CS201", "Data Structures", "Programming", 4, "Intermediate", 0, ["Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs"]),
    ("CS301", "Database Systems", "Data Science", 3, "Intermediate", 0, ["Relational Models", "SQL", "Normalization", "Transactions", "Indexing"]),
    ("CS401", "Machine Learning", "Data Science", 4, "Advanced", 0, ["Regression", "Classification", "Model Evaluation", "Neural Networks", "Clustering"]),
    ("EE101", "Circuit Fundamentals", "Engineering", 3, "Beginner", 1, ["Voltage", "Current", "Resistance", "Circuit Laws", "Network Analysis"]),
    ("EE201", "Digital Systems", "Engineering", 3, "Intermediate", 1, ["Number Systems", "Logic Gates", "Boolean Algebra", "Combinational Logic", "Sequential Logic"]),
    ("MA101", "Calculus", "Mathematics", 4, "Intermediate", 2, ["Limits", "Differentiation", "Integration", "Applications", "Series"]),
    ("MA201", "Linear Algebra", "Mathematics", 3, "Intermediate", 2, ["Vectors", "Matrices", "Determinants", "Eigenvalues", "Linear Transformations"]),
    ("SC101", "Physics for Computing", "Physics", 3, "Beginner", 2, ["Motion", "Energy", "Waves", "Electricity", "Optics"]),
    ("SC201", "Applied Statistics", "Statistics", 3, "Intermediate", 2, ["Descriptive Statistics", "Probability", "Distributions", "Hypothesis Testing", "Correlation"]),
    ("BU101", "Principles of Management", "Business", 3, "Beginner", 3, ["Management Functions", "Planning", "Organizing", "Leadership", "Control"]),
    ("BU201", "Business Communication", "Communication", 2, "Beginner", 3, ["Written Communication", "Presentations", "Listening", "Team Communication", "Reports"]),
]
LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Bengali"]
RESOURCE_TYPES = ["Video", "Article", "Quiz", "Exercise"]
ASSESSMENT_TYPES = ["Quiz", "Assignment", "Midterm Exam", "Practical", "Final Exam"]
DEVICES = ["Desktop", "Laptop", "Tablet", "Mobile"]
ACTIVITY_TYPES = ["Video", "Quiz", "Reading", "Practice"]
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]


def uid() -> str:
    return str(uuid.uuid4())


def iso_date(d: date) -> str:
    return d.isoformat()


def iso_dt(d: datetime) -> str:
    return d.replace(microsecond=0).isoformat(sep=" ")


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def weighted_choice(rng: np.random.Generator, values, weights):
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return values[int(rng.choice(len(values), p=p))]


def grade_for_percentage(pct: float) -> tuple[str, float]:
    if pct >= 90: return "A", 4.0
    if pct >= 80: return "B", 3.5
    if pct >= 70: return "C", 3.0
    if pct >= 60: return "D", 2.0
    if pct >= 50: return "E", 1.0
    return "F", 0.0


def build_terms(institution_id: str, start_year: int = 2023):
    rows = []
    terms = []
    for i in range(6):
        year = start_year + i // 2
        half = i % 2
        start = date(year, 1 if half == 0 else 7, 10)
        end = date(year, 6 if half == 0 else 12, 20)
        row = {
            "term_id": uid(), "institution_id": institution_id,
            "term_name": f"{year} {'Spring' if half == 0 else 'Fall'}",
            "academic_year": f"{year}-{year + 1}",
            "start_date": iso_date(start), "end_date": iso_date(end),
        }
        rows.append(row)
        terms.append((row["term_id"], start, end, i))
    return rows, terms


def make_institution(index: int, students_n: int, seed: int, output: Path):
    # Independent RNG streams and independent generation for every institution.
    rng = np.random.default_rng(seed + 1009 * (index + 1))
    py_rng = random.Random(seed + 1009 * (index + 1))
    institution_id = uid()
    institution_name, institution_type, country = INSTITUTIONS[index]
    now = datetime(2026, 1, 1, 10, 0)

    data: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_COLUMNS}

    # Institution management
    data["institutions"].append({
        "institution_id": institution_id, "institution_name": institution_name,
        "institution_type": institution_type, "country": country,
        "created_at": iso_dt(datetime(2010 + index, 1, 15, 9, 0)), "is_active": True,
    })

    department_ids = []
    for name in DEPARTMENT_NAMES:
        did = uid()
        department_ids.append(did)
        data["departments"].append({"department_id": did, "institution_id": institution_id, "department_name": name})

    program_ids = []
    program_info = {}
    for pname, level, duration, dept_idx in PROGRAMS:
        pid = uid()
        program_ids.append(pid)
        program_info[pid] = (pname, level, duration, dept_idx)
        data["programs"].append({
            "program_id": pid, "department_id": department_ids[dept_idx],
            "program_name": pname, "education_level": level, "duration_years": duration,
        })

    term_rows, terms = build_terms(institution_id)
    data["academic_terms"].extend(term_rows)

    # Course distribution differs by institution (Non-IID).
    course_weights = {
        0: [1.55 if c[2] in ("Programming", "Data Science", "Engineering") else 0.75 for c in COURSE_CATALOG],
        1: [1.35 if c[2] in ("Mathematics", "Statistics", "Physics") else 0.8 for c in COURSE_CATALOG],
        2: [1.15 for _ in COURSE_CATALOG],
    }[index]
    course_rows = []
    course_topics: dict[str, list[dict[str, Any]]] = {}
    course_by_id = {}
    for code, name, subject, credits, difficulty, dept_idx, topic_names in COURSE_CATALOG:
        cid = uid()
        c = {
            "course_id": cid, "institution_id": institution_id, "department_id": department_ids[dept_idx],
            "course_code": f"{'A' if index == 0 else 'B' if index == 1 else 'C'}-{code}",
            "course_name": name, "subject_area": subject, "credit_hours": float(credits),
            "difficulty_level": difficulty,
        }
        course_rows.append(c)
        course_by_id[cid] = c
        topic_rows = []
        root_id = uid()
        topic_rows.append({
            "topic_id": root_id, "course_id": cid, "parent_topic_id": None,
            "topic_name": f"{name} Foundations", "topic_description": f"Foundational concepts in {name}.",
            "sequence_number": 1,
        })
        for seq, topic_name in enumerate(topic_names, start=2):
            topic_rows.append({
                "topic_id": uid(), "course_id": cid, "parent_topic_id": root_id,
                "topic_name": topic_name, "topic_description": f"Study and apply {topic_name.lower()} in {name}.",
                "sequence_number": seq,
            })
        data["course_topics"].extend(topic_rows)
        course_topics[cid] = topic_rows
    data["courses"].extend(course_rows)

    # Programs have internal curriculum mappings; no extra mapping table is added.
    course_ids = [c["course_id"] for c in course_rows]
    curriculum = {
        program_ids[0]: [course_ids[i] for i in [0, 1, 2, 3, 4, 5, 8, 9]],
        program_ids[1]: [course_ids[i] for i in [6, 7, 8, 9, 0, 2, 10, 11]],
        program_ids[2]: [course_ids[i] for i in [10, 11, 9, 2, 3, 6, 7, 1]],
    }

    # Student records and latent behavioural profiles.
    student_rows = []
    student_profile = {}
    for s in range(students_n):
        # Program mix varies by institution.
        prog_weights = ([0.52, 0.28, 0.20], [0.25, 0.50, 0.25], [0.34, 0.33, 0.33])[index]
        pid = weighted_choice(rng, program_ids, prog_weights)
        pname, level, duration, _ = program_info[pid]
        profile = weighted_choice(rng, ["high", "average", "struggling", "inconsistent"], [0.22, 0.46, 0.20, 0.12])
        ability = {"high": 0.78, "average": 0.58, "struggling": 0.39, "inconsistent": 0.57}[profile] + rng.normal(0, 0.07)
        ability = clamp(ability, 0.15, 0.95)
        engagement = {"high": 0.78, "average": 0.56, "struggling": 0.34, "inconsistent": 0.48}[profile] + rng.normal(0, 0.12)
        engagement = clamp(engagement, 0.08, 0.98)
        attendance_base = {"high": 0.91, "average": 0.78, "struggling": 0.62, "inconsistent": 0.72}[profile]
        attendance_base += (0.06 if index == 0 else -0.04 if index == 2 else 0) + rng.normal(0, 0.08)
        attendance_base = clamp(attendance_base, 0.35, 0.99)
        start_term_idx = int(rng.integers(0, 3))
        _, term_start, _, _ = terms[start_term_idx]
        enroll_date = term_start - timedelta(days=int(rng.integers(7, 45)))
        sid = uid()
        student_rows.append({
            "student_id": sid, "institution_id": institution_id, "program_id": pid,
            "student_reference": f"INST{index+1}-STU-{s+1:05d}",
            "enrollment_date": iso_date(enroll_date), "education_level": level,
            "current_year": int(rng.integers(1, duration + 1)),
            "preferred_language": weighted_choice(rng, LANGUAGES, [0.52, 0.13, 0.18, 0.08, 0.05, 0.04]),
            "created_at": iso_dt(datetime.combine(enroll_date, time(9, 0))),
        })
        student_profile[sid] = {
            "program_id": pid, "profile": profile, "ability": ability,
            "engagement": engagement, "attendance": attendance_base,
            "start_term_idx": start_term_idx,
        }
    data["students"].extend(student_rows)

    # Learning resources
    resources_by_course = defaultdict(list)
    for cid, topics in course_topics.items():
        course_name = course_by_id[cid]["course_name"]
        for j in range(7):
            topic = topics[1 + (j % (len(topics) - 1))]
            rtype = RESOURCE_TYPES[j % len(RESOURCE_TYPES)]
            difficulty = DIFFICULTIES[min(2, max(0, (j // 2) % 3))]
            rid = uid()
            resource = {
                "resource_id": rid, "course_id": cid, "topic_id": topic["topic_id"],
                "resource_title": f"{topic['topic_name']} - {rtype} {j+1}",
                "resource_type": rtype, "difficulty_level": difficulty,
                "resource_url": f"https://example.invalid/{index+1}/{cid[:8]}/{rid[:8]}",
                "learning_objective": f"Build {difficulty.lower()} understanding of {topic['topic_name'].lower()} through {rtype.lower()}-based learning.",
            }
            data["learning_resources"].append(resource)
            resources_by_course[cid].append(resource)

    # Assessments: 5 per course per term; dates remain inside the term.
    assessments_by_course_term = {}
    assessment_lookup = {}
    for term_id, tstart, tend, term_idx in terms:
        for course in course_rows:
            cid = course["course_id"]
            rows = []
            for k, atype in enumerate(ASSESSMENT_TYPES):
                day_offset = [25, 45, 65, 82, 100][k]
                assessment_date = min(tend - timedelta(days=2), tstart + timedelta(days=day_offset))
                topic = course_topics[cid][1 + ((k * 2) % (len(course_topics[cid]) - 1))]
                aid = uid()
                a = {
                    "assessment_id": aid, "course_id": cid, "term_id": term_id,
                    "assessment_name": f"{atype} {k+1}", "assessment_type": atype,
                    "max_marks": 100.0, "assessment_date": iso_date(assessment_date),
                    "topic_id": topic["topic_id"],
                }
                data["assessments"].append(a)
                rows.append(a)
                assessment_lookup[aid] = a
            assessments_by_course_term[(cid, term_id)] = rows

    # Enrollment, learning, performance and outcomes.
    enrollment_lookup = set()
    student_course_term = {}
    topic_performance_index = {}
    term_lookup = {t[0]: t for t in terms}

    for student in student_rows:
        sid = student["student_id"]
        profile = student_profile[sid]
        eligible_courses = curriculum[profile["program_id"]]
        for term_id, tstart, tend, term_idx in terms:
            if term_idx < profile["start_term_idx"]:
                continue
            # 4-6 courses per term; curriculum courses repeat naturally across terms.
            n_courses = int(rng.integers(4, 7))
            weights = np.array([course_weights[course_ids.index(cid)] for cid in eligible_courses], dtype=float)
            weights /= weights.sum()
            chosen = list(rng.choice(eligible_courses, size=min(n_courses, len(eligible_courses)), replace=False, p=weights))
            for cid in chosen:
                key = (sid, cid, term_id)
                if key in enrollment_lookup:
                    continue
                enrollment_lookup.add(key)
                # Later terms are more likely to have completed records; some remain active.
                if term_idx == 5 and rng.random() < 0.16:
                    status = "Active"
                elif rng.random() < (0.055 + (0.025 if index == 2 else 0)):
                    status = "Withdrawn"
                else:
                    status = "Completed"
                withdrawal_date = (tstart + timedelta(days=int((tend - tstart).days * 0.55))) if status == "Withdrawn" else None
                participation_end = withdrawal_date if withdrawal_date is not None else tend
                enrollment_date = max(date.fromisoformat(student["enrollment_date"]), tstart - timedelta(days=int(rng.integers(1, 12))))
                enrollment_id = uid()
                data["student_enrollments"].append({
                    "enrollment_id": enrollment_id, "student_id": sid, "course_id": cid,
                    "term_id": term_id, "enrollment_date": iso_date(enrollment_date),
                    "enrollment_status": status,
                })
                student_course_term[key] = status

                # Attendance and course outcome latent score; institutional shifts create Non-IID.
                att = clamp(profile["attendance"] + rng.normal(0, 0.10) + (0.025 if index == 0 else -0.035 if index == 2 else 0), 0.20, 1.0)
                difficulty_penalty = {"Beginner": 0, "Intermediate": 0.035, "Advanced": 0.075}[course_by_id[cid]["difficulty_level"]]
                institution_shift = [0.015, -0.015, -0.005][index]
                latent = (profile["ability"] * 55 + att * 17 + profile["engagement"] * 12
                          - difficulty_penalty * 100 + institution_shift * 100 + rng.normal(0, 12))
                course_pct = clamp(latent, 8, 99)
                if status == "Withdrawn":
                    course_pct = clamp(course_pct - rng.uniform(12, 28), 5, 48)
                if status == "Active":
                    course_pct = clamp(course_pct - rng.uniform(2, 12), 5, 95)
                course_pct = round(float(course_pct), 2)
                grade, gpa = grade_for_percentage(course_pct)
                completion_status = "Withdrawn" if status == "Withdrawn" else ("Completed" if status == "Completed" and course_pct >= 50 else "Failed")
                # For an active enrollment, no final course history/outcome is fabricated.
                attendance_pct = round(att * 100, 2)

                # Learning sessions occur within the term, and only for enrolled course.
                session_count = int(rng.poisson(1.5 + profile["engagement"] * 2.5))
                session_count = int(clamp(session_count, 1, 8))
                course_sessions = []
                for _ in range(session_count):
                    available_days = max(6, (participation_end - tstart).days - 1)
                    day = tstart + timedelta(days=int(rng.integers(5, available_days)))
                    start_dt = datetime.combine(day, time(int(rng.integers(8, 21)), int(rng.choice([0, 15, 30, 45]))))
                    duration = float(clamp(rng.lognormal(math.log(42), 0.48), 8, 180))
                    end_dt = start_dt + timedelta(minutes=duration)
                    # Keep sessions inside the academic term.
                    if end_dt.date() > participation_end:
                        end_dt = datetime.combine(participation_end, time(18, 0))
                        duration = max(1.0, (end_dt - start_dt).total_seconds() / 60)
                    session_id = uid()
                    session = {
                        "session_id": session_id, "student_id": sid, "course_id": cid,
                        "session_start": iso_dt(start_dt), "session_end": iso_dt(end_dt),
                        "duration_minutes": round(duration, 2), "device_type": weighted_choice(rng, DEVICES, [0.22, 0.46, 0.16, 0.16]),
                    }
                    data["learning_sessions"].append(session)
                    course_sessions.append((session, start_dt, end_dt))

                    # 1-3 activities per session, timestamped inside the session.
                    for _a in range(int(rng.integers(1, 3))):
                        topic = topics = course_topics[cid][int(rng.integers(1, len(course_topics[cid])))]
                        seconds = int(min(rng.integers(60, 2401), max(1, duration * 60)))
                        ts = start_dt + timedelta(seconds=int(rng.integers(0, max(1, int(duration * 60)))))
                        ts = min(ts, end_dt)
                        data["learning_activity_logs"].append({
                            "activity_id": uid(), "session_id": session_id, "student_id": sid,
                            "course_id": cid, "topic_id": topic["topic_id"],
                            "activity_type": weighted_choice(rng, ACTIVITY_TYPES, [0.20, 0.22, 0.23, 0.35]),
                            "activity_timestamp": iso_dt(ts), "time_spent_seconds": seconds,
                            "completion_status": "Completed" if rng.random() < profile["engagement"] else "Started",
                        })

                # Topic practice aggregates. Accuracy correlates imperfectly with ability and topic difficulty.
                topic_perf_for_course = []
                for topic in course_topics[cid][1:]:
                    attempts = int(rng.poisson(1.5 + profile["engagement"] * 5))
                    attempts = max(0, attempts)
                    topic_effect = rng.normal(0, 0.13)
                    accuracy = clamp(profile["ability"] + topic_effect + (profile["engagement"] - 0.5) * 0.12, 0.05, 0.98)
                    correct = int(rng.binomial(attempts, accuracy)) if attempts else 0
                    acc_pct = round(correct / attempts * 100, 2) if attempts else None
                    avg_response = round(float(clamp(rng.normal(48 + (1 - accuracy) * 45, 17), 5, 240)), 2) if attempts else None
                    last_practiced = None
                    if attempts:
                        max_practice_day = min(participation_end, tstart + timedelta(days=max(6, int((tend - tstart).days * 0.25))))
                        last_day = tstart + timedelta(days=int(rng.integers(5, max(6, (max_practice_day - tstart).days + 1))))
                        last_practiced = iso_dt(datetime.combine(min(last_day, participation_end), time(16, 0)))
                    perf = {
                        "performance_id": uid(), "student_id": sid, "course_id": cid,
                        "topic_id": topic["topic_id"], "practice_attempts": attempts,
                        "correct_attempts": correct, "accuracy_percentage": acc_pct,
                        "average_response_time": avg_response, "last_practiced_at": last_practiced,
                    }
                    data["student_topic_performance"].append(perf)
                    topic_perf_for_course.append(perf)
                topic_performance_index[key] = topic_perf_for_course

                # Resource interactions take place during term after course enrollment.
                for resource in resources_by_course[cid]:
                    if rng.random() < (0.12 + profile["engagement"] * 0.50):
                        max_interaction_day = max(tstart + timedelta(days=7), participation_end)
                        interaction_day = tstart + timedelta(days=int(rng.integers(7, max(8, (max_interaction_day - tstart).days + 1))))
                        interaction_day = min(interaction_day, participation_end)
                        data["student_resource_interactions"].append({
                            "interaction_id": uid(), "student_id": sid, "resource_id": resource["resource_id"],
                            "interaction_date": iso_dt(datetime.combine(interaction_day, time(12, 0))),
                            "time_spent_seconds": int(rng.integers(45, 3601)),
                            "completion_percentage": round(float(clamp(rng.beta(2.0 + profile["engagement"] * 2, 2.4) * 100, 0, 100)), 2),
                            "feedback_rating": int(rng.integers(1, 6)) if rng.random() < 0.62 else None,
                        })

                # Assessment scores are generated from the same latent academic ability,
                # with noise and topic-specific deviations.
                assessments = assessments_by_course_term[(cid, term_id)]
                for a in assessments:
                    if date.fromisoformat(a["assessment_date"]) > participation_end:
                        continue
                    missing_prob = 0.035 + (1 - profile["engagement"]) * 0.10
                    missing = rng.random() < missing_prob
                    if missing:
                        marks = None
                        submission_date = None
                        submission_status = "Missing"
                    else:
                        score = clamp(course_pct + rng.normal(0, 11) + rng.normal(0, 4), 0, 100)
                        marks = round(float(score), 2)
                        submission_date = iso_dt(datetime.combine(date.fromisoformat(a["assessment_date"]), time(12, 0)) + timedelta(hours=int(rng.integers(0, 48))))
                        submission_status = "Submitted"
                    data["student_assessment_results"].append({
                        "result_id": uid(), "student_id": sid, "assessment_id": a["assessment_id"],
                        "marks_obtained": marks, "percentage": round(marks, 2) if marks is not None else None,
                        "submission_status": submission_status, "submission_date": submission_date,
                    })

                if status != "Active":
                    completion_date = tend if status == "Completed" else withdrawal_date
                    data["student_course_history"].append({
                        "history_id": uid(), "student_id": sid, "course_id": cid, "term_id": term_id,
                        "final_percentage": round(course_pct, 2), "final_grade": grade,
                        "gpa_points": gpa, "attendance_percentage": attendance_pct,
                        "completion_status": completion_status, "completion_date": iso_date(completion_date),
                    })
                    data["student_academic_outcomes"].append({
                        "outcome_id": uid(), "student_id": sid, "course_id": cid, "term_id": term_id,
                        "successful_completion": bool(completion_status == "Completed"),
                        "final_grade": grade, "final_percentage": round(course_pct, 2),
                        "outcome_date": iso_date(completion_date),
                    })

    # Recommendations: use only topic/resource information dated on or before generation time.
    # Recommendations are generated at a mid/late-term timestamp and never use outcome labels.
    resources_by_topic = defaultdict(list)
    for cid, resources in resources_by_course.items():
        for r in resources:
            resources_by_topic[(cid, r["topic_id"])].append(r)

    for student in student_rows:
        sid = student["student_id"]
        profile = student_profile[sid]
        candidate_rows = [
            (key, perfs) for key, perfs in topic_performance_index.items()
            if key[0] == sid
        ]
        # Limit recommendations per student to keep output useful rather than noisy.
        py_rng.shuffle(candidate_rows)
        for (key, perfs) in candidate_rows[:2]:
            _, cid, term_id = key
            term = term_lookup[term_id]
            tstart, tend, term_idx = term[1], term[2], term[3]
            generated_day = tstart + timedelta(days=int((tend - tstart).days * 0.35))
            generated_at = datetime.combine(generated_day, time(15, 0))
            if not perfs:
                continue
            weak = sorted(
                [p for p in perfs if p["accuracy_percentage"] is not None],
                key=lambda p: p["accuracy_percentage"]
            )
            if weak:
                target = weak[0]
                topic_id = target["topic_id"]
                acc = float(target["accuracy_percentage"])
                if acc < 72:
                    score = clamp((100 - acc) / 100 + rng.normal(0, 0.025), 0.05, 0.99)
                    data["personalized_recommendations"].append({
                        "recommendation_id": uid(), "student_id": sid, "course_id": cid,
                        "recommendation_type": "Improvement", "topic_id": topic_id, "resource_id": None,
                        "prediction_score": round(float(score), 4),
                        "recommendation_text": f"Review this topic and complete additional practice; recorded practice accuracy is {acc:.1f}%.",
                        "generated_at": iso_dt(generated_at),
                    })
                eligible_resources = [
                    r for r in resources_by_topic[(cid, topic_id)]
                    if r["difficulty_level"] in ("Beginner", "Intermediate")
                ]
                if eligible_resources:
                    resource = eligible_resources[0]
                    score = clamp(0.55 + (100 - acc) / 250 + rng.normal(0, 0.03), 0.05, 0.99)
                    data["personalized_recommendations"].append({
                        "recommendation_id": uid(), "student_id": sid, "course_id": cid,
                        "recommendation_type": "Resource", "topic_id": topic_id,
                        "resource_id": resource["resource_id"], "prediction_score": round(float(score), 4),
                        "recommendation_text": f"Use '{resource['resource_title']}' to reinforce the topic at an appropriate difficulty.",
                        "generated_at": iso_dt(generated_at),
                    })
            else:
                topic = course_topics[cid][1]
                data["personalized_recommendations"].append({
                    "recommendation_id": uid(), "student_id": sid, "course_id": cid,
                    "recommendation_type": "Activity", "topic_id": topic["topic_id"], "resource_id": None,
                    "prediction_score": round(float(clamp(0.55 + profile["engagement"] * 0.25, 0, 1)), 4),
                    "recommendation_text": f"Begin a short practice activity for {topic['topic_name']}.",
                    "generated_at": iso_dt(generated_at),
                })

    # One local federated client per institution; shared round history replicated consistently.
    client_id = uid()
    data["federated_clients"].append({
        "client_id": client_id, "institution_id": institution_id,
        "client_name": f"Institution {chr(65+index)} Local Training Client",
        "client_status": "Active", "registered_at": iso_dt(datetime(2025, 1, 5, 9, 0)),
    })

    # Use one shared set of round IDs across all three institutions (stable via UUID5).
    # This is metadata only: no model weights or student-level records are exported.
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    for round_no in range(1, 6):
        started = datetime(2025, 1, 10, 9, 0) + timedelta(days=(round_no - 1) * 30)
        completed = started + timedelta(hours=2)
        rid = str(uuid.uuid5(namespace, f"federated-round-{round_no}"))
        data["federated_training_rounds"].append({
            "round_id": rid, "global_model_version": f"global-v{round_no}",
            "started_at": iso_dt(started), "completed_at": iso_dt(completed),
            "aggregation_method": "FedAvg", "round_status": "Completed",
        })
        # sample_count is local and based on locally generated eligible course outcomes.
        local_samples = max(1, sum(1 for o in data["student_academic_outcomes"] if o["outcome_date"] <= "2025-12-31"))
        submitted = started + timedelta(minutes=int(rng.integers(10, 90)))
        data["federated_client_updates"].append({
            "update_id": str(uuid.uuid5(namespace, f"client-update-{round_no}-{index}")),
            "round_id": rid, "client_id": client_id, "model_version": f"local-v{round_no}",
            "sample_count": local_samples, "update_status": "Submitted",
            "submitted_at": iso_dt(submitted),
        })

    # Dataframes, exact column order, CSVs.
    inst_dir = output / f"institution_{chr(65+index)}"
    inst_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for table, columns in TABLE_COLUMNS.items():
        df = pd.DataFrame(data[table], columns=columns)
        frames[table] = df
        df.to_csv(inst_dir / f"{table}.csv", index=False, encoding="utf-8")

    return {
        "institution_index": index, "institution_id": institution_id,
        "institution_name": institution_name, "directory": inst_dir,
        "frames": frames,
    }


def write_documentation(results, output: Path, seed: int, students_n: int):
    docs = output / "shared_documentation"
    docs.mkdir(parents=True, exist_ok=True)

    pk_map = {
        "institutions":"institution_id", "departments":"department_id", "programs":"program_id",
        "academic_terms":"term_id", "students":"student_id", "student_enrollments":"enrollment_id",
        "courses":"course_id", "course_topics":"topic_id", "assessments":"assessment_id",
        "student_assessment_results":"result_id", "student_course_history":"history_id",
        "learning_sessions":"session_id", "learning_activity_logs":"activity_id",
        "student_topic_performance":"performance_id", "student_academic_outcomes":"outcome_id",
        "learning_resources":"resource_id", "student_resource_interactions":"interaction_id",
        "federated_clients":"client_id", "federated_training_rounds":"round_id",
        "federated_client_updates":"update_id", "personalized_recommendations":"recommendation_id",
    }
    fk_map = {
        ("departments","institution_id"):"institutions.institution_id",
        ("programs","department_id"):"departments.department_id",
        ("academic_terms","institution_id"):"institutions.institution_id",
        ("students","institution_id"):"institutions.institution_id", ("students","program_id"):"programs.program_id",
        ("student_enrollments","student_id"):"students.student_id", ("student_enrollments","course_id"):"courses.course_id",
        ("student_enrollments","term_id"):"academic_terms.term_id",
        ("courses","institution_id"):"institutions.institution_id", ("courses","department_id"):"departments.department_id",
        ("course_topics","course_id"):"courses.course_id", ("course_topics","parent_topic_id"):"course_topics.topic_id (self-reference)",
        ("assessments","course_id"):"courses.course_id", ("assessments","term_id"):"academic_terms.term_id",
        ("assessments","topic_id"):"course_topics.topic_id", ("student_assessment_results","student_id"):"students.student_id",
        ("student_assessment_results","assessment_id"):"assessments.assessment_id",
        ("student_course_history","student_id"):"students.student_id", ("student_course_history","course_id"):"courses.course_id",
        ("student_course_history","term_id"):"academic_terms.term_id",
        ("learning_sessions","student_id"):"students.student_id", ("learning_sessions","course_id"):"courses.course_id",
        ("learning_activity_logs","session_id"):"learning_sessions.session_id", ("learning_activity_logs","student_id"):"students.student_id",
        ("learning_activity_logs","course_id"):"courses.course_id", ("learning_activity_logs","topic_id"):"course_topics.topic_id",
        ("student_topic_performance","student_id"):"students.student_id", ("student_topic_performance","course_id"):"courses.course_id",
        ("student_topic_performance","topic_id"):"course_topics.topic_id",
        ("student_academic_outcomes","student_id"):"students.student_id", ("student_academic_outcomes","course_id"):"courses.course_id",
        ("student_academic_outcomes","term_id"):"academic_terms.term_id",
        ("learning_resources","course_id"):"courses.course_id", ("learning_resources","topic_id"):"course_topics.topic_id",
        ("student_resource_interactions","student_id"):"students.student_id", ("student_resource_interactions","resource_id"):"learning_resources.resource_id",
        ("federated_clients","institution_id"):"institutions.institution_id",
        ("federated_client_updates","round_id"):"federated_training_rounds.round_id", ("federated_client_updates","client_id"):"federated_clients.client_id",
        ("personalized_recommendations","student_id"):"students.student_id", ("personalized_recommendations","course_id"):"courses.course_id",
        ("personalized_recommendations","topic_id"):"course_topics.topic_id", ("personalized_recommendations","resource_id"):"learning_resources.resource_id",
    }
    meanings = {
        "institution_id":"Identifier of the institution owning the record.", "student_id":"Pseudonymous identifier of the student.",
        "program_id":"Program in which the student is enrolled.", "department_id":"Department that owns the program or course.",
        "course_id":"Identifier of the course in the institutional catalog.", "term_id":"Academic term associated with the record.",
        "topic_id":"Course topic associated with the record.", "parent_topic_id":"Parent topic identifier; NULL for a root topic.",
        "enrollment_date":"Date on which the student or course enrollment became effective.",
        "enrollment_status":"Course enrollment state: Active, Completed, or Withdrawn.",
        "successful_completion":"Supervised binary label; True means passed, False means failed or withdrawn.",
        "final_percentage":"Final synthetic course percentage, from 0 to 100.", "final_grade":"Letter grade derived from final percentage.",
        "gpa_points":"Synthetic GPA points derived from the agreed grading scale.",
        "attendance_percentage":"Attendance percentage observed for the course.",
        "marks_obtained":"Marks earned for an assessment; NULL when submission is Missing.",
        "percentage":"Assessment percentage calculated from marks_obtained and max_marks.",
        "submission_status":"Assessment submission state: Submitted or Missing.",
        "practice_attempts":"Number of practice attempts recorded for the student-topic pair.",
        "correct_attempts":"Number of correct practice attempts.",
        "accuracy_percentage":"Correct attempts divided by practice attempts, multiplied by 100; NULL when no attempts.",
        "average_response_time":"Average response time in seconds for the topic practice attempts.",
        "duration_minutes":"Elapsed session duration in minutes, calculated from session timestamps.",
        "time_spent_seconds":"Time spent on the associated learning activity or resource, in seconds.",
        "completion_percentage":"Percentage of the learning resource completed, from 0 to 100.",
        "feedback_rating":"Optional student feedback rating from 1 to 5.",
        "prediction_score":"Synthetic recommendation ranking/confidence score, from 0 to 1.",
        "resource_id":"Identifier of the learning resource associated with the record.",
        "assessment_id":"Identifier of the assessment associated with the result.",
        "session_id":"Identifier of the learning session containing the activity.",
        "round_id":"Identifier of the shared federated training round.", "client_id":"Identifier of the institution's local federated client.",
        "sample_count":"Number of local training examples represented by the client update.",
        "global_model_version":"Version label for the global model in this round.",
        "model_version":"Version label for the institution's local model update.",
        "aggregation_method":"Federated aggregation algorithm; FedAvg is used in this synthetic run.",
        "resource_url":"Fictional placeholder URL; not intended to resolve to a real resource.",
        "recommendation_text":"Human-readable explanation of the suggested learning action.",
        "learning_objective":"Learning outcome the resource is intended to support.",
    }
    nullable = {("course_topics","parent_topic_id"), ("assessments","topic_id"),
                ("student_assessment_results","marks_obtained"), ("student_assessment_results","submission_date"),
                ("student_course_history","completion_date"), ("student_topic_performance","accuracy_percentage"),
                ("student_topic_performance","average_response_time"), ("student_topic_performance","last_practiced_at"),
                ("student_academic_outcomes","outcome_date"), ("learning_resources","resource_url"),
                ("student_resource_interactions","feedback_rating"), ("federated_training_rounds","completed_at"),
                ("personalized_recommendations","topic_id"), ("personalized_recommendations","resource_id")}
    def inferred_type(table, col):
        if col == pk_map.get(table) or col.endswith("_id"):
            return "UUID"
        if col in {"created_at","registered_at","session_start","session_end","activity_timestamp","last_practiced_at","submission_date","interaction_date","generated_at","started_at","completed_at","submitted_at"}:
            return "TIMESTAMP"
        if col in {"enrollment_date","start_date","end_date","assessment_date","completion_date","outcome_date"}:
            return "DATE"
        if col in {"is_active","successful_completion"}:
            return "BOOLEAN"
        if col in {"current_year","duration_years","sequence_number","practice_attempts","correct_attempts","time_spent_seconds","feedback_rating","sample_count"}:
            return "INTEGER"
        if col in {"credit_hours","max_marks","marks_obtained","percentage","final_percentage","gpa_points","attendance_percentage","duration_minutes","average_response_time","accuracy_percentage","completion_percentage","prediction_score"}:
            return "DECIMAL"
        if col in {"topic_description","resource_url","learning_objective","recommendation_text"}:
            return "TEXT"
        return "VARCHAR"
    dictionary_rows = []
    for table, cols in TABLE_COLUMNS.items():
        for col in cols:
            fk_target = fk_map.get((table, col))
            key_role = "PRIMARY KEY" if col == pk_map.get(table) else (f"FOREIGN KEY -> {fk_target}" if fk_target else "")
            dictionary_rows.append({
                "table_name": table, "column_name": col, "data_type": inferred_type(table, col),
                "nullable": (table, col) in nullable,
                "key_role": key_role,
                "description": meanings.get(col, f"{col.replace('_', ' ').capitalize()} value used by the {table} table."),
            })
    pd.DataFrame(dictionary_rows).to_csv(docs / "data_dictionary.csv", index=False, encoding="utf-8")

    relationships = """# Database relationships

All IDs are UUID strings. Each institution folder is a separate local data silo.

- departments.institution_id -> institutions.institution_id
- programs.department_id -> departments.department_id
- academic_terms.institution_id -> institutions.institution_id
- students.institution_id -> institutions.institution_id
- students.program_id -> programs.program_id
- courses.institution_id -> institutions.institution_id
- courses.department_id -> departments.department_id
- student_enrollments.student_id -> students.student_id
- student_enrollments.course_id -> courses.course_id
- student_enrollments.term_id -> academic_terms.term_id
- course_topics.course_id -> courses.course_id
- course_topics.parent_topic_id -> course_topics.topic_id (nullable self-reference)
- assessments.course_id -> courses.course_id
- assessments.term_id -> academic_terms.term_id
- assessments.topic_id -> course_topics.topic_id (nullable)
- student_assessment_results.student_id -> students.student_id
- student_assessment_results.assessment_id -> assessments.assessment_id
- student_course_history.student_id -> students.student_id
- student_course_history.course_id -> courses.course_id
- student_course_history.term_id -> academic_terms.term_id
- learning_sessions.student_id -> students.student_id
- learning_sessions.course_id -> courses.course_id
- learning_activity_logs.session_id -> learning_sessions.session_id
- learning_activity_logs.student_id -> students.student_id
- learning_activity_logs.course_id -> courses.course_id
- learning_activity_logs.topic_id -> course_topics.topic_id
- student_topic_performance.student_id -> students.student_id
- student_topic_performance.course_id -> courses.course_id
- student_topic_performance.topic_id -> course_topics.topic_id
- student_academic_outcomes.student_id -> students.student_id
- student_academic_outcomes.course_id -> courses.course_id
- student_academic_outcomes.term_id -> academic_terms.term_id
- learning_resources.course_id -> courses.course_id
- learning_resources.topic_id -> course_topics.topic_id
- student_resource_interactions.student_id -> students.student_id
- student_resource_interactions.resource_id -> learning_resources.resource_id
- federated_clients.institution_id -> institutions.institution_id
- federated_client_updates.round_id -> federated_training_rounds.round_id
- federated_client_updates.client_id -> federated_clients.client_id
- personalized_recommendations.student_id -> students.student_id
- personalized_recommendations.course_id -> courses.course_id
- personalized_recommendations.topic_id -> course_topics.topic_id (nullable)
- personalized_recommendations.resource_id -> learning_resources.resource_id (nullable)

The agreed schema has no program-course junction table. Curriculum membership is therefore
used internally by the generator to select plausible course enrollments and is not exported.
"""
    (docs / "schema_relationships.md").write_text(relationships, encoding="utf-8")

    feature_text = """# Feature definitions

The 21 source tables are relational event/history data, not a single preassembled model matrix.
Build point-in-time training examples by joining records using student, course, term and timestamps.

## Course completion
- Target: student_academic_outcomes.successful_completion (boolean; map True=1, False=0).
- Candidate predictors available at prediction time: prior course history, assessments submitted by cutoff,
  attendance observed by cutoff, learning session frequency/duration, activity completion, and topic practice.
- Exclude same-course final_percentage, final_grade, completion_status, completion_date, outcome_date,
  and any assessment/session/activity recorded after the prediction cutoff.

## Academic improvement
- Candidate signals: student_topic_performance.accuracy_percentage, attempts, response time,
  and assessment scores mapped to topic_id.
- A transparent initial rule is to rank topics with lower accuracy first, with minimum-attempt
  safeguards. This is a derived heuristic, not an independently observed ground-truth label.

## Resource suitability
- Candidate signals: topic weakness, course/topic match, resource difficulty,
  prior student_resource_interactions completion_percentage and feedback_rating.
- An interaction is behavioural evidence, not proof of preference or learning gain.

## Personalized recommendation
- Combine course completion risk, weak-topic ranking, resource suitability and activity rules.
- Evaluate ranking with Precision@K/Recall@K or NDCG when independently collected relevance labels exist.
"""
    (docs / "feature_definitions.md").write_text(feature_text, encoding="utf-8")

    targets = """# Target variable documentation

1. Course completion: `student_academic_outcomes.successful_completion`; binary classification.
   True means passed/completed; False means failed or withdrawn. Do not use final course outcomes
   as predictors for the same course.
2. Improvement areas: no direct human-labelled target is included. Derive a transparent topic-weakness
   ranking from topic practice and topic-linked assessments, or create a separate mastery label.
3. Resource suitability: interaction history is implicit feedback, not a definitive suitability label.
   Use time-aware ranking/implicit-feedback evaluation and avoid treating every click as positive.
4. Personalized recommendations: a ranking/recommendation objective; evaluate with held-out,
   time-separated interactions or independently judged relevance labels.

The four objectives do not share one common target. Define separate training and evaluation pipelines.
"""
    (docs / "target_variable_documentation.md").write_text(targets, encoding="utf-8")

    methodology = f"""# Synthetic data generation report

- Random seed: {seed}
- Students per institution: {students_n}
- Institutions: 3
- All people, institutions and records are fictional.
- Each institution is generated with an independent random-number stream.
- Institutional course/program mixes, attendance, engagement and outcome distributions differ.
- Student profiles create correlated but imperfect academic/engagement patterns.
- Course scores combine latent ability, attendance, engagement, course difficulty, institution shift and noise.
- Assessments, practice, resource interactions and recommendations are generated within institutional context.
- The generator exports no model weights and no central student-level dataset.
- The program-to-course curriculum mapping is internal because the agreed 21-table schema has no junction table.

Important schema interpretation:
- Active latest-term enrollments do not receive final course history or final outcomes.
- Withdrawn courses receive an outcome label of False and a recorded synthetic final/withdrawal percentage.
- The `student_course_history` and `student_academic_outcomes` final values are aligned.
- Recommendations are created from topic-practice information and do not use same-course final outcome labels.
"""
    (docs / "synthetic_data_generation_report.md").write_text(methodology, encoding="utf-8")

    distribution = ["# Federated data distribution report", "",
                    "Each institution directory is an independent local silo. Raw student records are not exported to a shared directory.",
                    "Federated round IDs are shared across institutions; each local folder records only its own client update metadata.", ""]
    for r in results:
        f = r["frames"]
        distribution.append(f"## {r['institution_name']}")
        distribution.append(f"- Students: {len(f['students'])}")
        distribution.append(f"- Course enrollments: {len(f['student_enrollments'])}")
        distribution.append(f"- Academic outcomes: {len(f['student_academic_outcomes'])}")
        distribution.append("")
    (docs / "federated_data_distribution_report.md").write_text("\n".join(distribution), encoding="utf-8")


def validate_institution(result):
    frames = result["frames"]
    issues = []
    stats = []
    idx = result["institution_index"]

    # Exact schema and primary keys.
    for table, columns in TABLE_COLUMNS.items():
        df = frames[table]
        if list(df.columns) != columns:
            issues.append((table, "schema", "Column names/order do not match required schema"))
        pk = {
            "institutions": "institution_id", "departments": "department_id", "programs": "program_id",
            "academic_terms": "term_id", "students": "student_id", "student_enrollments": "enrollment_id",
            "courses": "course_id", "course_topics": "topic_id", "assessments": "assessment_id",
            "student_assessment_results": "result_id", "student_course_history": "history_id",
            "learning_sessions": "session_id", "learning_activity_logs": "activity_id",
            "student_topic_performance": "performance_id", "student_academic_outcomes": "outcome_id",
            "learning_resources": "resource_id", "student_resource_interactions": "interaction_id",
            "federated_clients": "client_id", "federated_training_rounds": "round_id",
            "federated_client_updates": "update_id", "personalized_recommendations": "recommendation_id",
        }[table]
        if df[pk].duplicated().any():
            issues.append((table, "primary_key", f"Duplicate {pk} values"))

    def fk(child, col, parent, parent_col):
        valid = set(frames[parent][parent_col].dropna())
        bad = frames[child][col].notna() & ~frames[child][col].isin(valid)
        if bad.any():
            issues.append((child, "foreign_key", f"{int(bad.sum())} invalid {col} references {parent}.{parent_col}"))

    checks = [
        ("departments","institution_id","institutions","institution_id"),
        ("programs","department_id","departments","department_id"),
        ("academic_terms","institution_id","institutions","institution_id"),
        ("students","institution_id","institutions","institution_id"),
        ("students","program_id","programs","program_id"),
        ("courses","institution_id","institutions","institution_id"),
        ("courses","department_id","departments","department_id"),
        ("student_enrollments","student_id","students","student_id"),
        ("student_enrollments","course_id","courses","course_id"),
        ("student_enrollments","term_id","academic_terms","term_id"),
        ("course_topics","course_id","courses","course_id"),
        ("course_topics","parent_topic_id","course_topics","topic_id"),
        ("assessments","course_id","courses","course_id"),
        ("assessments","term_id","academic_terms","term_id"),
        ("student_assessment_results","student_id","students","student_id"),
        ("student_assessment_results","assessment_id","assessments","assessment_id"),
        ("student_course_history","student_id","students","student_id"),
        ("student_course_history","course_id","courses","course_id"),
        ("student_course_history","term_id","academic_terms","term_id"),
        ("learning_sessions","student_id","students","student_id"),
        ("learning_sessions","course_id","courses","course_id"),
        ("learning_activity_logs","session_id","learning_sessions","session_id"),
        ("learning_activity_logs","student_id","students","student_id"),
        ("learning_activity_logs","course_id","courses","course_id"),
        ("learning_activity_logs","topic_id","course_topics","topic_id"),
        ("student_topic_performance","student_id","students","student_id"),
        ("student_topic_performance","course_id","courses","course_id"),
        ("student_topic_performance","topic_id","course_topics","topic_id"),
        ("student_academic_outcomes","student_id","students","student_id"),
        ("student_academic_outcomes","course_id","courses","course_id"),
        ("student_academic_outcomes","term_id","academic_terms","term_id"),
        ("learning_resources","course_id","courses","course_id"),
        ("learning_resources","topic_id","course_topics","topic_id"),
        ("student_resource_interactions","student_id","students","student_id"),
        ("student_resource_interactions","resource_id","learning_resources","resource_id"),
        ("federated_clients","institution_id","institutions","institution_id"),
        ("federated_client_updates","round_id","federated_training_rounds","round_id"),
        ("federated_client_updates","client_id","federated_clients","client_id"),
        ("personalized_recommendations","student_id","students","student_id"),
        ("personalized_recommendations","course_id","courses","course_id"),
        ("personalized_recommendations","topic_id","course_topics","topic_id"),
        ("personalized_recommendations","resource_id","learning_resources","resource_id"),
    ]
    for args in checks:
        fk(*args)

    # Same-course / same-institution relationships.
    inst_id = result["institution_id"]
    for table in ("students", "courses", "departments", "academic_terms", "federated_clients"):
        if len(frames[table]) and not (frames[table]["institution_id"] == inst_id).all():
            issues.append((table, "institution_scope", "Record belongs to another institution"))

    # Academic arithmetic and ranges.
    res = frames["student_assessment_results"]
    submitted = res[res["submission_status"] == "Submitted"]
    if not submitted.empty:
        if submitted["marks_obtained"].isna().any() or (submitted["marks_obtained"] > 100).any() or (submitted["marks_obtained"] < 0).any():
            issues.append(("student_assessment_results", "marks", "Submitted marks missing or outside 0..100"))
        if not np.allclose(submitted["percentage"].astype(float), submitted["marks_obtained"].astype(float), atol=0.011):
            issues.append(("student_assessment_results", "percentage", "Assessment percentage does not match marks/max marks"))
    missing = res[res["submission_status"] == "Missing"]
    if not missing.empty and (missing["marks_obtained"].notna().any() or missing["submission_date"].notna().any()):
        issues.append(("student_assessment_results", "missing_submission", "Missing submissions must have NULL marks/date"))

    hist = frames["student_course_history"]
    if not hist.empty:
        if ((hist["attendance_percentage"] < 0) | (hist["attendance_percentage"] > 100)).any():
            issues.append(("student_course_history", "attendance", "Attendance outside 0..100"))
        for _, row in hist.iterrows():
            g, gp = grade_for_percentage(float(row["final_percentage"]))
            if row["final_grade"] != g or abs(float(row["gpa_points"]) - gp) > 0.001:
                issues.append(("student_course_history", "grade", "Grade/GPA inconsistent with final percentage"))
                break

    perf = frames["student_topic_performance"]
    if not perf.empty:
        if (perf["correct_attempts"] > perf["practice_attempts"]).any():
            issues.append(("student_topic_performance", "attempts", "Correct attempts exceed total attempts"))
        nonzero = perf[perf["practice_attempts"] > 0]
        if not nonzero.empty:
            expected = nonzero["correct_attempts"] / nonzero["practice_attempts"] * 100
            if not np.allclose(expected, nonzero["accuracy_percentage"].astype(float), atol=0.02):
                issues.append(("student_topic_performance", "accuracy", "Topic accuracy formula mismatch"))
        zero = perf[perf["practice_attempts"] == 0]
        if not zero.empty and zero["accuracy_percentage"].notna().any():
            issues.append(("student_topic_performance", "zero_attempts", "Zero-attempt accuracy must be NULL"))

    outcomes = frames["student_academic_outcomes"]
    if not outcomes.empty:
        if not outcomes["final_percentage"].between(0, 100).all():
            issues.append(("student_academic_outcomes", "percentage", "Outcome percentage outside 0..100"))
        if ((outcomes["successful_completion"] == True) & (outcomes["final_percentage"] < 50)).any():
            issues.append(("student_academic_outcomes", "target", "Passing target conflicts with percentage"))
        if ((outcomes["successful_completion"] == False) & (outcomes["final_percentage"] >= 50) & (outcomes["final_grade"] != "F")).any():
            issues.append(("student_academic_outcomes", "target", "Unsuccessful outcome has passing grade"))

    # Unique enrollment and outcome natural keys.
    enr = frames["student_enrollments"]
    if enr.duplicated(["student_id", "course_id", "term_id"]).any():
        issues.append(("student_enrollments", "duplicate_natural_key", "Duplicate student-course-term"))
    if outcomes.duplicated(["student_id", "course_id", "term_id"]).any():
        issues.append(("student_academic_outcomes", "duplicate_natural_key", "Duplicate student-course-term outcomes"))
    if frames["student_assessment_results"].duplicated(["student_id", "assessment_id"]).any():
        issues.append(("student_assessment_results", "duplicate_natural_key", "Duplicate student-assessment results"))

    # Sessions and logs: enforce same student/course and activity timestamp bounds.
    sessions = frames["learning_sessions"].set_index("session_id", drop=False)
    logs = frames["learning_activity_logs"]
    if not logs.empty:
        session_check = logs.merge(
            frames["learning_sessions"][["session_id", "student_id", "course_id", "session_start", "session_end"]],
            on="session_id", suffixes=("", "_session"), how="left"
        )
        inconsistent = ((session_check["student_id"] != session_check["student_id_session"]) |
                        (session_check["course_id"] != session_check["course_id_session"]))
        if inconsistent.any():
            issues.append(("learning_activity_logs", "session_consistency", "Activity does not match its session"))
        activity_ts = pd.to_datetime(session_check["activity_timestamp"])
        starts = pd.to_datetime(session_check["session_start"])
        ends = pd.to_datetime(session_check["session_end"])
        if ((activity_ts < starts) | (activity_ts > ends)).any():
            issues.append(("learning_activity_logs", "timestamp", "Activity timestamp outside session"))

    # Outcomes and history must agree exactly.
    if not hist.empty and not outcomes.empty:
        merged = hist.merge(outcomes, on=["student_id", "course_id", "term_id"], suffixes=("_h", "_o"))
        if not merged.empty:
            bad = (
                (merged["final_grade_h"] != merged["final_grade_o"]) |
                (np.abs(merged["final_percentage_h"] - merged["final_percentage_o"]) > 0.011) |
                (merged["completion_status"] == "Completed") != merged["successful_completion"]
            )
            if bad.any():
                issues.append(("student_academic_outcomes", "history_alignment", "History and outcome values disagree"))

    # Stats.
    def mean_or_nan(series):
        return round(float(series.mean()), 3) if len(series.dropna()) else None
    stats.extend([
        ("students", len(frames["students"])),
        ("courses", len(frames["courses"])),
        ("course_enrollments", len(enr)),
        ("course_completion_rate", round(float(outcomes["successful_completion"].mean()), 4) if len(outcomes) else None),
        ("course_failure_or_withdrawal_rate", round(float(1 - outcomes["successful_completion"].mean()), 4) if len(outcomes) else None),
        ("withdrawal_rate", round(float((hist["completion_status"] == "Withdrawn").mean()), 4) if len(hist) else None),
        ("average_assessment_percentage", mean_or_nan(submitted["percentage"])),
        ("average_attendance", mean_or_nan(hist["attendance_percentage"])),
        ("average_topic_accuracy", mean_or_nan(perf["accuracy_percentage"])),
        ("average_session_duration_minutes", mean_or_nan(frames["learning_sessions"]["duration_minutes"])),
        ("resource_interaction_completion_rate", round(float((frames["student_resource_interactions"]["completion_percentage"] >= 90).mean()), 4) if len(frames["student_resource_interactions"]) else None),
    ])
    return issues, stats


def main():
    parser = argparse.ArgumentParser(description="Generate and validate the Federated Learning for Personalization synthetic datasets.")
    parser.add_argument("--output", default="federated_learning_dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed (default: 42)")
    parser.add_argument("--students-per-institution", type=int, default=500, help="Students generated per institution (default: 500)")
    args = parser.parse_args()
    if args.students_per_institution < 1:
        parser.error("--students-per-institution must be at least 1")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Generating datasets in: {output}")
    print(f"Seed: {args.seed}; students per institution: {args.students_per_institution}")

    results = []
    for i in range(3):
        print(f"Generating Institution {chr(65+i)}...")
        results.append(make_institution(i, args.students_per_institution, args.seed, output))

    write_documentation(results, output, args.seed, args.students_per_institution)

    all_issues = []
    quality_rows = []
    for result in results:
        issues, stats = validate_institution(result)
        for table, check, detail in issues:
            all_issues.append({
                "institution": result["institution_name"], "table_name": table,
                "check": check, "details": detail,
            })
        for metric, value in stats:
            quality_rows.append({
                "institution": result["institution_name"], "metric": metric, "value": value,
            })
        # Per-table row counts for acceptance reporting.
        for table, df in result["frames"].items():
            quality_rows.append({
                "institution": result["institution_name"], "metric": f"row_count:{table}", "value": len(df),
            })

    docs = output / "shared_documentation"
    pd.DataFrame(quality_rows).to_csv(docs / "data_quality_report.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_issues, columns=["institution", "table_name", "check", "details"]).to_csv(
        docs / "validation_failures.csv", index=False, encoding="utf-8"
    )

    summary = {
        "seed": args.seed,
        "students_per_institution": args.students_per_institution,
        "institution_count": 3,
        "tables_per_institution": len(TABLE_COLUMNS),
        "csv_count_expected": 3 * len(TABLE_COLUMNS),
        "validation_failure_count": len(all_issues),
        "validation_passed": len(all_issues) == 0,
        "row_counts": {
            result["institution_name"]: {
                table: len(df) for table, df in result["frames"].items()
            } for result in results
        },
    }
    (docs / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nGeneration summary")
    print("------------------")
    for result in results:
        print(f"{result['institution_name']}:")
        for table, df in result["frames"].items():
            print(f"  {table:34s} {len(df):>8,}")
    print(f"\nValidation failures: {len(all_issues)}")
    print(f"Validation report: {docs / 'validation_failures.csv'}")
    print(f"Data quality report: {docs / 'data_quality_report.csv'}")
    if all_issues:
        print("WARNING: validation did not fully pass. Review validation_failures.csv.")
        raise SystemExit(2)
    print("All implemented validation checks passed.")
    print("Note: review the generated documentation for target/feature caveats before model training.")


if __name__ == "__main__":
    main()
