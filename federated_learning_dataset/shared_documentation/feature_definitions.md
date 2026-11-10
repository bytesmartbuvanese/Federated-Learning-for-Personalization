# Feature definitions

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
