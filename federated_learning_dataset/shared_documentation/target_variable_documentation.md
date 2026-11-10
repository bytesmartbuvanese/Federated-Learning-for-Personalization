# Target variable documentation

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
