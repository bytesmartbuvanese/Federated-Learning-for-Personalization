# Synthetic data generation report

- Random seed: 42
- Students per institution: 500
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
