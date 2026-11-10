# Database relationships

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
