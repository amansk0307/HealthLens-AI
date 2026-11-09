import json
import unittest
from unittest.mock import patch

from code.health_tutor import (
    TUTOR_TOPICS,
    TutorResponseError,
    create_learning_plan,
    evaluate_answer,
    retrieve_topic_passages,
)


class HealthTutorTests(unittest.TestCase):
    def test_retrieves_reviewed_passages_for_supported_topics(self):
        for topic in TUTOR_TOPICS:
            with self.subTest(topic=topic):
                passages = retrieve_topic_passages(topic)
                self.assertTrue(passages)
                self.assertTrue(all(passage["source_name"] for passage in passages))

    @patch("code.health_tutor.generate_groq_response")
    def test_learning_plan_assigns_local_citations_without_model_ids(self, generate):
        passages = retrieve_topic_passages("Diabetes")
        generate.return_value = json.dumps(
            {
                "steps": [
                    {
                        "title": f"Lesson {index}",
                        "objective": "Explain a sourced concept.",
                    }
                    for index in range(3)
                ]
            }
        )

        plan = create_learning_plan("Diabetes", passages, "test-key")

        self.assertEqual(len(plan), 3)
        available_citations = {passage["citation"] for passage in passages}
        self.assertTrue(all(step["source_citations"] for step in plan))
        self.assertTrue(
            all(
                citation in available_citations
                for step in plan
                for citation in step["source_citations"]
            )
        )
        self.assertTrue(generate.call_args.kwargs["json_mode"])

    @patch("code.health_tutor.generate_groq_response")
    def test_learning_plan_ignores_model_generated_citations(self, generate):
        passages = retrieve_topic_passages("Diabetes")
        generate.return_value = json.dumps(
            {
                "steps": [
                    {
                        "title": "Lesson",
                        "objective": "Explain a sourced concept.",
                        "source_citations": ["S99"],
                    }
                    for _ in range(3)
                ]
            }
        )

        plan = create_learning_plan("Diabetes", passages, "test-key")

        available_citations = {passage["citation"] for passage in passages}
        self.assertTrue(
            all(
                citation in available_citations
                for step in plan
                for citation in step["source_citations"]
            )
        )

    @patch("code.health_tutor.generate_groq_response")
    def test_answer_feedback_requires_a_boolean_mastery_result(self, generate):
        passages = retrieve_topic_passages("Diabetes")
        step = {
            "objective": "Describe diabetes.",
            "source_citations": [passages[0]["citation"]],
        }
        generate.return_value = json.dumps(
            {"mastered": "yes", "feedback": "Good work."}
        )

        with self.assertRaisesRegex(TutorResponseError, "incomplete answer feedback"):
            evaluate_answer(
                "Diabetes",
                step,
                "Lesson and question",
                "Learner answer",
                passages,
                "test-key",
            )


if __name__ == "__main__":
    unittest.main()
