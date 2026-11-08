import unittest

from code.chatbot_rag import (
    build_memory_loss_information,
    build_retrieval_query,
    build_high_blood_sugar_guidance,
    format_retrieved_context,
    is_high_blood_sugar_action_question,
    is_low_blood_sugar_question,
    is_memory_loss_question,
    retrieve_passages,
    unique_sources,
)


class ChatbotRetrievalTests(unittest.TestCase):
    def test_retrieves_condition_specific_sources(self):
        examples = {
            "What are possible symptoms of Alzheimer's disease?": "Alzheimer",
            "What is diabetes and how is it diagnosed?": "Diabetes",
            "What are the symptoms of pneumonia?": "Pneumonia",
            "How does malaria spread?": "Malaria",
        }
        for question, expected_title in examples.items():
            with self.subTest(question=question):
                passages = retrieve_passages(question)
                self.assertTrue(passages)
                self.assertIn(expected_title, passages[0]["title"])

    def test_returns_no_passages_for_unrelated_question(self):
        self.assertEqual(
            retrieve_passages("Who won the world cup in 1966?"),
            [],
        )

    def test_retrieves_high_blood_sugar_and_emergency_guidance(self):
        question = "My sugar level spiked, what should I do?"
        passages = retrieve_passages(question)
        titles = {passage["title"] for passage in passages}

        self.assertTrue(is_high_blood_sugar_action_question(question))
        self.assertIn("High blood sugar (hyperglycemia): general steps", titles)
        self.assertIn("Diabetic ketoacidosis warning signs and emergency care", titles)
        answer = build_high_blood_sugar_guidance(passages)
        emergency_citation = next(
            passage["citation"]
            for passage in passages
            if passage["title"]
            == "Diabetic ketoacidosis warning signs and emergency care"
        )
        self.assertIn("[S1]", answer)
        self.assertIn(f"[{emergency_citation}]", answer)
        self.assertIn("call your local emergency number", answer)
        self.assertIn("don't take an extra dose", answer)

    def test_high_blood_sugar_action_detection_is_scoped(self):
        self.assertTrue(
            is_high_blood_sugar_action_question("My glucose is high, please help")
        )
        self.assertFalse(
            is_high_blood_sugar_action_question("What is the role of sugar in baking?")
        )
        self.assertFalse(
            is_high_blood_sugar_action_question("What does sugar do in the body?")
        )

    def test_memory_loss_question_retrieves_relevant_sources_and_information(self):
        question = "I have been forgetting a lot lately, what disease have similar symptoms?"
        passages = retrieve_passages(question)
        titles = {passage["title"] for passage in passages}

        self.assertTrue(is_memory_loss_question(question))
        self.assertIn("Forgetfulness and possible related conditions", titles)
        self.assertIn("Memory loss causes and evaluation", titles)
        self.assertNotIn("Malaria overview", titles)
        answer = build_memory_loss_information(passages)
        self.assertIn("Alzheimer's disease and other dementias", answer)
        self.assertIn("Parkinson's disease", answer)
        self.assertIn("not enough to identify a specific disease", answer)
        for citation in {passage["citation"] for passage in passages}:
            self.assertIn(f"[{citation}]", answer)

    def test_detects_brief_memory_concern(self):
        question = "I've been more forgetful recently."
        self.assertTrue(is_memory_loss_question(question))
        titles = {passage["title"] for passage in retrieve_passages(question)}
        self.assertIn("Memory loss causes and evaluation", titles)

    def test_expanded_disease_questions_retrieve_matching_passages(self):
        examples = {
            "What are symptoms of low blood sugar?": (
                "Low blood sugar (hypoglycemia): signs and safety",
                "High blood sugar (hyperglycemia): general steps",
            ),
            "Can diabetes affect other organs?": (
                "Possible long-term complications of diabetes",
                "High blood sugar (hyperglycemia): general steps",
            ),
            "What are warning signs of a heart attack?": (
                "Heart attack warning signs and emergency response",
                "Pneumonia overview",
            ),
            "How is Parkinson disease diagnosed?": (
                "Parkinson's disease evaluation",
                "Malaria overview",
            ),
            "Do antibiotics treat viral pneumonia?": (
                "Pneumonia causes, testing, and treatment",
                "Malaria overview",
            ),
            "How is malaria tested after travel?": (
                "Malaria testing and travel prevention",
                "Parkinson's disease overview",
            ),
        }
        for question, (expected_title, unrelated_title) in examples.items():
            with self.subTest(question=question):
                titles = {passage["title"] for passage in retrieve_passages(question)}
                self.assertIn(expected_title, titles)
                self.assertNotIn(unrelated_title, titles)

    def test_low_blood_sugar_detection(self):
        self.assertTrue(is_low_blood_sugar_question("My blood sugar is low"))
        self.assertTrue(is_low_blood_sugar_question("What is hypoglycemia?"))
        self.assertFalse(is_low_blood_sugar_question("My blood sugar is high"))

    def test_retrieval_only_uses_prior_question_for_clear_follow_up(self):
        previous = "What are possible symptoms of diabetes?"
        self.assertEqual(
            build_retrieval_query("I have been forgetting a lot lately.", previous),
            "I have been forgetting a lot lately.",
        )
        self.assertEqual(
            build_retrieval_query("What about its symptoms?", previous),
            f"{previous} What about its symptoms?",
        )

    def test_context_and_sources_keep_citation_ids(self):
        passages = retrieve_passages("What are possible symptoms of diabetes?")
        context = format_retrieved_context(passages)
        sources = unique_sources(passages)

        self.assertIn("[S1]", context)
        self.assertTrue(sources)
        self.assertEqual(sources[0]["citation"], "S1")
        self.assertTrue(
            all(source["source_url"].startswith("https://") for source in sources)
        )


if __name__ == "__main__":
    unittest.main()
