"""Local lexical retrieval for the HealthLens educational chatbot."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "chatbot_knowledge.json"
)
MIN_RELEVANCE_SCORE = 0.10
TOPIC_TERMS = {
    "alzheimers": {"alzheimer", "alzheimers", "dementia"},
    "diabetes": {
        "diabetes",
        "diabetic",
        "glucose",
        "hypoglycemia",
        "sugar",
        "hyperglycemia",
        "glycemic",
        "ketone",
        "ketones",
    },
    "heart": {"heart", "cardiac", "coronary", "cardiovascular"},
    "liver": {"liver", "hepatic", "jaundice"},
    "malaria": {"malaria"},
    "memory": {
        "forget",
        "forgetful",
        "forgetting",
        "forgot",
        "memory",
        "memories",
        "remember",
        "remembering",
        "cognitive",
    },
    "parkinsons": {"parkinson", "parkinsons"},
    "pneumonia": {"pneumonia"},
}
MEMORY_LOSS_DOCUMENT_IDS = (
    "memory-forgetfulness-overview",
    "memory-loss-causes",
)
HIGH_BLOOD_SUGAR_ACTION_TERMS = {
    "manage",
    "lower",
    "reduce",
    "bring",
    "action",
    "steps",
    "need",
}
LOW_BLOOD_SUGAR_DOCUMENT_ID = "diabetes-low-blood-sugar"
DIABETES_COMPLICATIONS_DOCUMENT_ID = "diabetes-long-term-complications"


def build_retrieval_query(question: str, previous_question: str = "") -> str:
    """Include prior context only when the new question clearly refers to it."""
    refers_to_previous_context = bool(
        re.search(
            r"\b(it|that|this|they|those|them|their|there|same)\b|"
            r"\b(what about|how about|tell me more)\b",
            question.lower(),
        )
    )
    if previous_question.strip() and refers_to_previous_context:
        return f"{previous_question.strip()} {question.strip()}"
    return question.strip()


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[dict[str, str]], TfidfVectorizer, Any]:
    with KNOWLEDGE_PATH.open(encoding="utf-8") as knowledge_file:
        documents = json.load(knowledge_file)
    if not documents or any(
        not all(document.get(field) for field in ("id", "title", "text", "source_name"))
        for document in documents
    ):
        raise ValueError(f"Knowledge base contains an incomplete entry: {KNOWLEDGE_PATH}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    document_vectors = vectorizer.fit_transform(
        document["title"] + " " + document["text"] for document in documents
    )
    return documents, vectorizer, document_vectors


def _query_topics(question: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return {
        topic
        for topic, terms in TOPIC_TERMS.items()
        if words.intersection(terms)
    }


def is_high_blood_sugar_action_question(question: str) -> bool:
    """Identify personal high-glucose questions that need immediate guidance."""
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    mentions_glucose = bool(
        words.intersection({"sugar", "glucose", "glycemic", "hyperglycemia"})
    )
    describes_high_reading = bool(
        words.intersection(
            {"high", "higher", "spike", "spiked", "spiking", "elevated", "rising"}
        )
    )
    asks_for_action = bool(words.intersection(HIGH_BLOOD_SUGAR_ACTION_TERMS)) or bool(
        re.search(
            r"\b(what should i do|what do i do|what can i do|"
            r"how can i|how do i|what steps|help me)\b",
            question.lower(),
        )
    )
    return mentions_glucose and (describes_high_reading or asks_for_action)


def is_low_blood_sugar_question(question: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    mentions_glucose = bool(
        words.intersection({"sugar", "glucose", "glycemic", "hypoglycemia"})
    )
    describes_low_reading = bool(
        words.intersection({"low", "lower", "hypo", "hypoglycemia"})
    )
    return mentions_glucose and describes_low_reading


def is_diabetes_complications_question(question: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    asks_about_effects = bool(
        words.intersection(
            {"affect", "affects", "complication", "complications", "effects", "organs"}
        )
    )
    return "diabetes" in _query_topics(question) and asks_about_effects


def is_memory_loss_question(question: str) -> bool:
    """Identify questions about forgetfulness or possible memory-loss causes."""
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    asks_about_memory = bool(words.intersection(TOPIC_TERMS["memory"]))
    asks_about_a_concern = bool(
        words.intersection(
            {
                "cause",
                "causes",
                "disease",
                "diseases",
                "change",
                "changes",
                "concern",
                "concerned",
                "condition",
                "forget",
                "forgetful",
                "forgetting",
                "forgot",
                "issue",
                "issues",
                "loss",
                "lately",
                "problem",
                "problems",
                "recently",
                "symptom",
                "symptoms",
                "worse",
                "worsening",
                "why",
            }
        )
    )
    return asks_about_memory and asks_about_a_concern


def retrieve_passages(
    question: str, top_k: int = 4, min_score: float = MIN_RELEVANCE_SCORE
) -> list[dict[str, str]]:
    """Return relevant local passages with stable citation identifiers."""
    normalized_question = question.strip()
    if not normalized_question:
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    documents, vectorizer, document_vectors = _load_index()
    query_vector = vectorizer.transform([normalized_question])
    scores = cosine_similarity(query_vector, document_vectors).ravel()
    ranked_indices = scores.argsort()[::-1]
    query_topics = _query_topics(normalized_question)

    selected: list[dict[str, str]] = []
    source_ids: dict[str, str] = {}
    source_counts: dict[str, int] = {}
    for index in ranked_indices:
        score = float(scores[index])
        if score < min_score or len(selected) >= top_k:
            break
        document = documents[int(index)]
        document_topic = document["id"].split("-", 1)[0]
        is_mismatched_glucose_document = (
            is_low_blood_sugar_question(normalized_question)
            and document["id"]
            in {
                "diabetes-high-blood-sugar",
                "diabetes-high-blood-sugar-ketones",
                "diabetes-ketoacidosis-emergency",
            }
        )
        is_memory_related_alzheimer_document = (
            "memory" in query_topics and document_topic == "alzheimers"
        )
        if is_mismatched_glucose_document:
            continue
        if (
            query_topics
            and document_topic != "healthlens"
            and document_topic not in query_topics
            and not is_memory_related_alzheimer_document
        ):
            continue
        source_url = document.get("source_url", "")
        source_key = source_url or document["source_name"]
        if source_counts.get(source_key, 0) >= 2:
            continue
        citation = source_ids.setdefault(source_key, f"S{len(source_ids) + 1}")
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected.append(
            {
                "citation": citation,
                "title": document["title"],
                "source_name": document["source_name"],
                "source_url": source_url,
                "text": document["text"],
                "reviewed_on": document.get("reviewed_on", ""),
            }
        )

    if is_high_blood_sugar_action_question(normalized_question):
        for required_id in (
            "diabetes-high-blood-sugar",
            "diabetes-ketoacidosis-emergency",
        ):
            required_document = next(
                document for document in documents if document["id"] == required_id
            )
            if any(
                passage["title"] == required_document["title"]
                for passage in selected
            ):
                continue
            source_key = required_document["source_url"]
            citation = source_ids.setdefault(source_key, f"S{len(source_ids) + 1}")
            selected.append(
                {
                    "citation": citation,
                    "title": required_document["title"],
                    "source_name": required_document["source_name"],
                    "source_url": source_key,
                    "text": required_document["text"],
                    "reviewed_on": required_document.get("reviewed_on", ""),
                }
            )

    for query_match, required_id in (
        (is_low_blood_sugar_question(normalized_question), LOW_BLOOD_SUGAR_DOCUMENT_ID),
        (
            is_diabetes_complications_question(normalized_question),
            DIABETES_COMPLICATIONS_DOCUMENT_ID,
        ),
    ):
        if not query_match:
            continue
        required_document = next(
            document for document in documents if document["id"] == required_id
        )
        if any(passage["title"] == required_document["title"] for passage in selected):
            continue
        source_key = required_document["source_url"]
        citation = source_ids.setdefault(source_key, f"S{len(source_ids) + 1}")
        selected.append(
            {
                "citation": citation,
                "title": required_document["title"],
                "source_name": required_document["source_name"],
                "source_url": source_key,
                "text": required_document["text"],
                "reviewed_on": required_document.get("reviewed_on", ""),
            }
        )

    if is_memory_loss_question(normalized_question):
        for required_id in MEMORY_LOSS_DOCUMENT_IDS:
            required_document = next(
                document for document in documents if document["id"] == required_id
            )
            if any(
                passage["title"] == required_document["title"]
                for passage in selected
            ):
                continue
            source_key = required_document["source_url"]
            citation = source_ids.setdefault(source_key, f"S{len(source_ids) + 1}")
            selected.append(
                {
                    "citation": citation,
                    "title": required_document["title"],
                    "source_name": required_document["source_name"],
                    "source_url": source_key,
                    "text": required_document["text"],
                    "reviewed_on": required_document.get("reviewed_on", ""),
                }
            )
    return selected


def format_retrieved_context(passages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{passage['citation']}] {passage['title']}\n{passage['text']}"
        for passage in passages
    )


def build_high_blood_sugar_guidance(passages: list[dict[str, str]]) -> str:
    """Build reviewed, non-personalized guidance with the retrieved citations."""
    passage_citations = {
        passage["title"]: passage["citation"] for passage in passages
    }
    self_care_citation = passage_citations[
        "High blood sugar (hyperglycemia): general steps"
    ]
    emergency_citation = passage_citations[
        "Diabetic ketoacidosis warning signs and emergency care"
    ]
    return (
        "A high reading can have different causes, and I can't assess your "
        "individual situation. If you have a high-glucose or sick-day plan from "
        "your diabetes care team, follow that plan and check your glucose as "
        "they instructed. Take medicine only as prescribed; don't take an extra "
        "dose unless your own prescribed plan specifically tells you to. "
        "Contact your care team if the reading stays above your target or you "
        f"aren't sure what caused it. [{self_care_citation}]\n\n"
        "If your plan says to check ketones, follow its instructions; ketones "
        "need prompt contact with a clinician. Confusion or reduced "
        "consciousness, fruity-smelling breath, vomiting, or trouble breathing "
        "can be emergency warning signs—call your local emergency number or go "
        f"to an emergency department now. [{emergency_citation}]"
    )


def build_memory_loss_information(passages: list[dict[str, str]]) -> str:
    """Summarize sourced possibilities without suggesting a diagnosis."""
    passage_citations = {
        passage["title"]: passage["citation"] for passage in passages
    }
    overview_citation = passage_citations["Forgetfulness and possible related conditions"]
    causes_citation = passage_citations["Memory loss causes and evaluation"]
    return (
        "Forgetfulness is a symptom, not enough to identify a specific disease. "
        "Possible associated conditions include Alzheimer's disease and other "
        "dementias, depression, stroke, head injury, thyroid, kidney, or liver "
        "problems, and reactions to some medicines. Other possible causes "
        "include Parkinson's disease, multiple sclerosis, infections, epilepsy, "
        "alcohol or drug effects, and low levels of vitamins such as B1 or B12. "
        f"[{overview_citation}] [{causes_citation}]\n\n"
        "These are possibilities, not a diagnosis, and many other causes may "
        "exist. If this is a change you have noticed, a health care provider can "
        "evaluate it; when it began and how it affects daily activities are "
        f"useful details to discuss. [{causes_citation}]"
    )


def unique_sources(passages: list[dict[str, str]]) -> list[dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for passage in passages:
        key = passage["source_url"] or passage["source_name"]
        source = sources.setdefault(
            key,
            {
                "citation": passage["citation"],
                "source_name": passage["source_name"],
                "source_url": passage["source_url"],
                "reviewed_on": passage["reviewed_on"],
            },
        )
    return list(sources.values())
