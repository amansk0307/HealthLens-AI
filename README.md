# HealthLens-AI

## Health Literacy Tutor

The Health Literacy Tutor creates a three-step learning path from HealthLens's
reviewed local knowledge base. Groq prepares short lessons and knowledge checks;
the tutor uses learner answers to adapt follow-up explanations and tracks progress
for the current Streamlit session. Lesson sources are shown as optional hint cards.
Learners earn points for each completed check and can watch their progress through
the three-step quest.

The tutor and Vitalis AI chatbot send prompts to Groq's API. Configure a
Groq API key before using either feature:

- Set `GROQ_API_KEY` in the environment, or
- Add `GROQ_API_KEY = "your-key"` to `.streamlit/secrets.toml`.

Do not submit names, identifiers, or personal health information. HealthLens is
an educational demonstration, not a medical service or diagnostic tool.