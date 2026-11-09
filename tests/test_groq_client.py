import unittest
from unittest.mock import Mock, patch

from code.groq_client import (
    GROQ_API_URL,
    GroqAPIError,
    generate_groq_response,
    normalize_generated_text,
    stream_groq_response,
)


class GroqClientTests(unittest.TestCase):
    @patch("code.groq_client.requests.post")
    def test_generate_response_uses_groq_and_extracts_text(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "  Hello from Groq  "}}]
        }
        post.return_value = response

        result = generate_groq_response(
            [{"role": "user", "content": "Hi"}], "test-key"
        )

        self.assertEqual(result, "Hello from Groq")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], GROQ_API_URL)
        self.assertEqual(
            post.call_args.kwargs["json"]["model"], "openai/gpt-oss-120b"
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )

    @patch("code.groq_client.requests.post")
    def test_json_mode_requests_json_response(self, post):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": '{"steps": []}'}}]
        }
        post.return_value = response

        generate_groq_response([], "test-key", json_mode=True)

        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"],
            {"type": "json_object"},
        )

    @patch("code.groq_client.requests.post")
    def test_invalid_response_raises_explicit_error(self, post):
        response = Mock()
        response.json.return_value = {}
        post.return_value = response

        with self.assertRaisesRegex(GroqAPIError, "invalid chat completion"):
            generate_groq_response([], "test-key")

    def test_normalizes_mojibake_and_duplicate_citations(self):
        result = normalize_generated_text(
            "Commonâ\u0080¯signs include â\u0080\u0099fatigueâ\u0080\u0099 [S1][S1]."
        )

        self.assertEqual(
            result,
            "Common\u202fsigns include \u2019fatigue\u2019 [S1].",
        )

    def test_normalizes_cp1252_mojibake(self):
        self.assertEqual(
            normalize_generated_text("Itâ€™s relevant."),
            "It\u2019s relevant.",
        )

    @patch("code.groq_client.requests.post")
    def test_stream_yields_text_deltas(self, post):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            'data: {"choices":[{"delta":{"content":"there"}}]}',
            "data: [DONE]",
        ]
        post.return_value = response

        self.assertEqual(
            list(stream_groq_response([], "test-key")),
            ["Hello ", "there"],
        )


if __name__ == "__main__":
    unittest.main()
