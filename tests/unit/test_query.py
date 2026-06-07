"""
Unit tests for lambdas/query/handler.py
"""
from __future__ import annotations
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ── Passthrough decorator ─────────────────────────────────────────────────────
def _passthrough(*args, **kwargs):
    def decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator

def _make_powertools_mock():
    m = MagicMock()
    m.inject_lambda_context = _passthrough
    m.capture_lambda_handler = _passthrough
    m.capture_method = _passthrough
    m.log_metrics = _passthrough
    return m

# ── Stubs ─────────────────────────────────────────────────────────────────────
def _stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

plt = _stub("aws_lambda_powertools")
plt.Logger  = lambda **kw: _make_powertools_mock()
plt.Tracer  = lambda **kw: _make_powertools_mock()
plt.Metrics = lambda **kw: _make_powertools_mock()
_stub("aws_lambda_powertools.metrics").MetricUnit = MagicMock()
_stub("aws_lambda_powertools.utilities")
_stub("aws_lambda_powertools.utilities.typing").LambdaContext = object

ospy = _stub("opensearchpy")
ospy.OpenSearch             = MagicMock()
ospy.RequestsHttpConnection = MagicMock()
ospy.AWSV4SignerAuth        = MagicMock()

import os
os.environ["COLLECTION_ENDPOINT_PARAM"] = "/ask-my-docs/collection-endpoint"
os.environ["INDEX_NAME_PARAM"]          = "/ask-my-docs/index-name"
os.environ["REGION_PARAM"]              = "/ask-my-docs/region"
os.environ["POWERTOOLS_SERVICE_NAME"]   = "test"
os.environ["POWERTOOLS_LOG_LEVEL"]      = "DEBUG"

sys.path.insert(0, "lambdas/query")
if "handler" in sys.modules:
    del sys.modules["handler"]
import handler as query_module


class TestSanitizeQuestion(unittest.TestCase):

    def test_strips_html_tags(self):
        result = query_module.sanitize_question("<script>alert('xss')</script>What is AI?")
        self.assertNotIn("<script>", result)
        self.assertIn("What is AI", result)

    def test_strips_control_characters(self):
        result = query_module.sanitize_question("normal\x00\x01question")
        self.assertNotIn("\x00", result)

    def test_enforces_max_length(self):
        result = query_module.sanitize_question("a" * 1000)
        self.assertLessEqual(len(result), query_module.QUESTION_MAX_LEN)

    def test_preserves_normal_question(self):
        raw = "What are the main findings of the report?"
        self.assertEqual(query_module.sanitize_question(raw), raw)

    def test_empty_string(self):
        self.assertEqual(query_module.sanitize_question(""), "")

    def test_collapses_whitespace(self):
        self.assertEqual(query_module.sanitize_question("What   is    the   answer?"), "What is the answer?")


class TestBuildContextBlock(unittest.TestCase):

    def _chunks(self, n=3):
        return [
            {
                "text":         f"This is chunk {i} with relevant content.",
                "source":       "document.pdf",
                "page_numbers": [i + 1],
                "doc_id":       "abc",
                "chunk_index":  i,
            }
            for i in range(n)
        ]

    def test_context_contains_all_chunks(self):
        context = query_module.build_context_block(self._chunks(3))
        for i in range(3):
            self.assertIn(f"chunk {i}", context)

    def test_context_within_char_limit(self):
        context = query_module.build_context_block(self._chunks(100))
        self.assertLessEqual(len(context), query_module.MAX_CONTEXT_CHARS + 500)

    def test_empty_chunks(self):
        self.assertEqual(query_module.build_context_block([]), "")

    def test_source_and_page_in_header(self):
        context = query_module.build_context_block(self._chunks(1))
        self.assertIn("document.pdf", context)
        self.assertIn("Pages:", context)


class TestQueryHandler(unittest.TestCase):

    def _api_event(self, body):
        return {
            "httpMethod": "POST",
            "body":       json.dumps(body),
            "headers":    {"Content-Type": "application/json"},
        }

    def _mock_os(self, hits=None):
        if hits is None:
            hits = [{"_source": {
                "text": "The answer is 42.",
                "source": "report.pdf",
                "page_numbers": [7],
                "doc_id": "abc",
                "chunk_index": 0,
            }}]
        mock_os = MagicMock()
        mock_os.search.return_value = {"hits": {"hits": hits}}
        return mock_os

    def _bedrock_resp(self, text="This is the answer."):
        return {"content": [{"text": text}], "usage": {"input_tokens": 100, "output_tokens": 50}}

    @patch.object(query_module, "_build_os_client")
    @patch.object(query_module, "embed_question", return_value=[0.1] * 1536)
    @patch.object(query_module, "generate_answer")
    @patch.object(query_module, "_get_param", return_value="https://endpoint")
    def test_successful_query(self, mp, mg, me, mo):
        mo.return_value = self._mock_os()
        mg.return_value = self._bedrock_resp()
        result = query_module.handler(self._api_event({"question": "What is the answer?"}), MagicMock())
        self.assertEqual(result["statusCode"], 200)
        body = json.loads(result["body"])
        self.assertIn("answer", body)
        self.assertIn("sources", body)

    def test_missing_question_returns_400(self):
        result = query_module.handler(self._api_event({}), MagicMock())
        self.assertEqual(result["statusCode"], 400)

    def test_invalid_json_returns_400(self):
        result = query_module.handler({"body": "not-json{{{"}, MagicMock())
        self.assertEqual(result["statusCode"], 400)

    def test_empty_question_after_sanitize_returns_400(self):
        result = query_module.handler(self._api_event({"question": "\x00\x01\x02"}), MagicMock())
        self.assertEqual(result["statusCode"], 400)

    @patch.object(query_module, "_build_os_client")
    @patch.object(query_module, "embed_question", return_value=[0.1] * 1536)
    @patch.object(query_module, "_get_param", return_value="https://endpoint")
    def test_no_chunks_returns_helpful_message(self, mp, me, mo):
        mock_os = MagicMock()
        mock_os.search.return_value = {"hits": {"hits": []}}
        mo.return_value = mock_os
        result = query_module.handler(self._api_event({"question": "Any question"}), MagicMock())
        self.assertEqual(result["statusCode"], 200)
        body = json.loads(result["body"])
        self.assertIn("upload", body["answer"].lower())

    @patch.object(query_module, "_build_os_client")
    @patch.object(query_module, "embed_question", side_effect=Exception("Bedrock throttle"))
    @patch.object(query_module, "_get_param", return_value="https://endpoint")
    def test_bedrock_error_returns_500(self, mp, me, mo):
        mo.return_value = self._mock_os()
        result = query_module.handler(self._api_event({"question": "Question?"}), MagicMock())
        self.assertEqual(result["statusCode"], 500)

    @patch.object(query_module, "_build_os_client")
    @patch.object(query_module, "embed_question", return_value=[0.1] * 1536)
    @patch.object(query_module, "generate_answer")
    @patch.object(query_module, "_get_param", return_value="https://endpoint")
    def test_top_k_capped_at_20(self, mp, mg, me, mo):
        mock_os = self._mock_os()
        mo.return_value = mock_os
        mg.return_value = self._bedrock_resp()
        query_module.handler(self._api_event({"question": "Q?", "top_k": 9999}), MagicMock())
        call_body = mock_os.search.call_args[1]["body"]
        knn_k = call_body["query"]["knn"]["embedding"]["k"]
        self.assertLessEqual(knn_k, 20)

    def test_security_headers_present(self):
        result = query_module.handler(self._api_event({}), MagicMock())
        self.assertIn("X-Content-Type-Options", result.get("headers", {}))


class TestSystemPromptIntegrity(unittest.TestCase):

    def test_system_prompt_contains_anchor_instructions(self):
        sp = query_module.SYSTEM_PROMPT
        self.assertIn("ONLY", sp.upper())
        self.assertIn("NEVER", sp.upper())

    def test_user_input_not_in_system_prompt(self):
        captured = []

        def capture_invoke(**kwargs):
            captured.append(json.loads(kwargs["body"]))
            return {"body": MagicMock(read=MagicMock(return_value=json.dumps({
                "content": [{"text": "answer"}],
                "usage":   {"input_tokens": 1, "output_tokens": 1},
            }).encode()))}

        with patch.object(query_module, "bedrock_client") as mb:
            mb.invoke_model.side_effect = capture_invoke
            query_module.generate_answer("TEST_QUESTION_MARKER", "some context")

        self.assertTrue(len(captured) > 0)
        body = captured[0]
        self.assertNotIn("TEST_QUESTION_MARKER", body.get("system", ""))
        self.assertIn("TEST_QUESTION_MARKER", body["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()