"""
Unit tests for lambdas/ingest/handler.py
All AWS calls are mocked - no live resources required.
"""
from __future__ import annotations
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Passthrough decorator ────────────────────────────────────────────────────
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

# ── Stub heavy dependencies before importing handler ─────────────────────────
def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

plt = _make_stub("aws_lambda_powertools")
plt.Logger  = lambda **kw: _make_powertools_mock()
plt.Tracer  = lambda **kw: _make_powertools_mock()
plt.Metrics = lambda **kw: _make_powertools_mock()
_make_stub("aws_lambda_powertools.metrics").MetricUnit = MagicMock()
_make_stub("aws_lambda_powertools.utilities")
_make_stub("aws_lambda_powertools.utilities.typing").LambdaContext = object

ospy = _make_stub("opensearchpy")
ospy.OpenSearch             = MagicMock()
ospy.RequestsHttpConnection = MagicMock()
ospy.AWSV4SignerAuth        = MagicMock()

magic_stub = _make_stub("magic")
magic_stub.from_buffer = MagicMock(return_value="application/pdf")

pypdf_stub = _make_stub("pypdf")
mock_page  = MagicMock()
mock_page.extract_text.return_value = "This is page text. " * 100
mock_reader = MagicMock()
mock_reader.pages = [mock_page, mock_page]
pypdf_stub.PdfReader = MagicMock(return_value=mock_reader)

import os
os.environ["MAX_FILE_SIZE_MB"]          = "50"
os.environ["COLLECTION_ENDPOINT_PARAM"] = "/ask-my-docs/collection-endpoint"
os.environ["INDEX_NAME_PARAM"]          = "/ask-my-docs/index-name"
os.environ["REGION_PARAM"]              = "/ask-my-docs/region"
os.environ["POWERTOOLS_SERVICE_NAME"]   = "test"
os.environ["POWERTOOLS_LOG_LEVEL"]      = "DEBUG"

sys.path.insert(0, "lambdas/ingest")
import handler as ingest_module


class TestSanitizeAndValidate(unittest.TestCase):

    def test_validate_pdf_rejects_oversized(self):
        large_body = b"%" * (55 * 1024 * 1024)
        with patch.object(ingest_module, "s3_client") as mock_s3:
            with self.assertRaises(ValueError) as ctx:
                ingest_module.validate_pdf("bucket", "key.pdf", large_body)
        self.assertIn("size limit", str(ctx.exception))

    def test_validate_pdf_rejects_non_pdf(self):
        magic_stub.from_buffer.return_value = "image/jpeg"
        with patch.object(ingest_module, "s3_client") as mock_s3:
            with self.assertRaises(ValueError) as ctx:
                ingest_module.validate_pdf("bucket", "key.pdf", b"fake jpeg" * 100)
        self.assertIn("non-PDF", str(ctx.exception))
        magic_stub.from_buffer.return_value = "application/pdf"

    def test_validate_pdf_accepts_valid(self):
        magic_stub.from_buffer.return_value = "application/pdf"
        with patch.object(ingest_module, "s3_client"):
            ingest_module.validate_pdf("bucket", "key.pdf", b"%PDF-1.4" * 100)


class TestChunking(unittest.TestCase):

    def _make_pages(self, n_pages=3, words_per_page=300):
        word = "information "
        return [(word * words_per_page, i + 1) for i in range(n_pages)]

    def test_chunks_are_created(self):
        pages = self._make_pages()
        chunks = ingest_module.chunk_pages(pages, doc_id="abc123", source="test.pdf")
        self.assertGreater(len(chunks), 0)

    def test_chunk_size_respected(self):
        pages = self._make_pages(n_pages=5, words_per_page=500)
        chunks = ingest_module.chunk_pages(pages, doc_id="abc123", source="test.pdf")
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), ingest_module.CHUNK_SIZE_CHARS + 50)

    def test_page_numbers_preserved(self):
        pages = [("content on page one", 1), ("content on page two", 2)]
        chunks = ingest_module.chunk_pages(pages, doc_id="id", source="s")
        for chunk in chunks:
            self.assertTrue(len(chunk.page_numbers) > 0)

    def test_empty_pages_returns_no_chunks(self):
        chunks = ingest_module.chunk_pages([], doc_id="id", source="s")
        self.assertEqual(chunks, [])

    def test_doc_id_preserved(self):
        pages = self._make_pages()
        chunks = ingest_module.chunk_pages(pages, doc_id="TESTID", source="file.pdf")
        for chunk in chunks:
            self.assertEqual(chunk.doc_id, "TESTID")

    def test_chunk_index_sequential(self):
        pages = self._make_pages(n_pages=5, words_per_page=400)
        chunks = ingest_module.chunk_pages(pages, doc_id="id", source="s")
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, i)


class TestIndexChunks(unittest.TestCase):

    def _make_chunks(self, n=3):
        Chunk = ingest_module.Chunk
        return [Chunk(text=f"chunk {i}", chunk_index=i, doc_id="d", source="s") for i in range(n)]

    def test_bulk_called_with_correct_structure(self):
        mock_os = MagicMock()
        mock_os.bulk.return_value = {"errors": False, "items": []}
        chunks     = self._make_chunks(3)
        embeddings = [[0.0] * 1536 for _ in chunks]
        ingest_module.index_chunks(mock_os, chunks, embeddings)
        mock_os.bulk.assert_called_once()

    def test_bulk_errors_raise(self):
        mock_os = MagicMock()
        mock_os.bulk.return_value = {
            "errors": True,
            "items": [{"index": {"error": {"reason": "mapping conflict"}}}],
        }
        with self.assertRaises(RuntimeError):
            ingest_module.index_chunks(mock_os, self._make_chunks(1), [[0.0] * 1536])

    def test_empty_chunks_no_call(self):
        mock_os = MagicMock()
        ingest_module.index_chunks(mock_os, [], [])
        mock_os.bulk.assert_not_called()


class TestHandlerSQSEvent(unittest.TestCase):

    def _sqs_event(self, bucket="my-bucket", key="doc.pdf"):
        return {
            "Records": [{
                "body": json.dumps({
                    "Records": [{
                        "s3": {
                            "bucket": {"name": bucket},
                            "object": {"key": key},
                        }
                    }]
                })
            }]
        }

    @patch.object(ingest_module, "_build_os_client")
    @patch.object(ingest_module, "s3_client")
    @patch.object(ingest_module, "embed_text", return_value=[0.1] * 1536)
    @patch.object(ingest_module, "validate_pdf")
    def test_handler_success_path(self, mock_validate, mock_embed, mock_s3, mock_os_builder):
        mock_os = MagicMock()
        mock_os.bulk.return_value = {"errors": False, "items": []}
        mock_os_builder.return_value = mock_os
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"%PDF-1.4" + b"x" * 1000))
        }
        result = ingest_module.handler(self._sqs_event(), MagicMock())
        self.assertEqual(result["statusCode"], 200)

    @patch.object(ingest_module, "_build_os_client")
    @patch.object(ingest_module, "s3_client")
    @patch.object(ingest_module, "validate_pdf", side_effect=ValueError("bad mime"))
    def test_handler_validation_failure_does_not_raise(self, mock_validate, mock_s3, mock_os):
        result = ingest_module.handler(self._sqs_event(), MagicMock())
        self.assertEqual(result["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()