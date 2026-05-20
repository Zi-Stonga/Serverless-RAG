"""
Unit tests for lambdas/ingest/handler.py
All AWS calls are mocked – no live resources required.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch, call


# ── Stub heavy dependencies before importing handler ────────────────────────

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# aws_lambda_powertools stubs
plt = _make_stub("aws_lambda_powertools")
plt.Logger   = lambda **kw: MagicMock()
plt.Tracer   = lambda **kw: MagicMock()
plt.Metrics  = lambda **kw: MagicMock()

_make_stub("aws_lambda_powertools.metrics").MetricUnit = MagicMock()
_make_stub("aws_lambda_powertools.utilities")
_make_stub("aws_lambda_powertools.utilities.typing").LambdaContext = object

# opensearchpy stubs
ospy = _make_stub("opensearchpy")
ospy.OpenSearch              = MagicMock()
ospy.RequestsHttpConnection  = MagicMock()
ospy.AWSV4SignerAuth         = MagicMock()

# magic stub
magic_stub = _make_stub("magic")
magic_stub.from_buffer = MagicMock(return_value="application/pdf")

# pypdf stub
pypdf_stub          = _make_stub("pypdf")
mock_page           = MagicMock()
mock_page.extract_text.return_value = "This is page text. " * 100
mock_reader         = MagicMock()
mock_reader.pages   = [mock_page, mock_page]
pypdf_stub.PdfReader = MagicMock(return_value=mock_reader)

import importlib
import os
os.environ.setdefault("MAX_FILE_SIZE_MB",           "50")
os.environ.setdefault("COLLECTION_ENDPOINT_PARAM",  "/ask-my-docs/collection-endpoint")
os.environ.setdefault("INDEX_NAME_PARAM",           "/ask-my-docs/index-name")
os.environ.setdefault("REGION_PARAM",               "/ask-my-docs/region")
os.environ.setdefault("POWERTOOLS_SERVICE_NAME",    "test")
os.environ.setdefault("POWERTOOLS_LOG_LEVEL",       "DEBUG")

# Now import handler
sys.path.insert(0, str(__file__ + "/../../lambdas/ingest").replace("tests/unit/test_ingest.py", "lambdas/ingest"))
from lambdas.ingest import handler as ingest_module  # type: ignore


class TestSanitizeAndValidate(unittest.TestCase):

    def test_validate_pdf_rejects_oversized(self):
        large_body = b"%" * (55 * 1024 * 1024)
        with self.assertRaises(ValueError) as ctx:
            with patch.object(ingest_module, "s3_client") as mock_s3:
                ingest_module.validate_pdf("bucket", "key.pdf", large_body)
        self.assertIn("size limit", str(ctx.exception))

    def test_validate_pdf_rejects_non_pdf(self):
        magic_stub.from_buffer.return_value = "image/jpeg"
        with self.assertRaises(ValueError) as ctx:
            with patch.object(ingest_module, "s3_client") as mock_s3:
                ingest_module.validate_pdf("bucket", "key.pdf", b"fake jpeg" * 100)
        self.assertIn("non-PDF", str(ctx.exception))
        magic_stub.from_buffer.return_value = "application/pdf"  # reset

    def test_validate_pdf_accepts_valid(self):
        magic_stub.from_buffer.return_value = "application/pdf"
        with patch.object(ingest_module, "s3_client"):
            # Should not raise
            ingest_module.validate_pdf("bucket", "key.pdf", b"%PDF-1.4" * 100)


class TestChunking(unittest.TestCase):

    def _make_pages(self, n_pages: int = 3, words_per_page: int = 300) -> list[tuple[str, int]]:
        """Generates synthetic page content."""
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

    def test_chunk_overlap_produces_shared_content(self):
        """Adjacent chunks should share some text due to overlap."""
        pages = [("word " * 500, 1)]
        chunks = ingest_module.chunk_pages(pages, doc_id="id", source="s")
        if len(chunks) >= 2:
            end_of_first   = chunks[0].text[-50:]
            start_of_second = chunks[1].text[:100]
            self.assertTrue(
                any(w in start_of_second for w in end_of_first.split()),
                "Expected overlap between adjacent chunks",
            )

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

    def _make_chunks(self, n: int = 3):
        from lambdas.ingest.handler import Chunk  # type: ignore
        return [Chunk(text=f"chunk {i}", chunk_index=i, doc_id="d", source="s") for i in range(n)]

    def test_bulk_called_with_correct_structure(self):
        mock_os = MagicMock()
        mock_os.bulk.return_value = {"errors": False, "items": []}
        chunks     = self._make_chunks(3)
        embeddings = [[0.0] * 1536 for _ in chunks]
        ingest_module.index_chunks(mock_os, chunks, embeddings)
        mock_os.bulk.assert_called_once()
        body = mock_os.bulk.call_args[1]["body"]
        # Alternating action + document pairs
        self.assertEqual(len(body), 6)

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

    def _sqs_event(self, bucket: str = "my-bucket", key: str = "doc.pdf") -> dict:
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
        """Validation failures must be swallowed (not retried via SQS)."""
        result = ingest_module.handler(self._sqs_event(), MagicMock())
        self.assertEqual(result["statusCode"], 200)


if __name__ == "__main__":
    unittest.main()
