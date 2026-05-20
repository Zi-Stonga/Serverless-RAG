"""
ask-my-docs  –  Ingest Lambda
==============================
Triggered by SQS (which receives S3 event notifications).

Flow
----
  S3 PUT (*.pdf)
    → SQS message
      → this Lambda
        1. Validate MIME type and file size
        2. Extract text with pypdf
        3. Chunk text with overlap
        4. Embed each chunk via Bedrock Titan Embeddings
        5. Batch-index chunks into OpenSearch Serverless
        6. Tag the S3 object as processed

Security controls
-----------------
  • MIME validation rejects non-PDF content before processing
  • File size cap enforced both here and at presign generation
  • All config from SSM Parameter Store – no plaintext env vars for secrets
  • AWS Lambda Powertools for structured logging + X-Ray tracing
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Generator

import boto3
import magic  # python-magic
import pypdf
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# ── Powertools ──────────────────────────────────────────────────────────────
logger  = Logger(service="ask-my-docs-ingest")
tracer  = Tracer(service="ask-my-docs-ingest")
metrics = Metrics(namespace="AskMyDocs", service="ingest")

# ── AWS clients ─────────────────────────────────────────────────────────────
s3_client      = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")
ssm_client     = boto3.client("ssm")

# ── Configuration ───────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES  = int(os.environ.get("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
CHUNK_SIZE_CHARS     = 1000   # target chunk size in characters
CHUNK_OVERLAP_CHARS  = 150    # overlap between consecutive chunks
EMBEDDING_MODEL_ID   = "amazon.titan-embed-text-v1"
INDEX_NAME           = "ask-my-docs"

_config_cache: dict[str, str] = {}


def _get_param(name: str) -> str:
    """Retrieve SSM Parameter Store value with in-memory caching."""
    if name not in _config_cache:
        resp = ssm_client.get_parameter(Name=name, WithDecryption=True)
        _config_cache[name] = resp["Parameter"]["Value"]
    return _config_cache[name]


@dataclass
class Chunk:
    text:        str
    chunk_index: int
    doc_id:      str
    source:      str
    page_numbers: list[int] = field(default_factory=list)


# ── Core utilities ──────────────────────────────────────────────────────────

@tracer.capture_method
def validate_pdf(bucket: str, key: str, body: bytes) -> None:
    """
    Validates MIME type via libmagic and file size.
    Deletes the S3 object and raises ValueError on failure so the
    SQS message is NOT retried (we log it and move on).
    """
    if len(body) > MAX_FILE_SIZE_BYTES:
        s3_client.delete_object(Bucket=bucket, Key=key)
        raise ValueError(f"File exceeds size limit ({len(body)} > {MAX_FILE_SIZE_BYTES})")

    mime = magic.from_buffer(body[:4096], mime=True)
    if mime != "application/pdf":
        s3_client.delete_object(Bucket=bucket, Key=key)
        raise ValueError(f"Rejected non-PDF upload: mime={mime} key={key}")


@tracer.capture_method
def extract_text_pages(body: bytes) -> list[tuple[str, int]]:
    """
    Extracts (text, page_number) tuples from a PDF byte stream.
    Returns a flat list ordered by page.
    """
    import io
    reader = pypdf.PdfReader(io.BytesIO(body))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((text, page_num))
    return pages


def chunk_pages(pages: list[tuple[str, int]], doc_id: str, source: str) -> list[Chunk]:
    """
    Splits page text into fixed-size chunks with overlap.
    Preserves page number provenance for citation.
    """
    chunks: list[Chunk] = []
    buffer         = ""
    buffer_pages: list[int] = []
    chunk_index    = 0

    for text, page_num in pages:
        buffer += " " + text
        buffer_pages.append(page_num)

        while len(buffer) >= CHUNK_SIZE_CHARS:
            chunk_text = buffer[:CHUNK_SIZE_CHARS].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    source=source,
                    page_numbers=sorted(set(buffer_pages)),
                ))
                chunk_index += 1
            buffer       = buffer[CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS:]
            buffer_pages = buffer_pages[:]  # keep provenance across overlap

    # Flush remainder
    remainder = buffer.strip()
    if remainder:
        chunks.append(Chunk(
            text=remainder,
            chunk_index=chunk_index,
            doc_id=doc_id,
            source=source,
            page_numbers=sorted(set(buffer_pages)),
        ))

    return chunks


@tracer.capture_method
def embed_text(text: str) -> list[float]:
    """Calls Bedrock Titan Embeddings and returns the 1536-dim vector."""
    body = json.dumps({"inputText": text[:8192]})  # Titan v1 max 8192 tokens
    response = bedrock_client.invoke_model(
        modelId     = EMBEDDING_MODEL_ID,
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )
    return json.loads(response["body"].read())["embedding"]


@tracer.capture_method
def index_chunks(os_client: OpenSearch, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Bulk-indexes chunks + embeddings into OpenSearch."""
    if not chunks:
        return

    bulk_body = []
    for chunk, embedding in zip(chunks, embeddings):
        bulk_body.append({"index": {"_index": INDEX_NAME}})
        bulk_body.append({
            "text":         chunk.text,
            "embedding":    embedding,
            "source":       chunk.source,
            "doc_id":       chunk.doc_id,
            "chunk_index":  chunk.chunk_index,
            "page_numbers": chunk.page_numbers,
            "ingested_at":  int(time.time()),
        })

    response = os_client.bulk(body=bulk_body)
    if response.get("errors"):
        failed = [
            item for item in response["items"]
            if item.get("index", {}).get("error")
        ]
        logger.error("Bulk index had errors", failed_count=len(failed), sample=failed[:3])
        raise RuntimeError(f"OpenSearch bulk index errors: {len(failed)} failures")

    logger.info("Indexed chunks", count=len(chunks))


def _build_os_client() -> OpenSearch:
    region   = _get_param(os.environ["REGION_PARAM"])
    endpoint = _get_param(os.environ["COLLECTION_ENDPOINT_PARAM"])
    host     = endpoint.replace("https://", "")
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, "aoss")
    return OpenSearch(
        hosts              = [{"host": host, "port": 443}],
        http_auth          = auth,
        use_ssl            = True,
        verify_certs       = True,
        connection_class   = RequestsHttpConnection,
        timeout            = 30,
    )


# ── Handler ─────────────────────────────────────────────────────────────────

@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event: dict, context: LambdaContext) -> dict:
    """
    SQS event handler. Each SQS record wraps one S3 event notification.
    We process one PDF per invocation (batchSize=1 in CDK).
    """
    os_client = _build_os_client()
    errors = []

    for sqs_record in event.get("Records", []):
        try:
            s3_event = json.loads(sqs_record["body"])

            for s3_record in s3_event.get("Records", []):
                bucket = s3_record["s3"]["bucket"]["name"]
                key    = urllib.parse.unquote_plus(s3_record["s3"]["object"]["key"])
                _process_pdf(bucket, key, os_client)

        except ValueError as exc:
            # Validation failures are NOT retried (message is deleted from queue)
            logger.warning("Validation failure – message discarded", error=str(exc))
            metrics.add_metric(name="ValidationFailures", unit=MetricUnit.Count, value=1)
        except Exception as exc:
            logger.exception("Unhandled error – message will be retried", error=str(exc))
            errors.append(str(exc))
            metrics.add_metric(name="ProcessingErrors", unit=MetricUnit.Count, value=1)

    if errors:
        raise RuntimeError(f"Ingest failed for {len(errors)} record(s): {errors[0]}")

    return {"statusCode": 200}


import urllib.parse  # noqa: E402  (needs to be after function defs to avoid circular)


@tracer.capture_method
def _process_pdf(bucket: str, key: str, os_client: OpenSearch) -> None:
    """Download, validate, parse, embed, and index a single PDF."""
    logger.info("Processing PDF", bucket=bucket, key=key)

    # 1. Download
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
    body   = s3_obj["Body"].read()

    # 2. Validate
    validate_pdf(bucket, key, body)

    # 3. Generate stable document ID from bucket+key
    doc_id = hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:16]

    # 4. Extract text
    pages = extract_text_pages(body)
    if not pages:
        logger.warning("No extractable text in PDF", key=key)
        return

    # 5. Chunk
    chunks = chunk_pages(pages, doc_id=doc_id, source=key)
    logger.info("Chunked document", chunk_count=len(chunks), key=key)
    metrics.add_metric(name="ChunksCreated", unit=MetricUnit.Count, value=len(chunks))

    # 6. Embed each chunk
    embeddings = [embed_text(c.text) for c in chunks]
    metrics.add_metric(name="EmbeddingsCalled", unit=MetricUnit.Count, value=len(embeddings))

    # 7. Index
    index_chunks(os_client, chunks, embeddings)
    metrics.add_metric(name="DocumentsIngested", unit=MetricUnit.Count, value=1)

    # 8. Tag S3 object as processed
    s3_client.put_object_tagging(
        Bucket=bucket, Key=key,
        Tagging={"TagSet": [
            {"Key": "indexed",   "Value": "true"},
            {"Key": "doc_id",    "Value": doc_id},
            {"Key": "chunk_count", "Value": str(len(chunks))},
        ]},
    )
    logger.info("PDF successfully indexed", key=key, doc_id=doc_id, chunks=len(chunks))
