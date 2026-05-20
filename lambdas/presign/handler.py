"""
ask-my-docs  –  Pre-Signed URL Lambda
=======================================
Generates a short-lived S3 pre-signed PUT URL so the frontend can upload
PDFs directly to S3 without routing them through API Gateway/Lambda.

The pre-signed URL enforces:
  • Content-Type: application/pdf
  • Content-Length-Range: 1 byte – MAX_FILE_SIZE_MB
  • SSE-KMS encryption via the bucket default
  • 15-minute expiry
"""

from __future__ import annotations

import json
import os
import time
import uuid

import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger(service="ask-my-docs-presign")
tracer = Tracer(service="ask-my-docs-presign")

s3_client    = boto3.client("s3")
BUCKET_NAME  = os.environ["BUCKET_NAME"]
MAX_BYTES    = int(os.environ.get("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
URL_EXPIRY_S = 900  # 15 minutes


def _cors_headers() -> dict:
    return {
        "Content-Type":                "application/json",
        "Access-Control-Allow-Origin": "*",
        "X-Content-Type-Options":      "nosniff",
    }


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def handler(event: dict, context: LambdaContext) -> dict:
    try:
        body     = json.loads(event.get("body") or "{}")
        filename = body.get("filename", "upload.pdf")

        # Sanitize filename: strip path components, keep only safe chars
        safe_name = "".join(c for c in os.path.basename(filename)
                            if c.isalnum() or c in "._-")[:128] or "upload.pdf"
        object_key = f"uploads/{int(time.time())}-{uuid.uuid4().hex[:8]}-{safe_name}"

        presigned = s3_client.generate_presigned_post(
            Bucket     = BUCKET_NAME,
            Key        = object_key,
            Fields     = {"Content-Type": "application/pdf"},
            Conditions = [
                {"Content-Type": "application/pdf"},
                ["content-length-range", 1, MAX_BYTES],
            ],
            ExpiresIn  = URL_EXPIRY_S,
        )

        logger.info("Presigned URL generated", key=object_key)
        return {
            "statusCode": 200,
            "headers":    _cors_headers(),
            "body":       json.dumps({
                "upload_url": presigned["url"],
                "fields":     presigned["fields"],
                "object_key": object_key,
                "expires_in": URL_EXPIRY_S,
            }),
        }

    except Exception as exc:
        logger.exception("Presign error", error=str(exc))
        return {
            "statusCode": 500,
            "headers":    _cors_headers(),
            "body":       json.dumps({"error": "Could not generate upload URL"}),
        }
