"""
ask-my-docs  –  AOSS Index Creator (CloudFormation Custom Resource)
====================================================================
Creates (or updates) the OpenSearch Serverless k-NN index at CDK deploy time.
This eliminates the manual "create index via Dev Tools" step documented in the
original project README.

Handles all three CFN lifecycle events:
  Create  → create index (idempotent – OK if already exists)
  Update  → no-op (index settings cannot be changed after creation)
  Delete  → no-op (index is deleted when the collection is deleted)
"""

from __future__ import annotations

import json
import logging
from typing import Any
import os
import time

import boto3
import urllib.request
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger(__name__)

COLLECTION_ENDPOINT = os.environ["COLLECTION_ENDPOINT"]
INDEX_NAME          = os.environ["INDEX_NAME"]
REGION              = os.environ["REGION"]


def _build_os_client() -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    host        = COLLECTION_ENDPOINT.replace("https://", "")
    auth        = AWSV4SignerAuth(credentials, REGION, "aoss")
    return OpenSearch(
        hosts            = [{"host": host, "port": 443}],
        http_auth        = auth,
        use_ssl          = True,
        verify_certs     = True,
        connection_class = RequestsHttpConnection,
        timeout          = 30,
    )


INDEX_BODY = {
    "settings": {
        "index": {
            "knn":             True,
            "knn.algo_param.ef_search": 512,
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type":      "knn_vector",
                "dimension": 1536,
                "method": {
                    "name":       "hnsw",
                    "space_type": "cosinesimil",
                    "engine":     "faiss",
                    "parameters": {"ef_construction": 512, "m": 16},
                },
            },
            "text":         {"type": "text"},
            "source":       {"type": "keyword"},
            "doc_id":       {"type": "keyword"},
            "chunk_index":  {"type": "integer"},
            "page_numbers": {"type": "integer"},
            "ingested_at":  {"type": "date", "format": "epoch_second"},
        }
    },
}


def _send_response(event: dict, context: Any, status: str, reason: str, data: dict = None) -> None:
    """Sends a response to the CloudFormation pre-signed S3 URL."""
    body = json.dumps({
        "Status":             status,
        "Reason":             reason,
        "PhysicalResourceId": f"aoss-index-{INDEX_NAME}",
        "StackId":            event["StackId"],
        "RequestId":          event["RequestId"],
        "LogicalResourceId":  event["LogicalResourceId"],
        "Data":               data or {},
    }).encode("utf-8")

    req = urllib.request.Request(
        event["ResponseURL"],
        data    = body,
        method  = "PUT",
        headers = {"Content-Type": "", "Content-Length": len(body)},
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def handler(event: dict, context: Any) -> None:
    logger.info("Custom resource event received", extra={"request_type": event.get("RequestType")})
    request_type = event["RequestType"]

    if request_type == "Delete":
        _send_response(event, context, "SUCCESS", "No cleanup needed")
        return

    if request_type == "Update":
        _send_response(event, context, "SUCCESS", "Index already exists – no update needed")
        return

    # Create
    # Wait for collection to be active (AOSS can take 5-10 min after stack completion)
    aoss_client = boto3.client("opensearchserverless", region_name=REGION)
    collection_name = INDEX_NAME  # collection and index share the same name in this project

    for attempt in range(24):  # max 12 minutes
        collection = next(
            (c for c in aoss_client.list_collections().get("collectionSummaries", []) if c["name"] == collection_name),
            None,
        )
        if collection and collection.get("status") == "ACTIVE":
            logger.info("OpenSearch collection reached ACTIVE status", extra={"elapsed_seconds": attempt * 30})
            break
        logger.info("Collection not yet ACTIVE, retrying", extra={"attempt": attempt + 1, "max_attempts": 24})
        time.sleep(30)
    else:
        _send_response(event, context, "FAILED", "Collection never reached ACTIVE status after 12 minutes")
        return

    try:
        os_client = _build_os_client()

        if os_client.indices.exists(INDEX_NAME):
            logger.info("Index already exists, skipping creation", extra={"index_name": INDEX_NAME})
        else:
            os_client.indices.create(index=INDEX_NAME, body=INDEX_BODY)
            logger.info("Index created successfully", extra={"index_name": INDEX_NAME})

        _send_response(event, context, "SUCCESS", "Index ready", {"IndexName": INDEX_NAME})

    except Exception as exc:
        logger.error("Failed to create OpenSearch index", extra={"error": str(exc)})
        _send_response(event, context, "FAILED", str(exc))
