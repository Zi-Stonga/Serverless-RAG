# Ask My Docs - Serverless RAG on AWS

Chat with any PDF using 100% serverless AWS infrastructure.
Upload a document, ask questions in natural language, get answers grounded in your content.
Deploy in minutes. Destroy just as fast.

---

## What This Is

Ask My Docs is a fully serverless Retrieval-Augmented Generation (RAG) pipeline on AWS.
Upload PDF documents and query their contents using natural language. Answers are generated
by Amazon Bedrock Claude and grounded entirely in your document content, with source
citations and page numbers. Built with AWS CDK v2. One command to deploy. One command to destroy.

### What Problem It Solves

Enterprise knowledge is locked in PDFs, reports, policies, and documents. Standard language
models cannot answer questions about private documents they have never seen. RAG solves this by:

1. Converting your documents into searchable vector embeddings
2. Retrieving the most relevant passages when a question is asked
3. Grounding the language model answer in those exact passages

The result is accurate, cited answers from your own documents.

---

## How It Works

### Ingest Flow

When a PDF is uploaded to S3, the following happens automatically:

    PDF Upload to S3
         |
         v
    SQS Queue receives S3 event
         |
         v
    Ingest Lambda
         |
         +--> 1. Validate MIME type and file size
         +--> 2. Parse PDF text per page
         +--> 3. Split into 1000-character chunks with overlap
         +--> 4. Embed each chunk via Bedrock Titan (1536 dimensions)
         +--> 5. Bulk index vectors into OpenSearch Serverless k-NN

### Query Flow

When a user submits a question:

    POST /query
         |
         v
    API Gateway + Cognito auth
         |
         v
    Query Lambda
         |
         +--> 1. Sanitize input (strip HTML, control chars)
         +--> 2. Embed question via Bedrock Titan
         +--> 3. k-NN search returns top 5 matching chunks
         +--> 4. Build prompt with retrieved context
         +--> 5. Claude 3 Haiku generates grounded answer
         +--> 6. Return answer + sources + latency

---

## Architecture

| Service | Role |
|---|---|
| Amazon S3 | PDF storage and ingest trigger |
| Amazon SQS | Reliable ingest queue with Dead Letter Queue |
| AWS Lambda Ingest | Parse, chunk, embed, and index PDFs |
| AWS Lambda Query | Embed question, search, generate answer |
| AWS Lambda Presign | Generate secure S3 upload URLs |
| Amazon Bedrock Titan Embeddings | Text to 1536-dim vectors |
| Amazon Bedrock Claude 3 Haiku | Grounded answer generation |
| OpenSearch Serverless | k-NN vector index and search |
| API Gateway REST | HTTPS endpoint with Cognito auth |
| Amazon Cognito | JWT-based user authentication |
| AWS KMS CMK | Encryption at rest with key rotation |
| AWS SSM Parameter Store | Secure configuration storage |
| AWS CDK v2 | Infrastructure as Code |

---

## Project Structure

    Serverless-RAG/
    |
    +-- cdk/
    |   +-- bin/app.ts                   CDK app entry point
    |   +-- lib/ask-my-docs-stack.ts     Full stack definition
    |   +-- cdk.json
    |   +-- package.json
    |   +-- tsconfig.json
    |
    +-- lambdas/
    |   +-- ingest/handler.py            PDF processing pipeline
    |   +-- query/handler.py             Query and answer pipeline
    |   +-- presign/handler.py           Secure upload URL generator
    |   +-- index_creator/handler.py     OpenSearch index setup
    |
    +-- tests/
    |   +-- unit/                        Unit tests
    |   +-- security/                    Security test suite
    |
    +-- scripts/
    |   +-- smoke_test.py                Post-deploy validation
    |   +-- destroy.sh                   Standard teardown script
    |   +-- nuclear_cleanup.sh           Full resource cleanup script
    |
    +-- .github/workflows/ci-cd.yml      GitHub Actions CI/CD pipeline
    +-- README.md

---

## Prerequisites

| Tool | Minimum Version |
|---|---|
| AWS CLI | 2.15+ |
| Node.js | 20 LTS |
| Python | 3.12+ |
| AWS CDK CLI | 2.130+ |

Enable Bedrock model access in your AWS region:
- amazon.titan-embed-text-v1
- anthropic.claude-3-haiku-20240307-v1:0

Go to: AWS Console > Amazon Bedrock > Model access > Modify model access

---

## Deployment

Step 1 - Clone the repository:

    git clone https://github.com/Zi-Stonga/Serverless-RAG.git
    cd Serverless-RAG

Step 2 - Install CDK dependencies:

    cd cdk && npm install

Step 3 - Configure AWS credentials:

    aws configure

Step 4 - Bootstrap CDK (first time only):

    cdk bootstrap

Step 5 - Deploy:

    npx cdk deploy --outputs-file ../outputs.json --require-approval never 2>&1 | tee ../deploy.log

Expected deploy time: 8-18 minutes.
OpenSearch Serverless provisioning dominates the wait time.
The IndexCreatorResource custom resource automatically waits for the collection
to reach ACTIVE status and creates the k-NN index. No manual steps required.

Step 6 - Confirm deployment:

    aws cloudformation describe-stacks --stack-name AskMyDocsStack --query "Stacks[0].StackStatus" --output text

Should return: CREATE_COMPLETE

---

## Usage

### Get stack outputs

    aws cloudformation describe-stacks --stack-name AskMyDocsStack --query "Stacks[0].Outputs"

### Create a Cognito user

    aws cognito-idp admin-create-user \
      --user-pool-id YOUR_POOL_ID \
      --username your@email.com \
      --temporary-password Temp123!@# \
      --user-attributes Name=email,Value=your@email.com Name=email_verified,Value=true

    aws cognito-idp admin-set-user-password \
      --user-pool-id YOUR_POOL_ID \
      --username your@email.com \
      --password YourPermanentPassword1! \
      --permanent

### Get a JWT token

    TOKEN=$(aws cognito-idp initiate-auth \
      --auth-flow USER_PASSWORD_AUTH \
      --auth-parameters USERNAME=your@email.com,PASSWORD=YourPermanentPassword1! \
      --client-id YOUR_CLIENT_ID \
      --query "AuthenticationResult.IdToken" \
      --output text)

Tokens are valid for 1 hour.

### Upload a PDF

    aws s3 cp your-document.pdf s3://YOUR_BUCKET/uploads/your-document.pdf

Wait 30-60 seconds for indexing. Verify it completed:

    aws s3api get-object-tagging --bucket YOUR_BUCKET --key uploads/your-document.pdf

Look for indexed=true in the tag set.

### Query your document

    curl -X POST YOUR_API_URL/query \
      -H "Authorization: $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"question\": \"What are the main topics in this document?\"}"

### Example response

    {
      "answer": "The document covers three main topics...",
      "sources": [
        {
          "source": "uploads/your-document.pdf",
          "page_numbers": [1, 2, 5]
        }
      ],
      "chunks_used": 5,
      "latency_ms": 2341
    }

### Monitor logs

    aws logs tail /aws/lambda/ask-my-docs-ingest --follow
    aws logs tail /aws/lambda/ask-my-docs-query --follow

---

## Security

Authentication
- Cognito JWT required on all API endpoints
- API Gateway validates token before Lambda is invoked
- Unauthenticated calls consume zero Lambda compute

IAM Least Privilege
- Separate execution roles for ingest and query Lambdas
- Ingest Lambda: S3 read, SQS consume, Titan Embeddings, OpenSearch write
- Query Lambda: Titan Embeddings, Claude Haiku, OpenSearch read only
- Query Lambda has zero S3 permissions
- Neither role can perform actions belonging to the other

Prompt Injection Defense
- Input sanitization strips HTML tags and control characters
- User input placed only in messages field, never in system prompt
- System prompt explicitly instructs Claude to answer from context only

Encryption
- S3 objects encrypted with SSE-KMS using customer-managed key
- KMS key rotation enabled
- S3 bucket policy denies unencrypted PutObject requests
- All data in transit via HTTPS

File Validation
- MIME type checked via libmagic not file extension
- A file named document.pdf containing an executable is detected and rejected
- File size enforced at both presign generation and ingest Lambda

Configuration Security
- All config stored in SSM Parameter Store
- No plaintext values in Lambda environment variables

---

## Cost Model

| Scenario | Monthly Cost | Main Driver |
|---|---|---|
| Demo - destroy same day 4 hours | Less than $0.25 | Lambda and Bedrock only |
| Persistent idle | Around $347/month | OpenSearch floor 99% |
| 100 queries per day persistent | Around $350/month | OpenSearch floor 97% |
| 5000 queries per day | Around $400-500/month | OpenSearch and Bedrock |

Always destroy when not in use.
OpenSearch Serverless charges approximately $0.48/hour regardless of query volume.

---

## Redeployment

If you destroy the stack and need to redeploy:

    cd Serverless-RAG/cdk
    npm install
    npx cdk deploy --outputs-file ../outputs.json --require-approval never 2>&1 | tee ../deploy.log

CDK bootstrap does not need to be re-run unless you changed AWS accounts or regions.
Expected redeploy time: 8-18 minutes.

---

## Teardown

Standard destroy:

    cd cdk
    npx cdk destroy --force

This removes all resources including OpenSearch collection, S3 bucket, Lambda functions,
API Gateway, Cognito User Pool, KMS key, SQS queues, and SSM parameters.

For stuck or failed stacks:

    bash scripts/nuclear_cleanup.sh

Confirm everything is gone:

    aws cloudformation describe-stacks --stack-name AskMyDocsStack --query "Stacks[0].StackStatus" --output text 2>&1

Should return: does not exist

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 403 on upload right after deploy | OpenSearch still provisioning | Wait 10 minutes. IndexCreatorResource handles this automatically. |
| No relevant document content found | PDF not yet indexed | Check ingest Lambda logs. Allow 30-60 seconds after upload. |
| 401 Unauthorized on query | Expired JWT token | Re-run the initiate-auth command to get a fresh token. |
| bedrock InvokeModel access denied | Model access not enabled | Enable Titan and Claude Haiku in Bedrock Console for your region. |
| Circular dependency on deploy | KMS key policy cycle | Use sqs.QueueEncryption.SQS_MANAGED instead of encryptionMasterKey on SQS queues. |
| ReservedConcurrentExecutions error | Account concurrency limit too low | Remove reservedConcurrentExecutions from query Lambda in CDK stack. |
| Lambda timeout on large PDFs | PDF has 300+ pages | Increase Lambda timeout in CDK stack or split PDF before uploading. |

---

## Windows Setup Notes

### Bracketed Paste Fix

Git Bash on Windows injects escape codes around pasted text causing commands to fail.
Fix by typing this manually before pasting anything:

    bind 'set enable-bracketed-paste off'

To make permanent:

    echo 'set enable-bracketed-paste off' >> ~/.inputrc

### Docker WSL2 Instability

CDK uses Docker to bundle Lambda dependencies. Docker Desktop on WSL2 frequently
crashes on Windows with errors like:

    WSL integration with distro Ubuntu unexpectedly stopped
    running wsl-bootstrap: exit status 1

Fix by switching Docker Desktop to Hyper-V backend:
1. Open PowerShell as Administrator
2. Run: bcdedit /set hypervisorlaunchtype auto
3. Run: Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
4. Restart computer
5. Docker Desktop > Settings > General > uncheck Use the WSL 2 based engine

Alternative fix - skip Docker bundling entirely by pre-installing dependencies:

    pip install -r lambdas/ingest/requirements.txt -t lambdas/ingest/ --upgrade
    pip install -r lambdas/query/requirements.txt -t lambdas/query/ --upgrade
    pip install -r lambdas/index_creator/requirements.txt -t lambdas/index_creator/ --upgrade
    pip install -r lambdas/presign/requirements.txt -t lambdas/presign/ --upgrade

Then in cdk/lib/ask-my-docs-stack.ts replace all Lambda code blocks that have
a bundling section with the simple asset reference:

    code: lambda.Code.fromAsset(path.join(__dirname, '../../lambdas/ingest')),

### Circular Dependency Error

If deploy fails with circular dependency between resources, the fixes are:
- Remove KMS encryption from SQS queues and use sqs.QueueEncryption.SQS_MANAGED
- Remove encryptionKey from the API Gateway log group
- Replace encryptionKey.grantDecrypt calls with explicit addToPolicy statements
- Remove the WAF CfnWebACLAssociation or add explicit addDependency on API stage

### Reserved Concurrency Error

If deploy fails with ReservedConcurrentExecutions error, remove this line
from the QueryLambda definition in cdk/lib/ask-my-docs-stack.ts:

    reservedConcurrentExecutions: 20,

---

## Architecture Decisions

Why Serverless
No EC2 instances, ECS clusters, or RDS databases to manage. Scales to zero when
idle and costs nothing beyond the OpenSearch floor.

Why SQS Between S3 and Lambda
Direct S3-to-Lambda invocation drops messages silently after two retries. SQS adds
reliable retry with exponential backoff and a Dead Letter Queue that preserves failed
messages for inspection and replay.

Why OpenSearch Serverless
Zero cluster management. No shard configuration, replica management, or upgrade windows.
FAISS-backed k-NN delivers high-quality approximate nearest-neighbor search. Trade-off
is the $0.48/hour minimum floor regardless of usage.

Why Claude 3 Haiku
The quality bottleneck in RAG is retrieval precision not generation capability. When
the right context is retrieved even a smaller model produces excellent answers. Haiku
is 10x cheaper than Sonnet per token.

Why Cognito Over API Keys
API keys are static strings that can be leaked via browser developer tools or logs.
Cognito issues short-lived JWT tokens that can be revoked per-user and support MFA.

Why CDK Over Raw CloudFormation
CDK resolves IAM role ARNs at synthesis time using object references. This eliminates
the manual ARN copy-paste error that caused silent 403 failures on OpenSearch data
access policy configuration.

---

## License

MIT
