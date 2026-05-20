#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { AskMyDocsStack } from '../lib/ask-my-docs-stack';

const app = new cdk.App();

// Retrieve environment from context or environment variables
const account = process.env.CDK_DEFAULT_ACCOUNT;
const region  = process.env.CDK_DEFAULT_REGION ?? 'us-east-1';

new AskMyDocsStack(app, 'AskMyDocsStack', {
  env: { account, region },
  description: 'Ask My Docs – Serverless RAG on AWS (PDF Q&A via Bedrock + OpenSearch Serverless)',
  // Pass through any context overrides
  allowedOrigins: app.node.tryGetContext('allowedOrigins') ?? '*',
  maxFileSizeMb:  Number(app.node.tryGetContext('maxFileSizeMb') ?? 50),
  enableWaf:      app.node.tryGetContext('enableWaf') !== 'false',
});

app.synth();
