# AWS Deployment Guide (Not Deployed — Documented for Reference)

This project is containerized and ready to deploy to AWS using ECR (Elastic
Container Registry) + ECS (Elastic Container Service) or EC2. No live AWS
resources are provisioned for this submission; this document describes the
steps a deployment would follow.

## Architecture
- Two containers (already built and tested locally): `purchase-intent-api`
  and `purchase-intent-dashboard` (see `docker-compose.yml`)
- Each pushed to its own ECR repository
- Run as ECS Fargate services (serverless containers, no EC2 management),
  or alternatively on a single EC2 instance running `docker-compose up`

## Step-by-step (ECR + ECS Fargate)

1. **Create ECR repositories** (one-time):

2. **Authenticate Docker to ECR:**

3. **Tag and push both images:**

4. **Create an ECS Fargate cluster and task definitions** pointing at each
   ECR image, exposing container ports 8000 (API) and 8501 (dashboard).

5. **Create an ECS service** for each task definition, attached to an
   Application Load Balancer for public access.

## Configuration parameterization

All paths, the decision threshold, and the random seed are already
externalized in `src/config/config.yaml` — no code changes are needed to
deploy; only environment-specific values (AWS region, account ID, ECR
repository URIs) are supplied at deploy time via the commands above, not
hardcoded in the application.

## Cost note

This guide is provided for completeness per the project requirements. No
AWS resources have been provisioned for this submission — running the
above would incur standard AWS ECR/ECS/ALB charges.