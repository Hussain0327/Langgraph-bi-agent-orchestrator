# Deployment Guide

Complete guide for deploying the Business Intelligence Orchestrator in different environments.

## Table of Contents

- [Local Development Deployment](#local-development-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [AWS ECS](#aws-ecs)
  - [GCP Cloud Run](#gcp-cloud-run)
  - [Azure Container Instances](#azure-container-instances)
- [Production Considerations](#production-considerations)
- [Monitoring and Observability](#monitoring-and-observability)

---

## Local Development Deployment

Best for development and testing.

### Prerequisites

- Python 3.12 or higher
- pip
- virtualenv (recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hussain0327/Business-Intelligence-Orchestrator.git
   cd Business-Intelligence-Orchestrator
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start Redis (optional but recommended):**
   ```bash
   # macOS
   brew install redis
   brew services start redis

   # Linux
   sudo apt-get install redis-server
   sudo systemctl start redis

   # Or use Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

6. **Run the application:**
   ```bash
   # Interactive CLI
   python cli.py

   # Or API server
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

---

## Docker Deployment

Recommended for consistent environments and easy deployment.

### Prerequisites

- Docker 20.10 or higher
- Docker Compose 2.0 or higher

### Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/Hussain0327/Business-Intelligence-Orchestrator.git
   cd Business-Intelligence-Orchestrator
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Check status:**
   ```bash
   docker-compose ps
   ```

4. **View logs:**
   ```bash
   # All services
   docker-compose logs -f

   # Specific service
   docker-compose logs -f orchestrator
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

### Docker Compose Configuration

The `docker-compose.yml` includes:
- **Redis**: Caching backend on port 6379
- **Orchestrator**: API service on port 8000

Services are configured with:
- Health checks
- Automatic restart policies
- Volume mounting for development
- Environment variable injection

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

---

## Cloud Deployment

### AWS ECS (Elastic Container Service)

Deploy as a containerized service on AWS.

#### Prerequisites

- AWS account with ECS access
- AWS CLI configured
- ECR repository for Docker images

#### Steps

1. **Create ECR repository:**
   ```bash
   aws ecr create-repository --repository-name bi-orchestrator
   ```

2. **Build and push Docker image:**
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build image
   docker build -t bi-orchestrator .

   # Tag image
   docker tag bi-orchestrator:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/bi-orchestrator:latest

   # Push image
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/bi-orchestrator:latest
   ```

3. **Set up infrastructure:**
   ```bash
   # Create ECS cluster
   aws ecs create-cluster --cluster-name bi-orchestrator-cluster

   # Create task definition (use task-definition.json)
   aws ecs register-task-definition --cli-input-json file://task-definition.json

   # Create service
   aws ecs create-service \
     --cluster bi-orchestrator-cluster \
     --service-name bi-orchestrator-service \
     --task-definition bi-orchestrator \
     --desired-count 1 \
     --launch-type FARGATE
   ```

4. **Set up ElastiCache for Redis:**
   ```bash
   aws elasticache create-cache-cluster \
     --cache-cluster-id bi-orchestrator-cache \
     --engine redis \
     --cache-node-type cache.t3.micro \
     --num-cache-nodes 1
   ```

5. **Configure environment variables:**
   - Use AWS Secrets Manager for API keys
   - Update task definition with secret ARNs
   - Set REDIS_URL to ElastiCache endpoint

6. **Set up Application Load Balancer (optional):**
   - Create ALB targeting ECS service
   - Configure health check on /health
   - Set up SSL certificate

#### Cost Estimate (Monthly)

- ECS Fargate (1 task, 0.5 vCPU, 1GB RAM): ~$15
- ElastiCache (t3.micro): ~$12
- Load Balancer (optional): ~$16
- **Total: ~$27-43/month** (excluding API costs)

---

### GCP Cloud Run

Serverless deployment on Google Cloud Platform.

#### Prerequisites

- GCP account with Cloud Run enabled
- gcloud CLI configured
- Artifact Registry repository

#### Steps

1. **Create Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create bi-orchestrator \
     --repository-format=docker \
     --location=us-central1
   ```

2. **Build and push image:**
   ```bash
   # Build with Cloud Build
   gcloud builds submit --tag us-central1-docker.pkg.dev/<project-id>/bi-orchestrator/app
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy bi-orchestrator \
     --image us-central1-docker.pkg.dev/<project-id>/bi-orchestrator/app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 1 \
     --timeout 300s
   ```

4. **Set environment variables:**
   ```bash
   gcloud run services update bi-orchestrator \
     --update-env-vars OPENAI_API_KEY=your-key,MODEL_STRATEGY=hybrid \
     --update-secrets DEEPSEEK_API_KEY=deepseek-secret:latest
   ```

5. **Set up Redis with Memorystore:**
   ```bash
   gcloud redis instances create bi-orchestrator-cache \
     --size=1 \
     --region=us-central1 \
     --redis-version=redis_7_0
   ```

6. **Configure VPC connector for Redis access:**
   ```bash
   gcloud compute networks vpc-access connectors create bi-connector \
     --region us-central1 \
     --range 10.8.0.0/28

   gcloud run services update bi-orchestrator \
     --vpc-connector bi-connector
   ```

#### Cost Estimate (Monthly, 1000 requests/day)

- Cloud Run: ~$5
- Memorystore (1GB): ~$35
- Egress: ~$5
- **Total: ~$45/month** (excluding API costs)

---

### Azure Container Instances

Deploy on Microsoft Azure.

#### Prerequisites

- Azure account
- Azure CLI configured
- Container Registry

#### Steps

1. **Create resource group:**
   ```bash
   az group create --name bi-orchestrator-rg --location eastus
   ```

2. **Create Container Registry:**
   ```bash
   az acr create --resource-group bi-orchestrator-rg \
     --name biorchestrator --sku Basic

   az acr login --name biorchestrator
   ```

3. **Build and push image:**
   ```bash
   # Build
   docker build -t bi-orchestrator .

   # Tag
   docker tag bi-orchestrator biorchestrator.azurecr.io/bi-orchestrator:latest

   # Push
   docker push biorchestrator.azurecr.io/bi-orchestrator:latest
   ```

4. **Create Azure Cache for Redis:**
   ```bash
   az redis create \
     --resource-group bi-orchestrator-rg \
     --name bi-orchestrator-cache \
     --location eastus \
     --sku Basic \
     --vm-size c0
   ```

5. **Deploy container:**
   ```bash
   az container create \
     --resource-group bi-orchestrator-rg \
     --name bi-orchestrator \
     --image biorchestrator.azurecr.io/bi-orchestrator:latest \
     --cpu 1 \
     --memory 2 \
     --registry-login-server biorchestrator.azurecr.io \
     --registry-username <username> \
     --registry-password <password> \
     --environment-variables OPENAI_API_KEY=<key> MODEL_STRATEGY=hybrid \
     --ports 8000 \
     --dns-name-label bi-orchestrator
   ```

#### Cost Estimate (Monthly)

- Container Instance (1 vCPU, 2GB): ~$30
- Azure Cache for Redis (Basic): ~$15
- **Total: ~$45/month** (excluding API costs)

---

## Production Considerations

### Security

1. **API Key Management:**
   - Use cloud provider's secrets manager
   - Never commit keys to version control
   - Rotate keys regularly

2. **Authentication:**
   - Add API key or JWT authentication
   - Use OAuth for user-facing applications
   - Implement rate limiting

3. **HTTPS:**
   ```bash
   # Example nginx reverse proxy config
   server {
       listen 443 ssl;
       server_name api.yourdomain.com;

       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Reliability

1. **Health Checks:**
   - Use `/health` endpoint for liveness checks
   - Configure readiness probes

2. **Error Handling:**
   - System has built-in fallbacks (Redis, DeepSeek, research APIs)
   - Monitor error rates

3. **Backup:**
   - Regular Redis backups if using persistence
   - Database backups if adding user data

### Performance

1. **Scaling:**
   - Horizontal scaling: Run multiple instances behind load balancer
   - Vertical scaling: Increase CPU/memory for faster processing

2. **Caching:**
   - Use Redis in production (not file cache)
   - Configure appropriate TTLs
   - Monitor cache hit rates

3. **Optimization:**
   - Enable hybrid model strategy for cost savings
   - Use ML routing once accuracy > 85%
   - Tune agent temperatures based on use case

### Cost Management

1. **LLM Costs:**
   - Use hybrid strategy (86% savings vs GPT-5 only)
   - Monitor usage with cost tracking
   - Set budget alerts

2. **Infrastructure:**
   - Use managed services in production
   - Auto-scale based on load
   - Use spot instances for non-critical workloads

---

## Monitoring and Observability

### Application Monitoring

1. **LangSmith Integration:**
   ```bash
   # Enable in .env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-key
   LANGCHAIN_PROJECT=bi-orchestrator-prod
   ```

2. **Prometheus Metrics:**
   ```python
   # Add to src/main.py
   from prometheus_client import Counter, Histogram

   query_counter = Counter('queries_total', 'Total queries')
   query_duration = Histogram('query_duration_seconds', 'Query duration')
   ```

3. **Logging:**
   ```python
   # Configure structured logging
   import logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

### Infrastructure Monitoring

1. **Cloud Provider Tools:**
   - AWS: CloudWatch
   - GCP: Cloud Monitoring
   - Azure: Monitor

2. **Key Metrics to Track:**
   - Request rate and latency
   - Error rate by type
   - Cache hit rate
   - LLM API costs
   - Memory and CPU usage

### Alerting

Set up alerts for:
- High error rate (> 5%)
- Slow responses (> 5 minutes)
- Cache failures
- High API costs
- Service downtime

---

## Updating in Production

### Zero-Downtime Deployment

1. **Blue-Green Deployment:**
   ```bash
   # Deploy new version alongside old
   # Switch traffic after validation
   # Keep old version for quick rollback
   ```

2. **Rolling Updates:**
   ```bash
   # Update instances one at a time
   # Monitor health after each update
   ```

### Rollback Procedure

```bash
# Docker
docker-compose down
git checkout <previous-commit>
docker-compose up --build -d

# Cloud (ECS example)
aws ecs update-service \
  --cluster bi-orchestrator-cluster \
  --service bi-orchestrator-service \
  --task-definition bi-orchestrator:<previous-version>
```

---

## Next Steps

After deployment:

1. Run benchmarks to verify performance
2. Set up monitoring dashboards
3. Configure automated backups
4. Test failover scenarios
5. Document your specific deployment
6. Set up CI/CD for automated deployments

For issues, see the [Troubleshooting Guide](TROUBLESHOOTING.md).
