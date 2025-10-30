# Business Intelligence Orchestrator

A multi-agent system for comprehensive business analysis powered by OpenAI. The orchestrator coordinates specialized AI agents to provide market analysis, operations auditing, financial modeling, and lead generation strategies.

## Architecture

This MVP implements the n8n workflow prototype as a production-ready Python application.

### System Overview

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Primary Orchestrator Agent         │
│  - Analyzes query                   │
│  - Routes to specialized agents     │
│  - Synthesizes findings             │
└──────┬──────────────────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  Market    │ │ Operations │ │ Financial  │ │   Lead     │
│  Analysis  │ │   Audit    │ │  Modeling  │ │ Generation │
│   Agent    │ │   Agent    │ │   Agent    │ │   Agent    │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Synthesized   │
              │ Recommendation │
              └────────────────┘
```

### Components

- **Primary Orchestrator**: Routes queries and synthesizes multi-agent findings
- **Market Analysis Agent**: Market research, trends, competitive analysis
- **Operations Audit Agent**: Process optimization, efficiency analysis
- **Financial Modeling Agent**: ROI calculations, financial projections
- **Lead Generation Agent**: Customer acquisition strategies, growth tactics
- **Tools**: Web research (simulated), calculator
- **Memory**: Conversation history with sliding window

## Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **AI Model**: OpenAI GPT (configurable)
- **API Server**: Uvicorn
- **CLI Interface**: Python with Colorama
- **Containerization**: Docker & Docker Compose

## Project Structure

```
multi_agent_workflow/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── orchestrator.py         # Primary orchestrator logic
│   ├── memory.py               # Conversation memory
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── market_analysis.py
│   │   ├── operations_audit.py
│   │   ├── financial_modeling.py
│   │   └── lead_generation.py
│   └── tools/
│       ├── __init__.py
│       ├── calculator.py
│       └── web_research.py
├── cli.py                      # CLI interface
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   cd /workspaces/multi_agent_workflow
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```

## Usage

### Option 1: CLI Interface (Interactive)

The CLI provides an interactive chat interface with colored output:

```bash
python cli.py
```

Commands:
- Type your business query and press Enter
- `history` - View conversation history
- `clear` - Clear conversation memory
- `quit` or `exit` - Exit the application

Example queries:
- "I want to launch a SaaS product in the project management space. What should I focus on?"
- "How can I optimize my e-commerce fulfillment operations?"
- "What's the ROI of investing $50k in content marketing for B2B SaaS?"
- "What are the best lead generation strategies for enterprise software?"

### Option 2: FastAPI Server

Start the API server:

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

**POST /query**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I improve customer retention for my SaaS business?",
    "use_memory": true
  }'
```

Response:
```json
{
  "query": "How can I improve customer retention for my SaaS business?",
  "agents_consulted": ["market", "operations", "financial", "leadgen"],
  "recommendation": "...",
  "detailed_findings": {
    "market_analysis": "...",
    "operations_audit": "...",
    "financial_modeling": "...",
    "lead_generation": "..."
  }
}
```

**GET /history** - View conversation history

**POST /clear** - Clear conversation memory

**GET /health** - Health check

**GET /** - API documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option 3: Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

To run in detached mode:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop the container:
```bash
docker-compose down
```

## Example Queries

### Market Analysis
> "What are the current trends in the AI automation market?"
>
> "Analyze the competitive landscape for project management tools."

### Operations Optimization
> "How can I streamline my customer onboarding process?"
>
> "Identify bottlenecks in our software development workflow."

### Financial Planning
> "Calculate the ROI of hiring two additional sales reps at $80k each."
>
> "What should our pricing strategy be for a new B2B analytics platform?"

### Lead Generation
> "Design a lead generation funnel for enterprise SaaS."
>
> "What are cost-effective customer acquisition strategies for bootstrapped startups?"

### Comprehensive Business Planning
> "I want to launch a B2B marketplace connecting freelance designers with agencies. Provide a comprehensive go-to-market strategy."

## Customization

### Adding New Agents

1. Create a new agent file in `src/agents/`
2. Implement the agent class with appropriate system prompts
3. Add the agent to the orchestrator in `src/orchestrator.py`
4. Update routing logic in `determine_agents_needed()`

### Modifying Agent Behavior

Edit the `system_prompt` in each agent class to customize their expertise and output style.

### Integrating Real Web Research

Replace the simulated web research in `src/tools/web_research.py` with actual API integrations:
- Google Custom Search API
- Bing Search API
- Web scraping tools (Beautiful Soup, Scrapy)

### Adding Persistent Memory

Replace the in-memory conversation storage with:
- **Redis**: For session-based caching
- **PostgreSQL/Supabase**: For persistent storage
- **Vector databases**: For semantic search over conversation history

## Production Considerations

### Scaling
- Deploy behind a load balancer (Nginx, Traefik)
- Use Redis for distributed memory across instances
- Implement rate limiting and authentication
- Add monitoring and logging (Prometheus, Grafana)

### Cost Optimization
- Implement request caching
- Use streaming responses for long analyses
- Consider using different models for different agents (smaller models for simpler tasks)
- Batch similar requests

### Security
- Add API authentication (JWT, API keys)
- Implement rate limiting
- Sanitize user inputs
- Secure environment variable management (AWS Secrets Manager, HashiCorp Vault)

## Development Roadmap

- [ ] Add Redis integration for distributed memory
- [ ] Implement real web research API integrations
- [ ] Add user authentication and multi-tenancy
- [ ] Create React/Next.js frontend dashboard
- [ ] Add agent performance metrics and monitoring
- [ ] Implement vector database for semantic memory
- [ ] Add support for document uploads and analysis
- [ ] Create agent fine-tuning pipeline
- [ ] Add async processing for long-running analyses
- [ ] Implement webhooks for agent completion notifications

## Troubleshooting

### "OPENAI_API_KEY not found"
Make sure you've created a `.env` file with your OpenAI API key.

### Module import errors
Ensure you're running from the project root and have installed all dependencies:
```bash
pip install -r requirements.txt
```

### Docker build fails
Make sure Docker is running and you have sufficient disk space. Try:
```bash
docker system prune
docker-compose build --no-cache
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing issues for solutions

## Credits

Based on the n8n workflow prototype, converted to a production-ready multi-agent system using FastAPI and OpenAI.
