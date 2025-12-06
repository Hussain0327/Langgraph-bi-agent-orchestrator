import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.config import Config
load_dotenv()
Config.validate()
app = FastAPI(title='Business Intelligence Orchestrator v2', description='LangGraph-powered multi-agent system with GPT-5, LangSmith tracing, and parallel execution', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
orchestrator = LangGraphOrchestrator()

class QueryRequest(BaseModel):
    query: str
    use_memory: Optional[bool] = True

class QueryResponse(BaseModel):
    query: str
    agents_consulted: list
    recommendation: str
    detailed_findings: dict

@app.get('/')
async def root():
    return {'name': 'Business Intelligence Orchestrator v2', 'version': '2.0.0', 'description': 'LangGraph-powered multi-agent system with GPT-5, LangSmith tracing, and parallel execution', 'features': ['GPT-5 Responses API with 40-80% cost reduction via caching', 'LangGraph state machine for intelligent routing', 'LangSmith tracing and monitoring', 'Parallel agent execution', 'Semantic routing (not keyword-based)'], 'agents': ['Market Analysis', 'Operations Audit', 'Financial Modeling', 'Lead Generation'], 'endpoints': {'/query': 'POST - Submit a business query for analysis', '/history': 'GET - Get conversation history', '/clear': 'POST - Clear conversation memory', '/cache/stats': 'GET - Get cache performance statistics', '/cache/clear': 'POST - Clear cache', '/health': 'GET - Health check'}}

@app.post('/query', response_model=QueryResponse)
async def analyze_query(request: QueryRequest):
    try:
        result = orchestrator.orchestrate(query=request.query, use_memory=request.use_memory)
        return QueryResponse(query=result['query'], agents_consulted=result['agents_consulted'], recommendation=result['recommendation'], detailed_findings=result['detailed_findings'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/history')
async def get_history():
    try:
        history = orchestrator.get_conversation_history()
        return {'history': history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/clear')
async def clear_memory():
    try:
        orchestrator.clear_memory()
        return {'message': 'Conversation memory cleared successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/cache/stats')
async def get_cache_stats():
    try:
        stats = orchestrator.get_cache_stats()
        return {'message': 'Cache statistics', 'stats': stats, 'explanation': {'hit_rate': 'Percentage of queries served from cache (higher = faster + cheaper)', 'cost_savings': 'Estimated cost savings from cache hits', 'backend': 'Cache backend (Redis for production, File for development)'}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/cache/clear')
async def clear_cache():
    try:
        orchestrator.clear_cache()
        return {'message': 'Cache cleared successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health_check():
    cache_stats = orchestrator.get_cache_stats()
    return {'status': 'healthy', 'openai_configured': bool(Config.OPENAI_API_KEY), 'openai_model': Config.OPENAI_MODEL, 'using_gpt5': Config.is_gpt5(), 'langsmith_tracing': Config.LANGCHAIN_TRACING_V2, 'langsmith_project': Config.LANGCHAIN_PROJECT if Config.LANGCHAIN_TRACING_V2 else None, 'cache': {'enabled': cache_stats['enabled'], 'backend': cache_stats['backend'], 'hit_rate': f"{cache_stats['hit_rate_percent']}%"}}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)