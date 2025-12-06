# Troubleshooting Guide

Common issues and solutions for the Business Intelligence Orchestrator.

## Table of Contents

- [API and Authentication Issues](#api-and-authentication-issues)
- [Caching and Performance Issues](#caching-and-performance-issues)
- [Agent and LLM Issues](#agent-and-llm-issues)
- [Document Generation Issues](#document-generation-issues)
- [Deployment and Environment Issues](#deployment-and-environment-issues)

---

## API and Authentication Issues

### OpenAI API Key Invalid

**Symptoms:**
- Error: "Invalid API key" or "Unauthorized"
- 401 authentication errors

**Solutions:**
1. Check your API key in `.env`:
   ```bash
   cat .env | grep OPENAI_API_KEY
   ```
2. Verify the key is active in your OpenAI account
3. Make sure there are no extra spaces or quotes around the key
4. Ensure the key starts with `sk-proj-` for project keys

### DeepSeek API Error

**Symptoms:**
- DeepSeek API calls failing
- Falling back to GPT-5 for all queries

**Solutions:**
1. Verify your DeepSeek API key:
   ```bash
   cat .env | grep DEEPSEEK_API_KEY
   ```
2. Check you have credits in your DeepSeek account
3. The system automatically falls back to GPT-5, so queries will still work
4. If you don't want DeepSeek, set `MODEL_STRATEGY=gpt5` in `.env`

### Semantic Scholar Rate Limit

**Symptoms:**
- Research retrieval fails
- "Rate limit exceeded" messages

**Solutions:**
- The system automatically falls back to arXiv
- 7-day caching reduces API calls by 60%
- Wait 1 hour and try again if both APIs are rate limited
- Consider adding a Semantic Scholar API key (optional) in `.env`

---

## Caching and Performance Issues

### Redis Connection Failed

**Symptoms:**
- Warning: "Could not connect to Redis"
- Cache operations failing

**Solutions:**
1. Check if Redis is running:
   ```bash
   docker-compose ps redis
   ```
2. Start Redis if needed:
   ```bash
   docker-compose up redis -d
   ```
3. System automatically falls back to file cache, so queries will still work
4. Check Redis URL in `.env`:
   ```bash
   REDIS_URL=redis://localhost:6379/0
   ```

### Slow Query Performance

**Symptoms:**
- Queries taking longer than expected
- No cache hits

**Solutions:**
1. Check if caching is enabled:
   ```bash
   cat .env | grep CACHE_ENABLED
   ```
2. Verify Redis is running (see above)
3. First-time queries are always slower - try the same query again
4. Check cache stats:
   ```bash
   curl http://localhost:8000/cache/stats
   ```

### Cache Not Working

**Symptoms:**
- Every query takes full time
- Cache hit rate is 0%

**Solutions:**
1. Make sure `CACHE_ENABLED=true` in `.env`
2. Restart the application after changing `.env`
3. Clear cache and try again:
   ```bash
   curl -X POST http://localhost:8000/cache/clear
   ```

---

## Agent and LLM Issues

### Empty Agent Outputs

**Symptoms:**
- Agents return empty or very short responses
- Missing analysis in final output

**Solutions:**
- This was a known issue and is now fixed
- Ensure you're using the latest code
- `reasoning_effort` is set to "low" in the code
- If still happening, check LangSmith traces for errors

### Agent Selection Issues

**Symptoms:**
- Wrong agents being called for queries
- All queries using the same agents

**Solutions:**
1. Check your routing strategy in `.env`:
   ```bash
   cat .env | grep ROUTING_STRATEGY
   ```
2. Try switching to GPT-5 routing:
   ```bash
   ROUTING_STRATEGY=gpt5
   ```
3. If using ML routing, the classifier might need retraining
4. Provide more specific queries to improve routing

### GPT-5 Responses Empty

**Symptoms:**
- GPT-5 calls return empty text
- `output_text` is None or empty

**Solutions:**
1. Check your OpenAI model name:
   ```bash
   cat .env | grep OPENAI_MODEL
   ```
2. Verify you have GPT-5 access
3. Try with `gpt-4o` as fallback
4. Check LangSmith traces for API errors

---

## Document Generation Issues

### PowerPoint Not Generated

**Symptoms:**
- No `.pptx` file created
- Error during document generation

**Solutions:**
1. Check if `python-pptx` is installed:
   ```bash
   pip list | grep python-pptx
   ```
2. Install missing dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Check file permissions in current directory
4. Try the test script:
   ```bash
   python test_document_automation.py
   ```

### Excel Workbook Errors

**Symptoms:**
- Excel file corrupted or won't open
- Missing sheets in workbook

**Solutions:**
1. Check `openpyxl` installation:
   ```bash
   pip list | grep openpyxl
   ```
2. Make sure synthesis output has required fields
3. Try opening with LibreOffice if Excel shows errors
4. Check the test workbook opens:
   ```bash
   python test_document_automation.py
   ```

### Charts Not Displaying

**Symptoms:**
- Blank charts in PowerPoint or Excel
- Chart generation errors

**Solutions:**
1. Verify `matplotlib` is installed:
   ```bash
   pip list | grep matplotlib
   ```
2. Check if synthesis contains numeric data for charts
3. Look for chart files in temp directory
4. Try regenerating with verbose logging

---

## Deployment and Environment Issues

### Environment Variables Not Loading

**Symptoms:**
- Default values being used instead of `.env` values
- "Missing API key" errors when key is in `.env`

**Solutions:**
1. Make sure `.env` file exists:
   ```bash
   ls -la .env
   ```
2. Copy from example if needed:
   ```bash
   cp .env.example .env
   ```
3. Restart the application after editing `.env`
4. For Docker, rebuild the container:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

### Docker Container Won't Start

**Symptoms:**
- Container exits immediately
- "Port already in use" errors

**Solutions:**
1. Check if port 8000 is already used:
   ```bash
   lsof -i :8000
   ```
2. Stop conflicting service or change port in `docker-compose.yml`
3. Check container logs:
   ```bash
   docker-compose logs orchestrator
   ```
4. Remove old containers and rebuild:
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

### Import Errors

**Symptoms:**
- `ModuleNotFoundError` when running scripts
- "No module named..." errors

**Solutions:**
1. Make sure you're in the project root directory
2. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Check Python version (requires 3.12+):
   ```bash
   python --version
   ```

### Memory Issues

**Symptoms:**
- Out of memory errors
- System freezing during queries

**Solutions:**
1. Reduce number of parallel agents in code
2. Increase Docker memory limits in Docker Desktop
3. Clear cache to free up memory:
   ```bash
   curl -X POST http://localhost:8000/cache/clear
   ```
4. Use DeepSeek only (uses less memory than GPT-5)

---

## Getting Additional Help

If your issue isn't covered here:

1. Check LangSmith traces if you have tracing enabled
2. Look at application logs for error messages
3. Try the minimal example first:
   ```bash
   python cli.py
   ```
4. Check GitHub issues for similar problems
5. Ensure you're using the latest code from main branch

## Debugging Tips

**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Test individual components:**
```bash
# Test document generation
python test_document_automation.py

# Test RAG system
python test_rag_system.py

# Test DeepSeek integration
python test_deepseek.py
```

**Check API connectivity:**
```bash
# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test DeepSeek
curl https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY"
```
