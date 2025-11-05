# Safe Commit Guide - Phase 2

**Last Updated**: November 5, 2025
**Status**: Ready to commit Phase 2 work

---

## ✅ PRE-COMMIT CHECKLIST

Before committing, verify:

### 1. Check .env is NOT being committed
```bash
git check-ignore .env
# Should output: .env
```

### 2. See what will be committed
```bash
git status
```

**Safe to commit** ✅:
- Python source files (.py)
- Documentation files (.md)
- Test files (test_*.py)
- Configuration (requirements.txt)
- Test data (eval/test_queries.json)

**NEVER commit** ❌:
- .env (API keys!)
- __pycache__/
- *.pyc files
- chroma_db/ (local vector database)
- research_cache/ (API cache)
- eval/results_*.json (evaluation results - optional)

---

## 🚀 SAFE COMMIT COMMANDS

### Step 1: Review Changes
```bash
# See all changed files
git status

# See detailed changes
git diff

# See just the file names
git diff --name-only
```

### Step 2: Stage Files (SAFE COMMAND)

**Option A: Stage specific safe files**
```bash
# Stage only new source code
git add src/vector_store.py
git add src/agents/research_synthesis.py
git add src/tools/research_retrieval.py
git add src/langgraph_orchestrator.py
git add src/agents/*.py

# Stage new test files
git add test_rag_system.py
git add eval/

# Stage documentation
git add *.md
git add phase2.md
git add requirements.txt
```

**Option B: Stage all (safer with good .gitignore)**
```bash
# This is safe ONLY because .gitignore is configured
git add .

# Double-check nothing sensitive staged
git status
```

### Step 3: Verify Before Commit
```bash
# See exactly what will be committed
git diff --cached

# List staged files
git diff --cached --name-only

# If you see .env or any secrets, UNSTAGE:
git reset HEAD .env
```

### Step 4: Commit
```bash
git commit -m "Phase 2: Add RAG integration and evaluation framework

- Implemented research-augmented generation (RAG) system
- Added vector store with ChromaDB
- Integrated Semantic Scholar and arXiv APIs
- Created research synthesis agent
- Updated all 4 agents to support citations
- Built comprehensive evaluation framework
- Added 25-query test suite for benchmarking
- All Phase 2 Week 1 tests passing (5/5)

Technical details:
- New files: ~2,100 lines of code
- RAG workflow: Router → Research → Agents → Synthesis
- Supports toggle between RAG and non-RAG modes
- LLM-as-judge evaluation for quality scoring

Next: Run evaluations to measure quality improvements"
```

---

## 🔍 WHAT'S BEING COMMITTED

### New Files (Safe to commit ✅)

**Source Code** (~1,200 lines):
- `src/vector_store.py` - ChromaDB wrapper
- `src/agents/research_synthesis.py` - RAG agent
- `src/tools/research_retrieval.py` - Research APIs
- `test_rag_system.py` - RAG test suite

**Evaluation Framework** (~900 lines):
- `eval/test_queries.json` - 25 test queries
- `eval/benchmark.py` - Evaluation system

**Documentation**:
- `PHASE2_SESSION_SUMMARY.md` - Week 1 summary
- `PHASE2_TEST_FINDINGS.md` - Test analysis
- `WEEK2_PLAN.md` - Week 2 roadmap
- `WEEK2_QUICK_START.md` - How-to guide
- `PICKUP_HERE.md` - Session summary
- `SAFE_COMMIT_GUIDE.md` - This file
- `phase2.md` - Updated progress

**Modified Files**:
- `src/langgraph_orchestrator.py` - RAG integration
- `src/agents/market_analysis.py` - Citations
- `src/agents/operations_audit.py` - Citations
- `src/agents/financial_modeling.py` - Citations
- `src/agents/lead_generation.py` - Citations
- `requirements.txt` - New dependencies
- `claude.md` - Updated context

---

## ❌ WHAT'S NOT BEING COMMITTED

**Protected by .gitignore**:
- `.env` - OpenAI + LangSmith API keys
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python
- `.venv/` - Virtual environment
- `chroma_db/` - Vector database (local data)
- `research_cache/` - API response cache
- `eval/results_*.json` - Evaluation results (optional to commit)

**These files contain**:
- API keys (security risk)
- Local runtime data (not portable)
- Large binary files (unnecessary)
- Cached responses (regeneratable)

---

## 🔒 SECURITY VERIFICATION

### Before First Commit
```bash
# 1. Verify .env is ignored
git check-ignore .env
# Expected: .env

# 2. Check for any API keys in tracked files
git grep -i "api_key" -- "*.py" "*.md"
# Should only show Config.OPENAI_API_KEY (reading from .env, not hardcoded)

# 3. List all files to be committed
git status
git diff --cached --name-only
# Review each file - nothing with "key", "secret", or "password"

# 4. Check .env is not in history
git log --all --full-history -- .env
# Should be empty (no commits with .env)
```

### If You Accidentally Staged .env
```bash
# IMMEDIATELY unstage it
git reset HEAD .env

# Verify it's unstaged
git status

# Make sure .env is in .gitignore
cat .gitignore | grep .env
```

---

## 📋 COMMIT MESSAGE TEMPLATE

```bash
git commit -m "Phase 2: [Brief summary]

[Detailed description of what was added]

Technical details:
- [Key technical points]
- [New capabilities]
- [Performance notes]

Testing:
- [What was tested]
- [Test results]

Next steps:
- [What comes next]"
```

---

## 🚨 EMERGENCY: If You Committed Secrets

**If you accidentally committed .env:**

### Option 1: If Not Pushed Yet (SAFE)
```bash
# Remove from last commit
git reset HEAD~1

# Unstage .env
git reset HEAD .env

# Re-commit without .env
git add .
git commit -m "Your commit message"
```

### Option 2: If Already Pushed (DANGER - Contact Support)
```bash
# DO NOT proceed without help
# Secrets are now public
# Need to:
# 1. Rotate all API keys immediately
# 2. Use git filter-branch or BFG to remove from history
# 3. Force push (dangerous)
```

**Prevention**: Always run `git diff --cached` before `git commit`

---

## 🎯 RECOMMENDED COMMIT STRATEGY

### Strategy 1: Single Commit (Simpler)
```bash
git add .
git status  # Verify
git diff --cached --name-only  # Double-check
git commit -m "Phase 2: Complete RAG integration and evaluation framework

- Implemented RAG system with academic paper retrieval
- Added evaluation framework with LLM-as-judge
- Updated all agents to support citations
- 2,100 lines of new code across 13 files
- All tests passing (5/5)

Ready for Week 2 evaluation runs"
```

### Strategy 2: Multiple Commits (Cleaner History)
```bash
# Commit 1: Core RAG implementation
git add src/vector_store.py src/agents/research_synthesis.py src/tools/research_retrieval.py
git commit -m "Add RAG core: vector store, research retrieval, synthesis agent"

# Commit 2: Agent updates
git add src/agents/*.py src/langgraph_orchestrator.py
git commit -m "Update agents and orchestrator for RAG integration"

# Commit 3: Testing
git add test_rag_system.py
git commit -m "Add comprehensive RAG test suite (5 tests, all passing)"

# Commit 4: Evaluation framework
git add eval/
git commit -m "Add evaluation framework with LLM-as-judge"

# Commit 5: Documentation
git add *.md phase2.md
git commit -m "Add Phase 2 documentation and guides"

# Commit 6: Dependencies
git add requirements.txt .gitignore
git commit -m "Update dependencies and gitignore for Phase 2"
```

**Recommended**: Strategy 1 (simpler, one commit for entire Phase 2)

---

## ✅ FINAL VERIFICATION

Before pushing:

```bash
# 1. Check commit history
git log -1 --stat

# 2. Verify no secrets
git show HEAD | grep -i "api_key"
# Should only show variable names, not actual keys

# 3. Double-check .env not in commit
git show HEAD --name-only | grep .env
# Should be empty

# 4. Review total changes
git show HEAD --stat
```

If all clear:
```bash
git push origin main
```

---

## 📊 EXPECTED GIT STATUS

**Before commit**:
```
M  claude.md
M  requirements.txt
M  src/agents/financial_modeling.py
M  src/agents/lead_generation.py
M  src/agents/market_analysis.py
M  src/agents/operations_audit.py
M  src/langgraph_orchestrator.py
?? PHASE2_SESSION_SUMMARY.md
?? PHASE2_TEST_FINDINGS.md
?? PICKUP_HERE.md
?? SAFE_COMMIT_GUIDE.md
?? WEEK2_PLAN.md
?? WEEK2_QUICK_START.md
?? eval/
?? phase2.md
?? src/agents/research_synthesis.py
?? src/tools/research_retrieval.py
?? src/vector_store.py
?? test_rag_system.py
```

**After commit**:
```
nothing to commit, working tree clean
```

**Files NOT shown** (correctly ignored):
- .env
- chroma_db/
- research_cache/
- __pycache__/

---

## 🎓 BEST PRACTICES

### Always Do ✅
1. `git status` before every commit
2. `git diff --cached` to review changes
3. Verify .env is gitignored: `git check-ignore .env`
4. Read commit message before pressing Enter
5. Check file names in `git status`

### Never Do ❌
1. `git add .` without checking .gitignore first
2. Commit without reviewing `git diff`
3. Hardcode API keys in source files
4. Commit large binary files
5. `git push --force` on main branch

### When Uncertain 🤔
1. Run `git status` and paste output
2. Ask for review before committing
3. Test in a new branch first
4. Keep .env file open in editor to spot it in status

---

## 🚀 READY TO COMMIT?

**Quick Checklist**:
- [ ] Ran `git check-ignore .env` (outputs ".env")
- [ ] Ran `git status` and reviewed files
- [ ] No files with "key", "secret", or "password"
- [ ] All files are source code, tests, or docs
- [ ] Ran `git diff` to review changes
- [ ] Commit message is clear and descriptive

**If all checked, you're safe to commit!**

```bash
git add .
git status  # Final check
git commit -m "Phase 2: Complete RAG integration and evaluation framework"
git push origin main
```

---

**Created**: November 5, 2025
**Purpose**: Safe commit guide for Phase 2
**Status**: Ready to commit safely
