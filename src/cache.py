import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional, Any, Dict
logger = logging.getLogger(__name__)

class CacheBackend:

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

class RedisCache(CacheBackend):

    def __init__(self, redis_url: str, namespace: str):
        import redis
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.redis.ping()
        self.namespace = namespace

    def _key(self, key: str) -> str:
        return f'{self.namespace}:{key}'

    def get(self, key: str) -> Optional[Any]:
        cached = self.redis.get(self._key(key))
        return json.loads(cached) if cached else None

    def set(self, key: str, value: Any, ttl: int):
        self.redis.setex(self._key(key), ttl, json.dumps(value, default=str))

    def clear(self):
        for key in self.redis.scan_iter(match=f'{self.namespace}:*'):
            self.redis.delete(key)

class FileCache(CacheBackend):

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f'{key_hash}.json'

    def get(self, key: str) -> Optional[Any]:
        cache_file = self._path(key)
        if not cache_file.exists():
            return None
        data = json.loads(cache_file.read_text())
        if data['expires_at'] > time.time():
            return data['value']
        cache_file.unlink()
        return None

    def set(self, key: str, value: Any, ttl: int):
        cache_file = self._path(key)
        cache_file.write_text(json.dumps({'value': value, 'expires_at': time.time() + ttl}, default=str))

    def clear(self):
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()

class QueryCache:
    TTL_RESEARCH = 604800
    TTL_AGENT = 86400
    TTL_SYNTHESIS = 86400
    TTL_SIMPLE = 604800

    def __init__(self, client_id: Optional[str]=None):
        self.client_id = client_id
        self.backend = self._init_backend()
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0}
        if self.backend:
            backend_type = type(self.backend).__name__
            logger.info(f'✓ Cache enabled: {backend_type}')
        else:
            logger.info('Cache disabled')

    def _init_backend(self) -> Optional[CacheBackend]:
        if os.getenv('CACHE_ENABLED', 'true').lower() != 'true':
            return None
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        try:
            backend = RedisCache(redis_url=redis_url, namespace=os.getenv('CACHE_NAMESPACE', 'bi'))
            logger.info(f'Using Redis cache: {redis_url}')
            return backend
        except Exception as e:
            logger.warning(f'Redis unavailable ({e}), using file cache')
        cache_dir = os.getenv('CACHE_DIR', '.cache')
        return FileCache(cache_dir)

    def _make_key(self, prefix: str, *parts: str) -> str:
        content = '::'.join((str(p) for p in parts))
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if self.client_id:
            return f'client:{self.client_id}:{prefix}:{content_hash}'
        return f'{prefix}:{content_hash}'

    def _get(self, key: str, label: str) -> Optional[Any]:
        if not self.backend:
            return None
        result = self.backend.get(key)
        if result:
            self.stats['hits'] += 1
            logger.info(f'Cache HIT: {label}')
        else:
            self.stats['misses'] += 1
        return result

    def _set(self, key: str, value: Any, ttl: int, label: str):
        if not self.backend:
            return
        self.backend.set(key, value, ttl)
        self.stats['saves'] += 1
        logger.debug(f'Cached: {label}')

    def get_research(self, query: str) -> Optional[Dict]:
        key = self._make_key('research', query)
        return self._get(key, f'research ({query[:50]}...)')

    def set_research(self, query: str, papers: Dict):
        key = self._make_key('research', query)
        self._set(key, papers, self.TTL_RESEARCH, 'research')

    def get_agent_response(self, agent: str, query: str, has_research: bool=False) -> Optional[str]:
        key = self._make_key('agent', agent, query, str(has_research))
        return self._get(key, f'{agent} agent ({query[:30]}...)')

    def set_agent_response(self, agent: str, query: str, response: str, has_research: bool=False):
        key = self._make_key('agent', agent, query, str(has_research))
        self._set(key, response, self.TTL_AGENT, f'{agent} agent')

    def get_synthesis(self, query: str, agents_used: list) -> Optional[str]:
        agents_key = ':'.join(sorted(agents_used))
        key = self._make_key('synthesis', query, agents_key)
        return self._get(key, f'synthesis ({query[:30]}...)')

    def set_synthesis(self, query: str, agents_used: list, synthesis: str):
        agents_key = ':'.join(sorted(agents_used))
        key = self._make_key('synthesis', query, agents_key)
        self._set(key, synthesis, self.TTL_SYNTHESIS, 'synthesis')

    def get_simple_answer(self, query: str) -> Optional[str]:
        key = self._make_key('simple', query)
        return self._get(key, f'simple answer ({query[:30]}...)')

    def set_simple_answer(self, query: str, answer: str):
        key = self._make_key('simple', query)
        self._set(key, answer, self.TTL_SIMPLE, 'simple answer')

    def clear(self):
        if self.backend:
            self.backend.clear()
            self.stats = {'hits': 0, 'misses': 0, 'saves': 0}
            logger.info('Cache cleared')

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total * 100 if total > 0 else 0
        return {'enabled': self.backend is not None, 'backend': type(self.backend).__name__ if self.backend else 'None', 'hits': self.stats['hits'], 'misses': self.stats['misses'], 'saves': self.stats['saves'], 'total_requests': total, 'hit_rate_percent': round(hit_rate, 1), 'cost_savings_estimate': round(hit_rate * 0.01, 2)}