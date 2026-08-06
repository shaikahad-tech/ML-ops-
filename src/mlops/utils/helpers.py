"""General utility helpers."""
from __future__ import annotations
import hashlib, json, time
from functools import wraps
from typing import Any, Callable

def timing(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        return result
    return wrapper

def hash_dict(d: dict) -> str:
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
