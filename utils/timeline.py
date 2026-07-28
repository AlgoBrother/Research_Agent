import time
from contextlib import contextmanager
@contextmanager 
def timed_stage(label: str, latency_log: dict):
    start = time.perf_counter()
    try:
        yield
    finally:
        latency_log[label] = round(time.perf_counter() - start, 3)