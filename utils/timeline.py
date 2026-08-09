import time
from contextlib import contextmanager

@contextmanager
def timed_stage(stage_name: str, step_callback=None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"⏱️ [{stage_name}] took {elapsed:.2f}s"
    if step_callback:
        step_callback(msg)
    else:
        print(f"\033[93m{msg}\033[0m")