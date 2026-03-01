import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        _time = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__} in {time.perf_counter() - _time:.4f}s")
        return result
    return wrapper