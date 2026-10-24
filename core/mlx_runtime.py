"""All MLX work runs on one dedicated thread.

mlx core 0.31.2 (PR ml-explore/mlx#3281) made GPU streams thread-local, so an array
keeps the affinity of the thread that allocated it. mlx_lm hasn't adapted yet
(open bug ml-explore/mlx-lm#1256), so loading the model/KV-cache on the main thread
and then generating on a Flask-SocketIO worker thread crashes with
"There is no Stream(gpu, N) in current thread". The CLI never hit it (all one thread).

Fix: load, cache allocation, and generation all go through this single worker, so the
affinity always matches. Generation is serialized anyway on one GPU, so nothing is lost.
"""
import threading
from concurrent.futures import Future, ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_mlx_thread_id = _executor.submit(threading.get_ident).result()


def run_on_mlx(fn, *args, **kwargs):
    """Run `fn` on the dedicated MLX thread; return its result (errors re-raise)."""
    if threading.get_ident() == _mlx_thread_id:
        return fn(*args, **kwargs)
    return _executor.submit(fn, *args, **kwargs).result()


def submit_on_mlx(fn, *args, **kwargs):
    """Like run_on_mlx, but return a Future instead of waiting for the result."""
    if threading.get_ident() == _mlx_thread_id:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:      # noqa: BLE001 - delivered through the future
            future.set_exception(e)
        return future
    return _executor.submit(fn, *args, **kwargs)
