"""Optional coordination of GPU memory between jobs on a shared host.

If RRT_MEM_SHARE_DIR names a directory that contains a `mem_share.py` module (a host-level memory manager providing
reserve(), claim() and budget_gb()), its functions are used unchanged. Otherwise:
  reserve(min_gb, holder) yields the vLLM/PyTorch memory fraction from RRT_GPU_MEM_UTIL (default 0.9),
  claim(holder, gb) does nothing.
The experiments were run with the host manager; the fraction it returned for each run is printed in the run logs
("vLLM gpu_memory_utilization=...") and listed in docs/environment.md.
"""
import contextlib
import os
import sys

_DIR = os.environ.get('RRT_MEM_SHARE_DIR')
_host = None
if _DIR and os.path.exists(os.path.join(_DIR, 'mem_share.py')):
    sys.path.insert(0, _DIR)
    import mem_share as _host  # noqa: E402


@contextlib.contextmanager
def reserve(min_gb, holder, **kw):
    if _host is not None:
        with _host.reserve(min_gb, holder, **kw) as util:
            yield util
    else:
        yield float(os.environ.get('RRT_GPU_MEM_UTIL', 0.9))


def claim(holder, gb, pid=None):
    if _host is not None:
        _host.claim(holder, gb, pid)
