"""Progress reporting for long steps (loading mods, building models, rendering).

Code calls report(fraction, text) as it goes; whoever started the work in this thread decides
where that goes (the window's progress bars) with listen(). stage(start, end, text) maps the
0..1 progress of an inner step onto part of the whole, so steps can nest.
"""
import contextlib
import threading
import time

_local = threading.local()


def _state():
    if not hasattr(_local, "cb"):
        _local.cb, _local.ranges, _local.last = None, [(0.0, 1.0, "")], 0.0
    return _local


@contextlib.contextmanager
def listen(callback):
    """Send this thread's progress to callback(fraction 0..1, text) while the block runs."""
    st = _state()
    old = st.cb, st.ranges
    st.cb, st.ranges, st.last = callback, [(0.0, 1.0, "")], 0.0
    try:
        yield
    finally:
        st.cb, st.ranges = old


@contextlib.contextmanager
def stage(start, end, text=None):
    """Progress reported inside the block fills [start, end] of the enclosing step."""
    st = _state()
    lo, hi, label = st.ranges[-1]
    st.ranges.append((lo + (hi - lo) * start, lo + (hi - lo) * end, text or label))
    report(0.0)
    try:
        yield
    finally:
        report(1.0)
        st.ranges.pop()


def report(fraction, text=None, force=False):
    st = _state()
    if st.cb is None:
        return
    now = time.monotonic()
    if not force and 0.0 < fraction < 1.0 and now - st.last < 0.05:
        return  # at most ~20 updates a second
    st.last = now
    lo, hi, label = st.ranges[-1]
    st.cb(lo + (hi - lo) * max(0.0, min(1.0, fraction)), text or label)
