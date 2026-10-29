"""Rate-limit slots of LLMService._generate: the pause comes before a call, never after the last one."""
import threading

import pytest

import src.llm.llm_service as svc


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []
        self._lock = threading.Lock()

    def monotonic(self):
        return self.now

    def sleep(self, s):
        with self._lock:
            self.sleeps.append(s)


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(svc.time, "monotonic", c.monotonic)
    monkeypatch.setattr(svc.time, "sleep", c.sleep)
    monkeypatch.setattr(svc, "_next_slot", 0.0)
    monkeypatch.setattr(svc, "call_llm", lambda *a, **k: ("ok", "fake/model"))
    return c


def _service():
    s = object.__new__(svc.LLMService)  # skip __init__ (needs no key/network)
    s.last_model = None
    return s


def test_first_call_does_not_wait(clock):
    assert _service()._generate("p", "general") == "ok"
    assert clock.sleeps == []


def test_next_call_waits_only_the_rest_of_the_interval(clock):
    s = _service()
    s._generate("p", "general")
    clock.now += 2.5  # e.g. the call itself + retrieval took 2.5 s
    s._generate("p", "general")
    assert clock.sleeps == [pytest.approx(svc.LLM_MIN_INTERVAL - 2.5)]


def test_no_wait_after_a_long_gap(clock):
    s = _service()
    s._generate("p", "general")
    clock.now += svc.LLM_MIN_INTERVAL + 1
    s._generate("p", "general")
    assert clock.sleeps == []


def test_no_pause_after_the_last_call(clock):
    _service()._generate("p", "general")
    assert clock.sleeps == []  # the old code slept 6 s here, after the answer was ready


def test_concurrent_calls_get_distinct_slots(clock):
    s = _service()
    threads = [threading.Thread(target=s._generate, args=("p", "general")) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # same instant, 3 callers: one starts now, the others 1 and 2 intervals later
    assert sorted(clock.sleeps) == [pytest.approx(svc.LLM_MIN_INTERVAL), pytest.approx(2 * svc.LLM_MIN_INTERVAL)]
