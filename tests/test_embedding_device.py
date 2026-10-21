"""Pemilihan device embedding (tanpa memuat model, tanpa jaringan)."""
import types

import pytest

from src.llm import embedding as E


@pytest.fixture
def fake_torch(monkeypatch):
    def install(cuda_available):
        mod = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda_available))
        monkeypatch.setitem(__import__("sys").modules, "torch", mod)
    return install


@pytest.mark.parametrize("cuda, expected", [(True, "cuda"), (False, "cpu")])
def test_auto_uses_gpu_only_when_available(fake_torch, cuda, expected):
    fake_torch(cuda)
    assert E.resolve_device("auto") == expected


@pytest.mark.parametrize("setting, expected", [("cpu", "cpu"), ("CUDA", "cuda"), (" cpu ", "cpu")])
def test_explicit_setting_is_respected(fake_torch, setting, expected):
    fake_torch(True)  # GPU ada, tapi pilihan eksplisit menang
    assert E.resolve_device(setting) == expected


def test_empty_setting_means_auto(fake_torch):
    fake_torch(False)
    assert E.resolve_device("") == "cpu"
