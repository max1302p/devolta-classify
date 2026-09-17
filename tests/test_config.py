"""Environment configuration: bad values must prevent startup."""

import pytest

from app.config import DEFAULT_TEMPLATE, MODEL_ID, ConfigError, Settings


def test_defaults(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    for name in (
        "API_KEYS",
        "NUM_THREADS",
        "MAX_CONCURRENCY",
        "PORT",
        "DEFAULT_HYPOTHESIS_TEMPLATE",
        "MODEL_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.api_keys == ("secret",)
    assert settings.max_concurrency == 2
    assert settings.port == 8000
    assert settings.num_threads >= 1
    assert settings.default_hypothesis_template == DEFAULT_TEMPLATE
    assert settings.model_id == MODEL_ID


@pytest.mark.parametrize("value", ["", "   ", ",", " , "])
def test_missing_api_key_prevents_start(monkeypatch, value):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.setenv("API_KEY", value)
    with pytest.raises(ConfigError, match="API_KEYS"):
        Settings.from_env()


def test_unset_api_key_prevents_start(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_KEYS", raising=False)
    with pytest.raises(ConfigError, match="API_KEYS"):
        Settings.from_env()


def test_multiple_keys_are_split_and_trimmed(monkeypatch):
    monkeypatch.setenv("API_KEYS", " key-a , key-b ,, key-c ")
    assert Settings.from_env().api_keys == ("key-a", "key-b", "key-c")


def test_api_keys_takes_precedence_over_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "alt")
    monkeypatch.setenv("API_KEYS", "neu-1,neu-2")
    assert Settings.from_env().api_keys == ("neu-1", "neu-2")


def test_env_template_must_contain_placeholder(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("DEFAULT_HYPOTHESIS_TEMPLATE", "Kein Platzhalter.")
    with pytest.raises(ConfigError, match="DEFAULT_HYPOTHESIS_TEMPLATE"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["null", "0", "-3"])
def test_invalid_num_threads(monkeypatch, value):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("NUM_THREADS", value)
    with pytest.raises(ConfigError, match="NUM_THREADS"):
        Settings.from_env()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("NUM_THREADS", "3")
    monkeypatch.setenv("MAX_CONCURRENCY", "1")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("DEFAULT_HYPOTHESIS_TEMPLATE", "Thema: {}")
    monkeypatch.setenv("MODEL_ID", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")

    settings = Settings.from_env()

    assert (settings.num_threads, settings.max_concurrency, settings.port) == (3, 1, 9000)
    assert settings.default_hypothesis_template == "Thema: {}"
    assert settings.model_id == "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


def test_backend_defaults_to_onnx(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.delenv("ONNX_DIR", raising=False)

    settings = Settings.from_env()

    assert settings.backend == "onnx"
    assert settings.onnx_dir == "/opt/onnx"


@pytest.mark.parametrize("value", ["torch", "ONNX", " onnx "])
def test_backend_accepts_valid_values(monkeypatch, value):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("BACKEND", value)

    assert Settings.from_env().backend == value.strip().lower()


@pytest.mark.parametrize("value", ["tensorflow", "cuda", "x"])
def test_invalid_backend_prevents_start(monkeypatch, value):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("BACKEND", value)

    with pytest.raises(ConfigError, match="BACKEND"):
        Settings.from_env()


def test_onnx_dir_override(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret")
    monkeypatch.setenv("ONNX_DIR", "/models/onnx")

    assert Settings.from_env().onnx_dir == "/models/onnx"
