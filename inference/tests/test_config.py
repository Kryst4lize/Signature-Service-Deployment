"""Settings parsing.

`cors_origins` has its own test because typing it `list[str]` looks obviously
right and crash-loops the container: pydantic-settings JSON-decodes
complex-typed fields in EnvSettingsSource before any validator runs, so
`CORS_ORIGINS=http://localhost:8110` raises SettingsError at import time.
"""

import pytest

from app.config import Settings

BASE = dict(
    postgres_host="h",
    postgres_db="d",
    postgres_user="u",
    postgres_password="p",
)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **BASE, **kw)


def test_plain_comma_separated_origins_parse():
    s = _settings(cors_origins="http://localhost:8110,http://127.0.0.1:8110")
    assert s.cors_origin_list == ["http://localhost:8110", "http://127.0.0.1:8110"]


def test_origins_are_stripped_and_blanks_dropped():
    s = _settings(cors_origins=" http://a:1 , , http://b:2 ,")
    assert s.cors_origin_list == ["http://a:1", "http://b:2"]


def test_empty_origins_means_no_cors_middleware():
    assert _settings().cors_origin_list == []
    assert _settings(cors_origins="   ").cors_origin_list == []


def test_single_origin_is_not_split_into_characters():
    s = _settings(cors_origins="http://only-one:8110")
    assert s.cors_origin_list == ["http://only-one:8110"]


def test_env_var_form_matches_dot_env_form(monkeypatch):
    """The path that actually crashed: value arriving via os.environ."""
    for k, v in BASE.items():
        monkeypatch.setenv(k.upper(), v)
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8110")
    s = Settings(_env_file=None)
    assert s.cors_origin_list == ["http://localhost:8110"]


def test_database_url_is_asyncpg():
    url = _settings().database_url
    assert url.startswith("postgresql+asyncpg://")
    assert "@h:5432/d" in url


def test_numeric_settings_coerce_from_strings(monkeypatch):
    s = _settings(match_threshold="0.42", max_pdf_pages="7")
    assert s.match_threshold == pytest.approx(0.42)
    assert s.max_pdf_pages == 7
