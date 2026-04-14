"""
Unit tests for vocabulary and SOC code lookup.
"""
import pytest
from app.services.vocabulary import (
    load_soc_codes,
    get_soc_by_code,
    get_soc_title,
    all_soc_codes,
    keyword_lookup,
)


def test_load_soc_codes():
    codes = load_soc_codes()
    assert len(codes) >= 20
    for entry in codes:
        assert "soc_code" in entry
        assert "title" in entry
        assert "keywords" in entry


def test_get_soc_by_code():
    entry = get_soc_by_code("15-1252.00")
    assert entry is not None
    assert entry["title"] == "Software Developers"


def test_get_soc_by_code_missing():
    assert get_soc_by_code("99-9999.99") is None


def test_get_soc_title():
    assert get_soc_title("15-1252.00") == "Software Developers"


def test_get_soc_title_missing():
    assert get_soc_title("99-9999.99") == "Unknown"


def test_all_soc_codes():
    codes = all_soc_codes()
    assert "15-1252.00" in codes
    assert "15-2051.00" in codes


def test_keyword_lookup_ml_engineer():
    results = keyword_lookup("machine learning engineer building ml pipelines with python", top_k=5)
    assert len(results) > 0
    assert any(r["soc_code"] == "15-2053.00" for r in results)  # ML Engineers


def test_keyword_lookup_data_scientist():
    results = keyword_lookup("data scientist building models and running experiments", top_k=3)
    assert any(r["soc_code"] == "15-2051.00" for r in results)  # Data Scientists


def test_keyword_lookup_no_match():
    results = keyword_lookup("experienced chef preparing gourmet meals", top_k=5)
    assert results == []
