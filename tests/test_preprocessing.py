"""Smoke tests for structure-aware chunking."""

from rag_core.ingestion.preprocessing import (
    _clean_html_fragment,
    _is_boilerplate,
    _normalize_header,
)


def test_clean_html_strips_tags():
    assert _clean_html_fragment("<p>Hello <b>world</b></p>") == "Hello world"


def test_normalize_header_maps_aliases():
    assert _normalize_header("Key Responsibilities") == "Responsibilities"
    assert _normalize_header("Nice to have") == "Preferred Qualifications"


def test_boilerplate_detection():
    text = (
        "We are an equal opportunity employer. All qualified applicants "
        "will receive consideration without regard to race, color, religion."
    )
    assert _is_boilerplate(text) is True
    assert _is_boilerplate("Build Python APIs and ship features weekly.") is False
