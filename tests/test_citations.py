from genai_platform.rag.prompts import REFUSAL, validate_citations


def test_valid_citations_are_grounded():
    r = validate_citations("Fact one [1]. Fact two [2].", 2)
    assert r == {"cited": [1, 2], "invalid": [], "refused": False, "grounded": True}


def test_out_of_range_citation_is_ungrounded():
    r = validate_citations("Fact one [1][7].", 2)
    assert r["invalid"] == [7] and not r["grounded"]


def test_refusal_counts_as_grounded():
    assert validate_citations(REFUSAL, 3)["grounded"] is True


def test_uncited_claim_is_ungrounded():
    assert validate_citations("Just an assertion with no brackets.", 3)["grounded"] is False
