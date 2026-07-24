from app.agent.semantic_planning.resolvers.common import select_one_candidate


def test_one_valid_candidate_is_resolved():
    candidate = object()
    result = select_one_candidate(
        raw_text="华南",
        candidate_ids=["v1"],
        catalog={"v1": candidate},
    )

    assert result.status == "resolved"
    assert result.candidate is candidate
    assert result.issue is None


def test_multiple_valid_ids_are_ambiguous_not_first_match():
    result = select_one_candidate(
        raw_text="华南",
        candidate_ids=["v1", "v2"],
        catalog={"v1": object(), "v2": object()},
    )

    assert result.status == "ambiguous"
    assert result.candidate is None
    assert result.issue.code == "value_ambiguous"
    assert result.issue.candidate_ids == ["v1", "v2"]


def test_zero_ids_are_unresolved():
    result = select_one_candidate(
        raw_text="火星",
        candidate_ids=[],
        catalog={},
    )

    assert result.status == "unresolved"
    assert result.issue.code == "value_not_bound"


def test_any_catalog_outside_id_is_invalid():
    result = select_one_candidate(
        raw_text="华南",
        candidate_ids=["v1", "invented"],
        catalog={"v1": object()},
    )

    assert result.status == "unresolved"
    assert result.issue.code == "invalid_candidate_id"
    assert result.issue.candidate_ids == ["invented"]


def test_raw_text_is_provenance_not_a_hard_gate():
    result = select_one_candidate(
        raw_text="华南地区",
        candidate_ids=["v1"],
        catalog={"v1": object()},
    )

    assert result.status == "resolved"
    assert result.candidate_id == "v1"
    assert result.issue is None
