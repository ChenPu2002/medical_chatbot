"""Unit tests for the pure symptom-matching logic in tree_model_medicine.py.

These tests only cover deterministic, side-effect-free helper functions.
They avoid exercising anything that depends on randomness (e.g. first_predict)
or on the trained model artifacts beyond simple loading, keeping the suite
fast and stable in CI.
"""
import tree_model_medicine as tmm


def test_check_pattern_finds_matching_symptom():
    """check_pattern should find symptoms containing the given substring."""
    found, matches = tmm.check_pattern(tmm.chk_dis, "itching")
    assert found == 1
    assert "itching" in matches


def test_check_pattern_matches_multi_word_symptom_with_space_input():
    """User input with spaces should be normalized to underscores before matching."""
    found, matches = tmm.check_pattern(tmm.chk_dis, "skin rash")
    assert found == 1
    assert "skin_rash" in matches


def test_check_pattern_returns_no_match_for_unknown_symptom():
    """check_pattern should report no match for symptoms absent from the list."""
    found, matches = tmm.check_pattern(tmm.chk_dis, "not_a_real_symptom_xyz")
    assert found == 0
    assert not matches


def test_get_poss_symptom_single_match_prompts_confirmation():
    """When exactly one symptom matches, the output should ask for confirmation."""
    output, conf, cnf_dis = tmm.get_poss_symptom("continuous_sneezing")
    assert conf == 1
    assert cnf_dis == ["continuous_sneezing"]
    assert "Is this the symptom you are experiencing?" in output


def test_get_poss_symptom_no_match():
    """When no symptom matches, conf should be 0 and the match list empty."""
    output, conf, cnf_dis = tmm.get_poss_symptom("not_a_real_symptom_xyz")
    assert conf == 0
    assert not cnf_dis
    assert "searches related to input" in output


def test_symptoms_dict_covers_all_training_columns():
    """symptoms_dict should map every symptom column to a unique index."""
    assert len(tmm.symptoms_dict) == len(tmm.cols)
    assert len(set(tmm.symptoms_dict.values())) == len(tmm.symptoms_dict)


def test_description_and_precaution_dictionaries_are_populated():
    """Description and precaution lookups should be loaded for known diseases."""
    assert len(tmm.description_list) > 0
    assert len(tmm.precautionDictionary) > 0
