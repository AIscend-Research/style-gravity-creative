from stylegravity.features import (
    FEATURE_NAMES, N_FEATURES, line_features, split_lines, tokenize, _syllables,
)
from stylegravity.seeds import SEEDS, resolve


def test_vector_length_matches_names():
    assert len(FEATURE_NAMES) == N_FEATURES
    assert len(line_features("The morning arrives.")) == N_FEATURES


def test_empty_and_symbol_only_lines_still_vectorise():
    """Erasure seeds contain lines with no words at all; they must not collapse
    to an all-zero vector indistinguishable from a blank."""
    lacuna = line_features("[  ]")
    assert len(lacuna) == N_FEATURES
    assert lacuna[FEATURE_NAMES.index("bracket_rate")] > 0


def test_monosyllable_detection():
    i = FEATURE_NAMES.index("monosyllable_rate")
    mono = line_features("The dog is old. The gate is shut.")
    poly = line_features("Interminable recapitulation of the vernal apparatus")
    assert mono[i] == 1.0
    assert poly[i] < 0.4   # of/the are the only monosyllables in six words


def test_latinate_and_archaism_separate_registers():
    lat = FEATURE_NAMES.index("latinate_rate")
    arc = FEATURE_NAMES.index("archaism_rate")
    modern = line_features("Insofar as the recapitulation constitutes an inducement")
    old = line_features("Whan that the frost bigynneth for to bite")
    assert modern[lat] > old[lat]
    assert old[arc] > modern[arc]


def test_compound_rate_catches_kennings():
    i = FEATURE_NAMES.index("compound_rate")
    assert line_features("Whale-road widened. Wound-dew reddened the oar-bench.")[i] > 0.3
    assert line_features("The morning arrives the way it always does")[i] == 0.0


def test_lipogram_feature_is_a_real_tell():
    i = FEATURE_NAMES.index("letter_e_rate")
    assert line_features("No lamp burns in that far room, no music, no talk.")[i] == 0.0
    assert line_features("The evening settles gently over everything")[i] > 0.1


def test_conjunction_open_detects_polysyndeton():
    i = FEATURE_NAMES.index("conjunction_open")
    assert line_features("And the water went down, and the land came up dry.")[i] == 1.0
    assert line_features("The water went down.")[i] == 0.0


def test_internal_space_run_detects_typographic_scatter():
    i = FEATURE_NAMES.index("internal_space_run")
    assert line_features("        the       hour")[i] == 1.0
    assert line_features("the hour unhinges itself")[i] == 0.0


def test_enjambment_and_terminal_stop_are_exclusive():
    term = FEATURE_NAMES.index("terminal_stop")
    enj = FEATURE_NAMES.index("enjambment")
    for text in ("A full stop.", "a soft comma,", "no punctuation at all"):
        v = line_features(text)
        assert not (v[term] and v[enj])


def test_syllable_estimator_is_stable_not_perfect():
    assert _syllables("cat") == 1
    assert _syllables("water") == 2
    assert _syllables("apparatus") == 4
    assert _syllables("the") == 1


def test_tokenizer_keeps_internal_apostrophes_and_hyphens():
    assert tokenize("o'er the whale-road") == ["o'er", "the", "whale-road"]


def test_split_lines_drops_blanks_and_chat_formatting():
    text = "```\n# A Title\n\nfirst line\n\nsecond line\n**Bold heading**\n```"
    assert split_lines(text) == ["first line", "second line"]


def test_split_lines_preserves_leading_whitespace():
    """Typographic seeds encode meaning in indentation — stripping it would erase
    the very feature the seed varies."""
    assert split_lines("        the       hour")[0].startswith("        ")


def test_every_seed_has_lines_and_unique_id():
    ids = [s.id for s in SEEDS]
    assert len(ids) == len(set(ids))
    for s in SEEDS:
        assert s.lines, f"{s.id} has no content lines"
        assert resolve(s.id) is s
