"""Line-level stylometric features.

Every line of every poem — seed lines, prefilled continuations, and unprompted
baselines — is reduced to the same fixed-length vector. Drift is then measured
in this space, so the whole method rests on these features being (a) computable
without a model, (b) sensitive to the axes the seeds actually vary along
(diction, morphology, register, form), and (c) insensitive to topic.

Deliberately no embeddings: an embedding model's own priors would sit between
the measurement and the thing measured, and embeddings are dominated by topic
rather than style. These are surface features a prosodist would recognise.
"""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

#  Closed-class words: rate of these separates paratactic/plain registers from
#  nominal/compound ones (skaldic and legalese sit at opposite extremes).
FUNCTION_WORDS = frozenset("""
a an the and or but nor for yet so if then than that which who whom whose
of in on at by to from with without within into onto upon over under above
below through during before after between among against about across
is are was were be been being am do does did done have has had having
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs this these those there here what when where while
as such not no nor never all any some each every both either neither
shall should will would can could may might must
""".split())

#  Period markers. A continuation that keeps these is still wearing the costume.
ARCHAISMS = frozenset("""
thee thou thy thine ye hath doth dost art wert shalt canst hast
o'er e'er ne'er 'tis 'twas oft ere whilst amongst amidst betwixt
whence whither hence hither thence nigh yea nay lo behold verily
whan thanne bigynneth longen tellen sitten wyf brydde thei nowe seven
""".split())

LATINATE_SUFFIXES = (
    "tion", "sion", "ment", "ance", "ence", "ity", "ities", "ous", "ious",
    "ive", "ate", "able", "ible", "ical", "ism", "ist", "ency", "ancy",
)

#  Kenning-style compounding: the skaldic seed is almost pure hyphenated
#  compound nouns, a shape nothing in the default register produces.
COMPOUND_RE = re.compile(r"[A-Za-z]{2,}-[A-Za-z]{2,}")

FEATURE_NAMES: tuple[str, ...] = (
    "words_per_line",
    "chars_per_word",
    "syllables_per_word",
    "monosyllable_rate",
    "long_word_rate",
    "type_token_ratio",
    "function_word_rate",
    "archaism_rate",
    "latinate_rate",
    "compound_rate",
    "punct_density",
    "comma_rate",
    "dash_rate",
    "terminal_stop",
    "enjambment",
    "initial_capital",
    "all_lower",
    "first_person_rate",
    "conjunction_open",
    "bracket_rate",
    "digit_rate",
    "internal_space_run",
    "letter_e_rate",
    "repeat_word_rate",
)

N_FEATURES = len(FEATURE_NAMES)


def _syllables(word: str) -> int:
    """Cheap English syllable estimate. Consistent, not correct — and consistency
    is what a distance metric needs."""
    w = word.lower().strip("'’-")
    if not w:
        return 0
    groups = VOWEL_GROUP_RE.findall(w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


def tokenize(line: str) -> list[str]:
    return WORD_RE.findall(line)


def line_features(line: str) -> list[float]:
    """Reduce one line to a fixed-length feature vector.

    A line with no alphabetic tokens (a bare ``[  ]`` lacuna, a row of spaces)
    still yields a valid vector — the orthographic features carry it. Returning
    zeros for those would make erasure-style seeds look like blank lines.
    """
    raw = line.rstrip("\n")
    stripped = raw.strip()
    words = tokenize(stripped)
    lower = [w.lower() for w in words]
    n = len(words)
    nchars = max(len(stripped), 1)

    def rate(count: int) -> float:
        return count / n if n else 0.0

    syl = [_syllables(w) for w in words]
    punct = sum(1 for ch in stripped if unicodedata.category(ch).startswith("P"))

    ends_terminal = bool(re.search(r"[.!?]['\"’”]?$", stripped))
    ends_soft = bool(re.search(r"[,;:—–-]['\"’”]?$", stripped))
    letters = [ch for ch in stripped.lower() if ch.isalpha()]

    return [
        float(n),
        (sum(len(w) for w in words) / n) if n else 0.0,
        (sum(syl) / n) if n else 0.0,
        rate(sum(1 for s in syl if s == 1)),
        rate(sum(1 for w in words if len(w) > 6)),
        (len(set(lower)) / n) if n else 0.0,
        rate(sum(1 for w in lower if w in FUNCTION_WORDS)),
        rate(sum(1 for w in lower if w.strip("'’") in ARCHAISMS)),
        rate(sum(1 for w in lower if w.endswith(LATINATE_SUFFIXES))),
        len(COMPOUND_RE.findall(stripped)) / max(n, 1),
        punct / nchars,
        stripped.count(",") / nchars,
        (stripped.count("—") + stripped.count("–") + stripped.count(" - ")) / nchars,
        float(ends_terminal),
        float(not ends_terminal and not ends_soft),
        float(bool(stripped[:1].isupper())),
        float(bool(letters) and stripped == stripped.lower()),
        rate(sum(1 for w in lower if w in {"i", "me", "my", "mine", "we", "us", "our"})),
        float(bool(lower[:1]) and lower[0] in {"and", "or", "but", "nor", "for", "yet"}),
        sum(stripped.count(c) for c in "[]{}()") / nchars,
        sum(1 for ch in stripped if ch.isdigit()) / nchars,
        float(bool(re.search(r"\S {2,}\S", raw))),
        (letters.count("e") / len(letters)) if letters else 0.0,
        rate(n - len(set(lower))),
    ]


def poem_features(lines: list[str]) -> list[list[float]]:
    return [line_features(ln) for ln in lines]


def split_lines(text: str, *, keep_blank: bool = False) -> list[str]:
    """Split generated text into analysable lines.

    Blank lines are stanza breaks, not content, and would otherwise dominate the
    orthographic features. Markdown fencing and the occasional bolded title are
    stripped — models sometimes wrap verse in them and that formatting is an
    artefact of the chat surface, not of the poem's style.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            continue
        if not line.strip():
            if keep_blank:
                out.append("")
            continue
        if re.fullmatch(r"\s*(#{1,6}\s+.*|\*\*.*\*\*)\s*", line):
            continue
        out.append(line)
    return out
