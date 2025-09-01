# proof_tactics.py
# Stricter, LaTeX-aware labeling functions for proof tactics (weak supervision)

import re
try:
    from snorkel.labeling import labeling_function
except Exception:
    # fallback: allow file to work without snorkel installed
    def labeling_function(*args, **kwargs):
        def _wrap(f): return f
        return _wrap

# ===== Labels (unchanged) =====
ABSTAIN = -1
DIRECT = 0
CONTRADICTION = 1
CONTRAPOSITIVE = 2
INDUCTION = 3
CASE_ANALYSIS = 4
EXISTENCE = 5
COUNTEREXAMPLE = 6

TACTIC_NAMES = {
    ABSTAIN: "Abstain",
    DIRECT: "Direct Proof",
    CONTRADICTION: "Proof by Contradiction",
    CONTRAPOSITIVE: "Contrapositive",
    INDUCTION: "Induction",
    CASE_ANALYSIS: "Case Analysis",
    EXISTENCE: "Existence / Construction",
    COUNTEREXAMPLE: "Counterexample",
}

# ===== Utilities =====
def _t(x: str) -> str:
    return (x or "").strip().lower()

def _regex(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.I) is not None
    except re.error:
        return False

def _regex_any(text: str, patterns) -> bool:
    return any(_regex(text, p) for p in patterns)

def _any(text: str, needles) -> bool:
    return any(n in text for n in needles)

def _near(text: str, pat_a: str, pat_b: str, window: int = 60) -> bool:
    """True if matches of pat_a and pat_b occur within `window` chars."""
    try:
        A = [m.start() for m in re.finditer(pat_a, text, flags=re.I)]
        B = [m.start() for m in re.finditer(pat_b, text, flags=re.I)]
    except re.error:
        return False
    return any(abs(a - b) <= window for a in A for b in B)

# Negative masks so generic rules don’t overfire
NEG_CONTRA = r"contradict"
NEG_CASES  = r"\bcase\b|\bcase\s*\d|\beither\b|\botherwise\b|\bw\.?l\.?o\.?g\.?"
NEG_CP     = r"contrapositive"
NEG_IND    = r"\binduction\b|\binductive\b|\bbase case\b|\binduction hypothesis\b|\bfor all\s+(n|k)\b|\bk\+?1\b|\bn\+?1\b"
NEG_EXISTS = r"\bthere exists\b|\bexists\b|\bfor some\b|\bchoose\b|\bconstruct\b|\bdefine\b"
NEG_CE     = r"counter[- ]?example|fails for|not always true"

# ------------------------------------------------------------------------------------
# COUNTEREXAMPLE (most specific)
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_ce_keywords(row):
    t = _t(row.text)
    if _regex_any(t, [r"\bcounter[- ]?example\b", r"\bdisprove\b", r"\bnot always true\b", r"\bfails for\b"]):
        return COUNTEREXAMPLE
    return ABSTAIN

@labeling_function()
def lf_ce_exists_not(row):
    t = _t(row.text)
    # “There exists … such that … not …” (textual or LaTeX)
    if _near(t, r"\b(there exists|exists|\\exists)\b", r"\b(such that|:\\text\{such that\}|\\text\{s\.t\.\})\b", window=50) and \
       _regex_any(t, [r"\bnot\b", r"\\neg\b", r"¬"]):
        return COUNTEREXAMPLE
    # simple: “for some … not …”
    if _near(t, r"\bfor some\b", r"\bnot\b", window=50):
        return COUNTEREXAMPLE
    return ABSTAIN

# ------------------------------------------------------------------------------------
# CONTRAPOSITIVE
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_cp_keywords(row):
    t = _t(row.text)
    if _any(t, ["contrapositive", "prove the contrapositive", "by contrapositive", "it suffices to prove the contrapositive"]):
        return CONTRAPOSITIVE
    return ABSTAIN

@labeling_function()
def lf_cp_structure(row):
    t = _t(row.text)
    if "contradiction" in t:  # avoid confusion
        return ABSTAIN
    # “if not B then not A”, textual or LaTeX (¬B ⇒ ¬A)
    if _regex_any(t, [
        r"\bif\s+not\s+.+\s+then\s+not\s+.+",
        r"\bnot\s+.+\s+implies\s+not\s+.+",
        r"(\\neg|¬)\s*.+(\\Rightarrow|⇒)\s*(\\neg|¬)\s*.+",
    ]):
        return CONTRAPOSITIVE
    # “assume B is false; then A is false”
    if _near(t, r"\bassume\b.*\bnot\b", r"\bthen\b.*\bnot\b", window=100):
        return CONTRAPOSITIVE
    return ABSTAIN

# ------------------------------------------------------------------------------------
# CONTRADICTION
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_contra_keywords(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\bassume (the )?contrary\b", r"\bsuppose (the )?contrary\b",
        r"\bfor a contradiction\b", r"\btowards a contradiction\b",
        r"\breach(es)? a contradiction\b", r"\bleads? to a contradiction\b",
        r"\bcontradiction\b", r"\bcontradict(s|ed|ion)\b", r"\babsurd\b", r"⊥"
    ]):
        return CONTRADICTION
    return ABSTAIN

@labeling_function()
def lf_contra_incompatible(row):
    t = _t(row.text)
    # “assume P and not P”, “both … and not …”, “impossible”
    if _regex_any(t, [
        r"\bassume\b.+\band\b.+\bnot\b",
        r"\bboth\b.+\band\b.+\bnot\b",
        r"\bthis is impossible\b", r"\bcannot be\b",
    ]):
        return CONTRADICTION
    return ABSTAIN

# ------------------------------------------------------------------------------------
# INDUCTION
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_ind_keywords(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\bby induction\b|\binductive\b|\binductive step\b|\binduction hypothesis\b|\bIH\b",
        r"\bbase case\b|\banchor step\b",
        r"\bfor all\s+(n|k)\b", r"\bn\s*=\s*1\b|\bk\s*=\s*1\b",
        r"\bn\+?1\b|\bk\+?1\b",
        r"\\forall\s+(n|k)", r"\\mathbb\{n\}",
    ]):
        return INDUCTION
    return ABSTAIN

@labeling_function()
def lf_ind_structure(row):
    t = _t(row.text)
    # assume true for k, show k+1 / assume n, show n+1
    if _regex_any(t, [
        r"\bassume\b.+\bfor\s+k\b.+\b(show|prove)\b.+\bfor\s+k\+?1\b",
        r"\bassume\b.+\bfor\s+n\b.+\b(show|prove)\b.+\bfor\s+n\+?1\b",
    ]):
        return INDUCTION
    return ABSTAIN

# ------------------------------------------------------------------------------------
# CASE ANALYSIS
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_cases_keywords(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\bcase\s*1\b|\bcase\s*2\b|\bcases:\b",
        r"\bconsider the cases\b|\bsplit into cases\b|\bcasework\b",
        r"\bw\.?l\.?o\.?g\.?\b|\bwlog\b",
    ]):
        return CASE_ANALYSIS
    return ABSTAIN

@labeling_function()
def lf_cases_disjunction(row):
    t = _t(row.text)
    if "contradiction" in t:
        return ABSTAIN
    if _regex_any(t, [
        r"\beither\b.+\bor\b.+",
        r"\botherwise\b",
        r"\b(is|are)\s+(even|odd)\b.*\bor\b.*\b(even|odd)\b",
    ]):
        return CASE_ANALYSIS
    return ABSTAIN

# ------------------------------------------------------------------------------------
# EXISTENCE / CONSTRUCTION
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_exist_quantifiers(row):
    t = _t(row.text)
    # ∃, “there exists”, “for some”, “choose”, “pick”
    if _regex_any(t, [
        r"\\exists\b|∃", r"\bthere exists\b|\bthere is a\b|\bfor some\b",
        r"\bchoose\b|\bpick\b",
    ]):
        # But if negation is nearby, it's likely a counterexample → abstain here
        if _near(t, r"(\\exists|there exists|for some)", r"(\\neg|¬|\bnot\b)", 60):
            return ABSTAIN
        return EXISTENCE
    return ABSTAIN

@labeling_function()
def lf_constructive_verbs(row):
    t = _t(row.text)
    if _any(t, ["construct", "we construct", "define", "we define", "explicit construction", "let us define", "we build"]):
        if _regex_any(t, [NEG_CE]):  # if explicitly counterexample-ish, abstain
            return ABSTAIN
        return EXISTENCE
    return ABSTAIN

# ------------------------------------------------------------------------------------
# DIRECT (generic, kept LAST and heavily masked)
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_direct_generic(row):
    t = _t(row.text)
    # If any other tactic cues are present, abstain
    if _regex_any(t, [NEG_CONTRA, NEG_CP, NEG_IND, NEG_CASES, NEG_EXISTS, NEG_CE]):
        return ABSTAIN
    # Generic direct phrasing
    if _regex_any(t, [
        r"\bif\s+.+\s+then\s+.+",           # if ... then ...
        r"\bimplies\b", r"\\Rightarrow|⇒",
        r"\btherefore\b|\bthus\b|\bhence\b|\bit follows that\b",
        r"\bso that\b|\bas a result\b",
        r"\bfor all\b|\bfor every\b|\\forall",
    ]):
        return DIRECT
    return ABSTAIN

# ===== Export in priority order (most specific → most generic) =====
def get_tactic_lfs():
    return [
        # Counterexample
        lf_ce_exists_not, lf_ce_keywords,
        # Contrapositive
        lf_cp_structure, lf_cp_keywords,
        # Contradiction
        lf_contra_incompatible, lf_contra_keywords,
        # Induction
        lf_ind_structure, lf_ind_keywords,
        # Case analysis
        lf_cases_disjunction, lf_cases_keywords,
        # Existence / construction
        lf_exist_quantifiers, lf_constructive_verbs,
        # Direct (generic last)
        lf_direct_generic,
    ]