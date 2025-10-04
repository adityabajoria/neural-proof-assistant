# labeling_functions_subjects.py
# Stricter, LaTeX-aware subject LFs (weak supervision)

from snorkel.labeling import labeling_function
import re

# ===== Labels (keep IDs exactly as before) =====
ABSTAIN = -1
ALGEBRA = 0
LINEAR_ALGEBRA = 1
NUMBER_THEORY = 2
REAL_ANALYSIS = 3
COMPLEX_ANALYSIS = 4
TOPOLOGY = 5          # kept for compatibility; no LFs emit this below
COMBINATORICS = 6
PROBABILITY = 7
GEOMETRY = 8
CALCULUS = 9
LOGIC = 10
SET_THEORY = 11

SUBJECT_ID2NAME = {
    0: "Linear Algebra",
    1: "Number Theory",
    2: "Combinatorics",
    3: "Probability & Statistics",
    4: "Geometry",
    5: "Calculus",
    6: "Mathematical Logic",
    7: "Set Theory",
}

# ===== Utilities =====
def _t(x: str) -> str:
    return (x or "").strip().lower()

def _any(text: str, needles):
    return any(n in text for n in needles)

def _regex(text: str, pattern: str):
    try:
        return re.search(pattern, text, flags=re.I) is not None
    except re.error:
        return False

def _regex_any(text: str, patterns):
    return any(_regex(text, p) for p in patterns)

# Some cross-subject negatives to reduce collisions
NT_NEG = r"(group|ring|field|module|homomorph|isomorph|vector|matrix|eigen)"
ALG_NEG = r"(mod|congru|prime|gcd|\bdivides\b|\\(gcd|mid|equiv)|pmod)"
RA_NEG  = r"(derivative|integral|\\int|\\frac\{d\}|gradient|\\partial|\\nabla)"
CA_NEG  = r"(epsilon|delta|cauchy|sequence|series|\\lim|\blim\b|uniform)"
PROB_NEG= r"(graph|vertex|edge|pigeonhole|permutation|combination|\\binom)"
COMB_NEG= r"(probability|random|\\mathbb\{p\}|\\mathbb\{e\}|expectation|variance)"
LA_NEG  = r"(prime|modulo|congruence|gcd|divides)"
LOG_NEG = r"(group|ring|matrix|derivative|integral|prime)"

# ------------------------------------------------------------------------------------
# NUMBER THEORY  (very specific, LaTeX-heavy)
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_nt_keywords(row):
    t = _t(row.text)
    if _any(t, ["prime", "composite", "gcd", "congruence", "congruent", "mod", "modulo", "totient"]):
        if not _regex(t, ALG_NEG):  # avoid algebra words
            return NUMBER_THEORY
    return ABSTAIN

@labeling_function()
def lf_nt_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\b\d+\s*\\mid\s*\d+",                 # a \mid b
        r"\\gcd\s*\(",                          # \gcd(a,b)
        r"\\varphi\s*\(",                       # \varphi(n)
        r"\\equiv\s*.*?\\pmod\s*\{",            # a \equiv b \pmod{m}
        r"\\bmod\b",                            # \bmod
        r"\bmod\s*\d+\b",                       # mod 7
        r"\\mathbb\{Z\}",                       # integers
    ]):
        return NUMBER_THEORY
    return ABSTAIN

# ------------------------------------------------------------------------------------
# LINEAR ALGEBRA
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_la_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "matrix", "matrices", "vector", "vectors", "vector space", "basis", "span",
        "eigenvalue", "eigenvector", "eigenspace", "rank", "determinant",
        "null space", "column space", "row space", "orthogonal", "diagonalize"
    ]):
        if not _regex(t, LA_NEG):  # avoid number-theory words in passing
            return LINEAR_ALGEBRA
    return ABSTAIN

@labeling_function()
def lf_la_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\\det\s*\(",                          # \det(A)
        r"\brank\b",                            # rank
        r"[A-Za-z]\s*\^\s*(T|\\top)",           # A^T, A^\top
        r"\\begin\{bmatrix\}",                  # matrices
    ]):
        return LINEAR_ALGEBRA
    return ABSTAIN



@labeling_function()
def lf_alg_regex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\bcyclic\b",
        r"\bsymmetric\s+group\b|\bdihedral\s+group\b",
        r"\bgroup\s+theory\b",
    ]):
        return ALGEBRA
    return ABSTAIN

# ------------------------------------------------------------------------------------
# REAL ANALYSIS
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_ra_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "limit", "sequence", "converge", "convergent", "diverge", "bounded",
        "supremum", "infimum", "cauchy sequence", "uniform continuity",
        "monotone", "complete metric space", "compact interval"
    ]):
        if not _regex(t, RA_NEG):   # keep calculus words out
            return REAL_ANALYSIS
    return ABSTAIN

@labeling_function()
def lf_ra_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\blim\s*[_({]",                        # lim_{...} or lim(
        r"(?:\\epsilon|\\varepsilon|ε)",         # epsilon
        r"(?:\\delta|δ)",                        # delta
        r"\\sum\s*_",                            # \sum_{n=...} (often series)
    ]):
        return REAL_ANALYSIS
    return ABSTAIN

# ------------------------------------------------------------------------------------
# CALCULUS  (derivatives/integrals only; series handled by RA)
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_calc_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "derivative", "differentiable", "integral", "integrate",
        "gradient", "jacobian", "hessian",
        "partial derivative", "directional derivative"
    ]):
        return CALCULUS
    return ABSTAIN

@labeling_function()
def lf_calc_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\\int\b|∫",                             # integrals
        r"\\frac\{d\}\{dx\}|\bd/dx\b",            # d/dx
        r"\\partial|\b∂",                          # partial derivatives
        r"\\nabla",                                # gradient
    ]):
        return CALCULUS
    return ABSTAIN

# ------------------------------------------------------------------------------------
# COMBINATORICS
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_comb_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "pigeonhole", "permutation", "combination", "counting",
        "graph", "vertex", "edge", "matching", "coloring", "tree", "cycle",
    ]):
        if not _regex(t, COMB_NEG):  # avoid probability phrasing
            return COMBINATORICS
    return ABSTAIN

@labeling_function()
def lf_comb_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\\binom\s*\(",                         # \binom{n}{k}
        r"\bK[_-]?\d+\b",                        # K_n
        r"\\mathrm\{deg\}\s*\(",                 # degree(v)
    ]):
        return COMBINATORICS
    return ABSTAIN

# ------------------------------------------------------------------------------------
# PROBABILITY & STATS
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_prob_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "probability", "random variable", "distribution", "independent",
        "expectation", "expected value", "variance", "standard deviation",
        "markov", "bayes", "binomial", "normal", "gaussian", "poisson", "geometric"
    ]):
        if not _regex(t, PROB_NEG):  # keep combinatorics words out
            return PROBABILITY
    return ABSTAIN

@labeling_function()
def lf_prob_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"(\\mathbb\{P\}|\bP)\s*\(",            # \mathbb{P}(A) or P(A)
        r"(\\mathbb\{E\}|\bE)\s*\[",            # \mathbb{E}[X] or E[X]
        r"\\operatorname\{Var\}|\bVar\s*\(",    # Var(X)
        r"\\operatorname\{Cov\}|\bCov\s*\(",    # Cov(X,Y)
        r"\\Pr\s*\(",                           # \Pr(A)
    ]):
        return PROBABILITY
    return ABSTAIN

# ------------------------------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_geom_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "triangle", "circle", "radius", "diameter",
        "angle", "polygon", "perpendicular", "parallel", "euclidean"
    ]):
        return GEOMETRY
    return ABSTAIN

@labeling_function()
def lf_geom_regex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\bright\s+triangle\b|\bisosceles\b|\bequilateral\b",
        r"\bangle\s*bisector\b",
        r"\b\w+\s*is\s*perpendicular\s*to\s*\w+",
        r"\b\w+\s*is\s*parallel\s*to\s*\w+",
    ]):
        return GEOMETRY
    return ABSTAIN

# ------------------------------------------------------------------------------------
# LOGIC
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_logic_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "proposition", "predicate", "quantifier", "tautology",
        "negation", "conjunction", "disjunction", "equivalent", "equivalence"
    ]):
        if not _regex(t, LOG_NEG):
            return LOGIC
    return ABSTAIN

@labeling_function()
def lf_logic_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\\forall|\b∀",                        # quantifiers
        r"\\exists|\b∃",
        r"\\Rightarrow|\\to|\bimplies\b",       # implication
        r"\\iff|\biff\b|\bif and only if\b",
        r"\\neg|\b¬|\bnot\b",
    ]):
        return LOGIC
    return ABSTAIN

# ------------------------------------------------------------------------------------
# SET THEORY
# ------------------------------------------------------------------------------------
@labeling_function()
def lf_set_keywords(row):
    t = _t(row.text)
    if _any(t, [
        "set", "subset", "proper subset", "superset",
        "union", "intersection", "disjoint",
        "cardinality", "power set", "zorn", "axiom of choice"
    ]):
        return SET_THEORY
    return ABSTAIN

@labeling_function()
def lf_set_latex(row):
    t = _t(row.text)
    if _regex_any(t, [
        r"\\subseteq|\\subset|\b⊆|\b⊂",
        r"\\supseteq|\\supset|\b⊇|\b⊃",
        r"\\cup|\b∪|\bcup\b",
        r"\\cap|\b∩|\bcap\b",
        r"\\in\b|\b∈|\bnotin|\b∉",
        r"\\mathcal\{P\}\s*\(",                 # power set
    ]):
        return SET_THEORY
    return ABSTAIN

# ===== Export: ordered from most specific to more generic =====
# (If you use Snorkel LabelModel, order doesn’t matter; if you do a
# simple override/last-win aggregator, this order helps.)
def get_subject_lfs():
    return [
        # very specific, latex-heavy first
        lf_nt_latex, lf_nt_keywords,
        lf_la_latex, lf_la_keywords,
        lf_alg_regex,lf_ra_latex, lf_ra_keywords,
        lf_calc_latex, lf_calc_keywords,
        lf_prob_latex, lf_prob_keywords,
        lf_comb_latex, lf_comb_keywords,
        lf_geom_regex, lf_geom_keywords,
        lf_logic_latex, lf_logic_keywords,
        lf_set_latex, lf_set_keywords,
    ]