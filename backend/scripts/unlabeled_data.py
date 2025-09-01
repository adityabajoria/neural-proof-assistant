#!/usr/bin/env python3
import random
import pandas as pd

# --- Primitive symbols ---
RANDOM_VARS = ["x","y","z","n","m","k","u","v","w","a","b","c","d","p","q","r","s","t"]
GREEK = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","rho","sigma","tau","phi","psi","omega"]
SETS = ["A","B","C","D","E","F","G","H","S","T","U","V","W"]
NUM_SETS = ["N","Z","Q","R"]
GRAPHS = ["G","H","K_n","P_n","C_n"]
MATS = ["M","N","P"]
SPACES = ["R^2","R^3","R^n","C^n"]
FUNCS = ["f","g","h","T","F"]
OPS = ["+","-","*","/"]

def choice(xs): return random.choice(xs)
def var(): return choice(RANDOM_VARS)
def cvar(): return var() + str(random.randint(1, 9)) if random.random() < 0.3 else var()
def ivar(): return str(random.randint(1, 50))
def grek(): return choice(GREEK)
def setname(): return choice(SETS)
def numset(): return choice(NUM_SETS)
def graphname(): return choice(GRAPHS)
def matname(): return choice(MATS)
def func(): return choice(FUNCS)
def space(): return choice(SPACES)
def op(): return choice(OPS)

# --- Template families ---
def number_theory():
    v1, v2, k = cvar(), cvar(), ivar()
    return choice([
        f"For all {v1} in {numset()}, there exists {v2} such that {v1} = {k}{v2}.",
        f"If {v1} divides {v2} and {v2} divides {k}, then {v1} divides {k}.",
        f"{v1} is prime implies {v1} has no nontrivial divisors.",
        f"There exists infinitely many primes greater than {k}.",
        f"The gcd({v1}, {v2}) can be expressed as a linear combination of {v1} and {v2}."
    ])

def set_theory():
    A, B, C, v1 = setname(), setname(), setname(), cvar()
    return choice([
        f"For all {v1}, if {v1} in {A} and {A} subset {B}, then {v1} in {B}.",
        f"{A} ∩ ({B} ∪ {C}) = ({A} ∩ {B}) ∪ ({A} ∩ {C}).",
        f"{A} \\ {B} = {A} ∩ {B}^c.",
        f"({A} ∪ {B})^c = {A}^c ∩ {B}^c.",
        f"There exists a bijection between {A} and {B} under certain conditions."
    ])

def graph_theory():
    G, H, n = graphname(), graphname(), ivar()
    return choice([
        f"Let {G} be a simple graph with {n} vertices; then the sum of degrees is even.",
        f"If {G} is connected, there exists a spanning tree of {G}.",
        f"By the pigeonhole principle, there exist two vertices with the same degree.",
        f"If {G} is bipartite, it contains no odd cycle.",
        f"There exists a path between any two vertices in a connected {G}."
    ])

def linear_algebra():
    M, sp, v1, v2 = matname(), space(), cvar(), cvar()
    return choice([
        f"Let {func()}: {sp} -> {sp} be linear; then {func()}({v1} + {v2}) = {func()}({v1}) + {func()}({v2}).",
        f"The null space of a matrix equals the set of solutions to {M}{v1} = 0.",
        f"The rank-nullity theorem implies dim(ker({func()})) + dim(im({func()})) = dim({sp}).",
        f"If {M} is invertible, then det({M}) ≠ 0.",
        f"Eigenvalues of a triangular matrix are its diagonal entries."
    ])

def calculus():
    v1, a, b = cvar(), ivar(), ivar()
    return choice([
        f"If a function is differentiable at {v1}, it is continuous at {v1}.",
        f"By the mean value theorem, there exists c in ({a}, {b}) such that f'(c) = (f({b}) - f({a})) / ({b} - {a}).",
        f"The derivative of a sum is the sum of derivatives.",
        f"If a series of continuous functions converges uniformly, the limit is continuous.",
        f"Integration by parts: ∫ u dv = uv - ∫ v du."
    ])

def probability_stats():
    v1, v2 = cvar(), cvar()
    return choice([
        f"If {v1} and {v2} are independent, then P({v1} ∩ {v2}) = P({v1}) P({v2}).",
        f"By the law of total probability, P(A) = Σ P(A|B_i)P(B_i).",
        f"Expectation is linear even when variables are not independent.",
        f"Var(X) = E[X^2] - (E[X])^2.",
        f"Chebyshev’s inequality bounds tail probabilities via variance."
    ])

def logic_templates():
    v1, v2 = cvar(), cvar()
    return choice([
        f"For all {v1}, if P({v1}) then Q({v1}).",
        f"There exists {v1} such that P({v1}) and not Q({v1}).",
        f"(P -> Q) is logically equivalent to (¬Q -> ¬P).",
        f"By contradiction: assume not P, derive a contradiction, conclude P.",
        f"Existential instantiation introduces a fresh witness {v2}."
    ])

def misc_mathy_noise():
    v1, v2 = cvar(), cvar()
    return choice([
        f"{v1} {op()} {v2} = {v2} {op()} {v1} under commutativity.",
        f"Consider the mapping {func()} with unknown regularity.",
        f"Let us fix parameters {grek()} and {grek()} in ({ivar()}, {ivar()}).",
        f"Boundary conditions are omitted for brevity.",
        f"Assume standard smoothness; details are routine."
    ])

# --- Families + weights ---
FAMILIES = [
    number_theory, set_theory, graph_theory,
    linear_algebra, calculus, probability_stats,
    logic_templates, misc_mathy_noise
]
WEIGHTS = [0.15, 0.12, 0.12, 0.15, 0.12, 0.12, 0.12, 0.10]

def generate_sample():
    """Generate one proof: sometimes 1 statement, sometimes multiple joined."""
    num_statements = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    parts = []
    for _ in range(num_statements):
        family = random.choices(FAMILIES, weights=WEIGHTS, k=1)[0]
        parts.append(family())
    # Join with logical connectors for variety
    connectors = [" Therefore, ", " Hence, ", " Moreover, ", " Also, "]
    text = parts[0]
    for extra in parts[1:]:
        text += random.choice(connectors) + extra
    return text

def main(n=10000, out="unlabeled_goals.csv"):
    seen = set()
    samples = []
    while len(samples) < n:
        s = generate_sample()
        if s not in seen:
            seen.add(s)
            samples.append(s)
    df = pd.DataFrame(samples, columns=["text"])
    df.to_csv(out, index=False)
    print(f"[INFO] Generated {len(samples)} distinct mathy samples -> {out}")

if __name__ == "__main__":
    main()