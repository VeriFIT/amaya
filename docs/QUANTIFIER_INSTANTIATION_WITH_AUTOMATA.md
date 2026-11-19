# Counterexample-guided quantifier instantiation over automata

Status: **proposal, not implemented.** No measurement of the proposed technique exists. Section 8 lists
what remains unquantified.

## 1. Amaya in brief

Amaya decides satisfiability of linear integer arithmetic (LIA) formulae given in SMT-LIB 2.

### 1.1 Automata representation

| Formula element | Automaton operation |
|---|---|
| Atom `a·x ≤ b`, `a·x = b`, congruence `a·x ≡ b (mod m)` | Direct construction of an automaton over the alphabet `{0,1}^k` (one track per variable, least significant bit first, two's complement). States of an inequality automaton are integer residues of `b`. |
| `φ ∧ ψ` | Intersection (product construction) |
| `φ ∨ ψ` | Union |
| `¬φ` | Determinization, then complement |
| `∃x. φ` | Projection of track `x`, then padding closure |
| Satisfiability | Non-emptiness of the language; `find_model` returns an accepting word, decoded into integers |

Each operation output is minimized when `-m hopcroft` is set (implied by `--fast`). Projection produces an
NFA, so a subsequent complement or minimization determinizes it.

The MTBDD backend (`--backend MTBDD`, implied by `--fast`) stores the transition function as MTBDDs
(C++ library in `mtbdd-backend/`, Sylvan BDD package).

### 1.2 Evaluation pipeline

1. Parsing into an AST of `Relation`, `Congruence`, `Var` (Bool), `AST_Connective`, `AST_Negation`,
   `AST_Quantifier` (`amaya/relations_structures.py`).
2. Formula-level optimizations selected with `-O` (`amaya/parse.py:optimize_formula_structure`; fixpoint
   scheduler `amaya/preprocessing/pipeline.py`, enabled with `--opt-fixpoint`). Passes relevant here:
   `miniscope` (antiprenexing), `unit-coefficient-fm` (Fourier-Motzkin elimination of variables occurring only
   with coefficients ±1; `--ucfm-guarded-bounds` admits bounds guarded by Boolean literals).
3. Evaluation strategy, one of:

| Strategy | Entry point | Operation |
|---|---|---|
| Default | `amaya/parse.py:evaluate_prepared_formula_with_automata` | Bottom-up construction of one automaton for the whole formula (`amaya/parse.py:run_evaluation_procedure`). `--early-stop-lazy-construction` decides an existentially quantified conjunction of atoms by exploring the product of the atom automata until the first accepting state (`amaya/parse.py:try_decide_atom_conjunction_with_early_stop`); the result is not determinized and is used only for the emptiness test and model extraction. |
| Top-level SAT (`--use-toplevel-sat`, experimental) | `amaya/sat_toplevel.py:solve_with_toplevel_sat` | A CDCL SAT solver (python-sat) enumerates assignments of free Bool variables; each residual LIA formula is decided by the default strategy; a refuted assignment is blocked with a clause restricted to the variables reported by `amaya/sat_toplevel.py:_substitute_bool_vars_tracking_relevance`. `--lift-toplevel-bool-existentials` makes Bool variables of positive-position existentials free so that the SAT solver assigns them. |
| DPLL(T) over automata (`--use-dpllt-automata`, experimental) | `amaya/dpllt_automata.py:solve_with_dpllt_over_automata` | The formula is split into a general part and a positive-existential part; minimal implicants of the Boolean abstraction of the latter are enumerated with a SAT solver; each implicant is first checked by bound propagation (`amaya/dpllt_automata.py:find_bounds_refutation`) and otherwise evaluated as a conjunction of atoms with automata and intersected with the automaton of the general part. See `docs/DPLLT_WITH_AUTOMATA.md`. |

### 1.3 Cost properties of the automata operations

| Operation | Cost property |
|---|---|
| Atom construction | Number of states bounded by the range of reachable residues; grows with `Σ|a_i|` and with the bit length of `b` (property of the LSBF construction, not measured in Amaya). |
| Intersection of automata over disjoint variable sets | Number of states of the product equals the product of the operand sizes (synchronous reading of all tracks). |
| Projection followed by determinization | Exponential in the worst case. |
| Number of tracks | Every automaton is defined over the alphabet of the run; MTBDD operations scale with the number of tracks the transition MTBDDs depend on. |

## 2. Target formula class

Benchmark: `/tmp/Integer_Transition_Systems_From_T2_fibcall_t2_fixed_ari__04.smt2` (source
`tpdb-ari/Integer_Transition_Systems/From_T2/fibcall.t2_fixed.ari`, generator `gen_its_hard`, status
`unknown` in the file).

Shape: `∃P ∀X ψ(P, X)`.

| Component | Content |
|---|---|
| `P` | Template parameters for 4 locations: guard Bools `e_lo_l_i`, `e_hi_l_i`, `e_ij_l_i_j`, `e_ji_l_i_j` and integer constants `c_*`; 48 Bool, 48 Int. |
| `X` | `x_0, x_1, x_4, x_5, y_0, y_1, y_4, y_5` (8 Int). |
| `ψ` | Conjunction of 5 Horn clauses `inv_pre(P, x) ∧ trans(x, y) → inv_post(P, y)` and one safety clause `inv_2(P, x) → ¬(x_0 ≥ 2)`; 120 guarded atoms `e → atom` in total. |
| `inv_l(P, x)` | Conjunction of guarded bounds `e → c ≤ x_i`, `e → x_i ≤ c`, `e → x_i - x_j ≤ c` (interval and octagon-difference template). |

All atoms have coefficients in `{-1, 0, 1}`.

The formula encodes the existence of an inductive invariant from the template that proves the safety
clause. It is unsatisfiable (derivation by hand; not confirmed by a solver, z3 did not terminate within 30 s):

1. Clause `true ∧ y = x → inv_3(y)` forces every `e_*_3_*` to False.
2. Clause `inv_3(x) ∧ y = (1, 0, 2, 30) → inv_2(y)` gives `inv_2(1, 0, 2, 30)`.
3. Clause `inv_2(x) ∧ y = x → inv_0(y)` gives `inv_0(1, 0, 2, 30)`.
4. Clause `inv_0(x) ∧ x_4 ≤ x_5 ∧ y = (x_0 + x_1, x_0, x_4 + 1, x_5) → inv_2(y)` gives `inv_2(1, 1, 3, 30)`;
   repeating steps 3-4 gives `inv_2(2, 1, 4, 30)`.
5. The safety clause requires `x_0 < 2` for every point of `inv_2`; `x_0 = 2` violates it.

The refutation uses 5 ground instances of the universal clauses and does not depend on `P`.

## 3. Measurements with the existing strategies

All runs: `./venv/bin/python run-amaya.py --fast -O all <flags> --debug get-sat <benchmark>`, `timeout 80`,
2026-11-17, branch `guarded-fourier-motzkin` at `0913a09`.

| Flags | Result | Observed bottleneck |
|---|---|---|
| `--opt-fixpoint --lift-toplevel-bool-existentials --ucfm-guarded-bounds --use-toplevel-sat --toplevel-sat-optimize-residuals` | timeout | 10 SAT iterations; from iteration 4 every blocking clause contains all 24 assigned literals; iteration 9 took 18 s; iteration 10 did not finish a projection over an automaton with 145 tracks and 1568 states. 5 of 7 residual optimizations were discarded because a pass allocated variables (`amaya/sat_toplevel.py:_optimize_residual`). |
| `--opt-fixpoint --ucfm-guarded-bounds` (lift and top-level SAT disabled) | timeout | Chain of single-variable existentials over `c_*`. NFA size before successive determinizations: 12, 23, 57, 151, 441, 1247, 3729; determinization of the last produced 18098 states. 7 of ~24 projections completed. |
| `--ucfm-guarded-bounds` | timeout | Same pattern: 16, 32, 82, 221, 861, 2271, 6803. |

Growth factor per projected `c_*` variable: 2.6-3.9 (ratios of consecutive values above).

Interpretation of the measurements:

* The top-level SAT loop needs up to `2^24` theory calls on an unsatisfiable formula, because the blocking
  clause generalization of `amaya/sat_toplevel.py:_substitute_bool_vars_tracking_relevance` is syntactic and
  does not shrink when the residual is refuted by arithmetic.
* The default strategy materializes the solution set of the template over all `c_*` and `e_*` tracks; its
  size grows geometrically in the number of projected parameters.
* Neither strategy separates the quantifier alternation (`∀X`, 8 variables) from the parameter search
  (`P`, 96 variables).

Generalizing the blocking clause by re-evaluating the formula with only the relevant variables unassigned is
unsound (counterexample: `root = (s ∧ t) ∨ R`, `R` unsatisfiable, refuted assignment `s = t = False`, relevance
set `{s}`; `root[t := False]` is unsatisfiable while `s = t = True` is a model). Its sound form leaves all
variables unassigned, which is the default strategy above.

## 4. Technique: counterexample-guided quantifier instantiation (CEGQI)

References: Reynolds, Deters, Kuncak, Tinelli, Barrett, "Counterexample-Guided Quantifier Instantiation for
Synthesis in SMT", CAV 2015; Bjørner, Janota, "Playing with Quantified Satisfaction", LPAR 2015.

State: a finite set `G` of instances `ψ(P, X_k)`, initially empty. `G` is quantifier-free in `P`.

| Step | Query | Result |
|---|---|---|
| Synthesis | Find `P*` with `P* ⊨ ⋀ G`. | None: **UNSAT**. |
| Verification | Find `X*` with `X* ⊨ ¬ψ(P*, X)`. | None: **SAT**, model `P*`. Otherwise add an instance derived from `X*` to `G` and repeat. |

Soundness:

* UNSAT: `∀X ψ(P, X)` entails every element of `G`, so `G` is a relaxation of the formula.
* SAT: the verification step decides `∀X ψ(P*, X)` exactly.

## 5. Integration with automata

### 5.1 Verification step

After substitution of `P*`:

1. Assigned guards fold (`amaya/sat_toplevel.py:substitute_bool_vars`).
2. Every `c_*` becomes an integer constant; it appears only as the right-hand side `b` of atoms.
3. `¬ψ(P*, X)` is a disjunction over the 6 clauses of `inv_pre ∧ trans ∧ ¬inv_post`; `¬inv_post` is a
   disjunction of single negated atoms. The result is a union of at most 6 × 12 conjunctions of atoms, each
   existentially closed over at most 8 variables; the equations of `trans` reduce this to 4 free variables
   per conjunction.

| Property | Consequence for automata |
|---|---|
| Each disjunct is an existentially closed conjunction of atoms | Input shape of `amaya/parse.py:try_decide_atom_conjunction_with_early_stop`; exploration stops at the first accepting state; no complement and no determinization. |
| The accepting word is a model | `find_model` yields `X*`. |
| 4-8 tracks per automaton | The 145-track automaton of the top-level SAT loop (Section 3) does not occur. |
| `P*` enters only through `b` | State count depends on the bit length of the `P*` constants, not on the number of parameters. |
| Disjuncts are independent | They can be decided one at a time, stopping at the first counterexample. |

The quantifier alternation is handled only in this step, over 4-8 variables with `P` fixed.

### 5.2 Synthesis step

`G` is quantifier-free over 96 parameters with clauses such as `¬e ∨ c ≤ k`. A single automaton for `G`
materializes the joint solution set over all `c_*`, which is the geometric growth measured in Section 3.
The step is therefore assigned to the SAT and bound-reasoning components, with automata used only for
conjunctions of atoms:

| Component | Role |
|---|---|
| SAT solver over the Boolean abstraction of `G` | Fixes the guards and selects the atoms to assert. New instances are added as clauses incrementally. |
| `amaya/dpllt_automata.py:find_bounds_refutation` | Point instances produce atoms `c ≤ k`, `c ≥ k`, `c_1 - c_2 ≤ k`; bound propagation (`--dpllt-bound-propagation-rounds`) refutes conflicting sets without constructing an automaton. |
| `amaya/dpllt_automata.py:_try_decide_iteration_by_early_stopping_lazy_construction` | Decides an asserted conjunction not refuted by bound propagation and returns `P*` without materializing the solution set. |

`G` contains guards in both polarities. Whether `amaya/dpllt_automata.py:split_formula_into_phi_and_chi`
and its monotone abstraction accept `G` has not been checked; the non-monotone abstraction
`amaya/sat_toplevel.py:abstract_to_sat_formula` is the alternative.

`amaya/mtbdd_automatons.py:MTBDD_NFA.find_model` explores depth-first
(`mtbdd-backend/src/base.cpp:NFA::find_model`); the returned word is not the shortest one, so the magnitude
of the `P*` constants is not bounded by the search. A shortest-word variant (breadth-first search) would keep
the `b` constants of the next verification step minimal in bit length.

### 5.3 Comparison

| | Top-level SAT loop | Default strategy | CEGQI |
|---|---|---|---|
| Quantifier alternation decided | in every residual | in one automaton | only in verification, `P` fixed |
| Tracks per automaton | up to 145 (measured) | all of `P` and `X` | 4-8 in verification |
| Complement or determinization over `c_*` tracks | yes | yes, factor 2.6-3.9 per variable (measured) | none |
| Automaton calls | up to `2^24` | 1 | rounds × at most 72 small conjunctions, stopping at the first model |

## 6. Instance selection and termination

With point instances `ψ(P, X*)` the loop is not guaranteed to terminate: `c_*` range over all integers and
each `P*` can move one bound to exclude the last `X*`. The counterexamples returned by the verification step
are arbitrary violating points, not reachable states, so they need not converge to the 5 instances of
Section 2.

| Instance selection | Termination | Integration |
|---|---|---|
| Point `X*` from `find_model` | not guaranteed | existing components only |
| Model-based projection: instantiate `X` with terms over `P` selected by the model of the verification step (e.g. `x_0 := c_lo_0_0` when that lower bound is tight under the model) | finite number of candidate terms per clause | With coefficients ±1, Fourier-Motzkin elimination is exact over the integers; the selection of the tight bound is the bound selection performed by `-O unit-coefficient-fm` (`amaya/preprocessing/`, pass `unit-coefficient-fm`). The instances become difference constraints `c_1 - c_2 ≤ k`, handled by `find_bounds_refutation`, which is not complete for cycles of difference constraints; unrefuted sets fall back to the lazy construction. |
| Forward ground seeding: substitute the ground points produced by the clause equations (e.g. `y = (1, 0, 2, 30)`) into the universal clauses and propagate for a bounded number of steps | bounded unrolling | no automaton involved; produces the refutation of Section 2 with 5 instances |

## 7. Implementation outline

Components to add, in order:

1. Detection of the `∃P ∀X ψ` shape after preprocessing; fall through to the default strategy otherwise.
2. Substitution of integer constants for `c_*` in `Relation` nodes. Existing related helpers:
   `amaya/preprocessing/unbound_vars.py:_substitute_var_with_value`,
   `amaya/preprocessing/eval.py:apply_substitution`.
3. Verification step: split `¬ψ(P*, X)` into conjunctions of atoms; decide each with
   `amaya/parse.py:try_decide_atom_conjunction_with_early_stop`; decode `X*` with
   `amaya/parse.py:convert_binary_model_into_decadic`.
4. Synthesis step: incremental SAT over the abstraction of `G`, bound refutation, early-stop lazy
   construction (Section 5.2).
5. Forward ground seeding of `G`.
6. Model-based projection, if point instances do not converge.
7. A CLI flag following the existing convention (`(EXPERIMENTAL)` help text, `run-amaya.py`), disabled by
   default.

## 8. Unquantified

* Runtime of CEGQI on the benchmark, with point instances, with model-based projection, and with forward
  ground seeding.
* Number of rounds; whether point instances diverge on this benchmark.
* Cost of the synthesis step as `G` grows.
* Size of the verification automata for the `P*` constants the synthesis step returns.
* Whether the 145 tracks are the cause of the slow projection in iteration 10 of the top-level SAT run.
* Applicability to other instances of `Integer_Transition_Systems/From_T2`; one benchmark was inspected.
* Unsatisfiability of the benchmark has been derived by hand only (Section 2).
