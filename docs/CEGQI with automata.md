# Design: Exists-Forall Solving with Automata Projection Lemmas (CEGQI) in Amaya

Nov 17, 2026 · @Michal

## Summary and motivation

We add an experimental strategy, `--use-cegqi-automata`, that decides formulas of shape `∃P ∀X ψ(P, X)` by alternating candidate search over `P` with verification over `X`, and learns lemmas by exact automata projection. Status: design, not implemented; no measurement of the new strategy exists.

The existing strategies time out on the motivating benchmark (`Integer_Transition_Systems/From_T2/fibcall.t2_fixed.ari`, instance 04: 48 Bool and 48 Int parameters, 8 universal Int variables, 120 guarded atoms). Measurements from 2026-11-17, branch `guarded-fourier-motzkin` at `0913a09`, timeout 80 s:

| Strategy | Failure |
| --- | --- |
| Top-level SAT | Blocking clauses do not shrink (all 24 literals from iteration 4); iteration 10 stalls projecting a 145-track, 1568-state automaton. |
| Default, `--opt-fixpoint --ucfm-guarded-bounds` | Projecting the `c_*` parameters one by one grows the NFA 12 → 3729 states; the last determinization gives 18098 states; 7 of \~24 projections done. |
| Default, `--ucfm-guarded-bounds` | Same pattern, 16 → 6803 states. |

Both strategies decide the quantifier alternation over all 96 parameter tracks at once. The new strategy decides it only for one clause at a time, with the universal variables projected and the parameter set limited to that clause.

The role of automata is exact projection: a counterexample for one candidate is generalized into the full set of bad constants for that clause, for any coefficients and congruences. This is the property the design must demonstrate; on unit-coefficient instances, non-automata reasoning may do most of the work, and the evaluation reports that separately.

## Goals and non-goals

Goals:

- Decide `∃P ∀X ψ` formulas without building an automaton over all parameter tracks.
- Keep every automaton's track count bounded by one clause's variables, independent of the total number of parameters.
- Guarantee termination for any LIA matrix, without relying on unit coefficients.
- Reuse existing components: substitution, early-stop lazy construction, bound refutation, SAT abstraction, model decoding.
- Ship behind an experimental flag, off by default, with fall-through to the default strategy.

Non-goals:

- Formulas with deeper alternation (`∃∀∃` and beyond); they fall through.
- Replacing the default, top-level SAT, or DPLL(T) strategies.
- A complete difference-logic theory solver; bound propagation stays a filter, and automata remain the complete fallback.
- Changes to the MTBDD backend beyond a shortest-word model search (Section on synthesis).

## Target formula class and shape detection

The strategy applies when, after preprocessing, the formula is `∃P ∀X ⋀ᵢ Cᵢ(Pᵢ, X)` with a quantifier-free matrix in Presburger arithmetic. `P = B ∪ K` splits into Bool parameters `B` and Int parameters `K`; `X` are Int (Bool universals are allowed and become tracks). `Pᵢ ⊆ P` is the set of parameters clause `Cᵢ` mentions.

The strategy pays off when the `Pᵢ` are small compared with `P`. In template invariant synthesis, `Pᵢ` is the template of the two locations a clause connects (about 24 of 96 parameters on the motivating benchmark).

Detection, run after the `-O` pipeline (including `miniscope`):

1. Strip top-level `∃` blocks into `P`, including free variables of the assertion (they are implicitly existential).
2. Require the remaining body to be one `∀` block, possibly distributed over the conjuncts by miniscoping; collect its variables into `X`.
3. Require each conjunct's body to be quantifier-free after (2); otherwise fall through.
4. Record `Pᵢ` per conjunct and report `max |Pᵢ| / |P|` in debug output; this ratio predicts the benefit.

Any formula outside this shape falls through to the default strategy unchanged.

## Algorithm overview

The loop alternates synthesis over the parameters and verification over the universal variables; automata do the exact work in verification and generalization, never over all parameter tracks at once.

&#91;embedded content: CEGQI loop · 5 steps, 2 decisions, 1 learning edge\]

State carried across rounds: the SAT solver with its blocking clauses, seeded instances, and the lemma store keyed by `(i, d, γ)`. Each round either ends the run or adds one lemma with a new key.

## Verification step

Given a candidate `P* = (B*, K*)`, verification finds a clause `Cᵢ` and a point `X*` with `¬Cᵢ(P*, X*)`, or proves none exists.

Per clause, in a fixed order (clauses that failed most recently first):

1. Substitute `B*` and fold guards (`amaya/sat_toplevel.py:substitute_bool_vars`).
2. Substitute `K*` into `Relation` nodes; constants move into the right-hand side `b`.
3. Push the negation through: `¬Cᵢ` becomes a disjunction of existentially closed conjunctions of atoms (for a Horn clause `pre ∧ trans → post`: one disjunct per negated atom of `post`).
4. Decide each disjunct with `amaya/parse.py:try_decide_atom_conjunction_with_early_stop`; on the first accepting state, decode `X*` with `convert_binary_model_into_decadic`.
5. Return `(i, d, X*)`, the clause, the violated disjunct, and the point.

Properties:

- Tracks per automaton: the variables of one disjunct, 4 to 8 on the benchmark.
- No complement and no determinization: the early-stop product is only tested for emptiness.
- State count depends on the bit length of `K*`, not on `|P|`.
- On the final round of a satisfiable formula, every disjunct is shown empty, which explores each product fully. This is the worst-case cost of verification and is measured separately.

A cheap pre-check runs before step 4: if bound propagation (`find_bounds_refutation`) refutes the disjunct, it is empty and no automaton is built.

## Generalization by projection

From `(i, d, X*)`, generalization builds the exact set of integer parameter values that violate disjunct `d` of clause `Cᵢ` under the current guards, and learns its complement as a lemma. This turns one failed candidate into a region of failed candidates; `X*` itself is projected away, so the lemma covers every counterexample, not only the one found.

Construction:

1. Guard restriction `γ`: the values in `B*` of the Bool parameters that occur in disjunct `d`. Guards of other atoms of `Cᵢ` are left out, so `γ` is usually far smaller than `B*` restricted to `Pᵢ`.
2. Build the disjunct with `γ` substituted and `Kᵢ` symbolic: a conjunction of atoms over `Kᵢ ∪ X`.
3. Construct its automaton (product of atom automata), project the `X` tracks, determinize, minimize: this is `Bad_{i,d,γ}` over the `Kᵢ` tracks only.
4. Learn the lemma below and record the key `(i, d, γ)` so the same lemma is never rebuilt.

```latex
\gamma \;\rightarrow\; K_i \notin \mathrm{Bad}_{i,d,\gamma}, \qquad \mathrm{Bad}_{i,d,\gamma} = \{\, K_i \mid \exists X.\; \neg C_i^{d}(\gamma, K_i, X) \,\}
```

The complement is not materialized: the synthesis step uses `Bad` through emptiness of `candidate ∧ Bad`, and its complement only inside the lazy product, where a deterministic `Bad` is complemented by swapping final states.

Granularity options, chosen by flag, all sound:

| Option | Lemma covers | Cost |
| --- | --- | --- |
| `disjunct` (default) | one negated atom of `post` | smallest projection |
| `clause` | whole `¬Cᵢ`, all disjuncts | one larger projection, fewer rounds |
| `point` | `ψ(P, X*)` only, no projection | baseline for evaluation; no termination guarantee |

Fallback: if projection or determinization exceeds a state budget (`--cegqi-projection-state-limit`), learn the `point` instance for this round and log the overrun. Soundness is unaffected; only the termination guarantee is lost for that key.

Why automata here: projection over `X` is exact for any LIA, including non-unit coefficients and congruences. Fourier-Motzkin (used by `unit-coefficient-fm`) is exact only for coefficients ±1; Cooper-style model-based projection needs divisibility terms. On unit-coefficient instances, `Bad` is a union of difference-constraint polyhedra and could be computed without automata; the evaluation must separate these cases.

## Synthesis step

Synthesis finds `P* = (B*, K*)` satisfying all learned lemmas and seeded instances, or reports UNSAT. It is a small DPLL(T) loop: SAT chooses `B*`, the theory part finds `K*` from bound reasoning and a lazy product of lemma automata.

State:

- An incremental SAT solver (python-sat) over `B` plus abstraction variables for atoms of seeded instances, built with `amaya/sat_toplevel.py:abstract_to_sat_formula` (non-monotone; `G` has guards in both polarities).
- The lemma store: for each key `(i, d, γ)`, the deterministic, minimized automaton `Bad_{i,d,γ}` over `Kᵢ` tracks, cached across rounds.

One synthesis call:

1. SAT returns `B*` and a set of asserted atoms over `K`. UNSAT here means the formula is UNSAT.
2. Active lemmas: those whose `γ` holds under `B*`.
3. Bound filter: run `find_bounds_refutation` on the asserted atoms. A refutation yields a core; block it in SAT and go to 1.
4. Lazy product of the asserted atom automata and the complements of active `Bad` automata (complement by swapping final states; each `Bad` is deterministic). Search for an accepting word.
5. Found: decode `K*`; components of `K` not on any track are set to 0. Return `(B*, K*)`.
6. Empty: block `B*` with a clause over the union of the `γ` of the active lemmas and the literals of the asserted atoms; optionally shrink it by deletion-based core minimization under a time budget. Go to 1.

Model search order: `MTBDD_NFA.find_model` is depth-first (`mtbdd-backend/src/base.cpp:NFA::find_model`). Add a breadth-first variant that returns a shortest accepting word. With LSBF encoding, shortest means smallest bit length, which keeps the next verification automata small and keeps candidates near zero.

Cost caveat: early stop helps only when a candidate exists. Step 6 explores the whole product, and the last synthesis call on an UNSAT formula always reaches it. The bound filter in step 3 and seeded instances (next section) exist to keep that case cheap; the product size there is a measured quantity, not an assumed one.

## Forward ground seeding

Before the first round, seeding adds ground instances of the clauses along concrete forward executions, so shallow refutations are found by SAT alone. It is optional (`--cegqi-seed-depth N`, default 0) and needs no automata.

Procedure:

1. Initial points: for clauses whose consequent holds for all `y` (facts), instantiate at `y = 0`.
2. For each point `p` reached at location `l`, instantiate every clause with antecedent `inv_l` at `x = p`. If the transition's guard folds to true and `y` is determined by equations, the result `y = t(p)` is a new point at the target location.
3. Repeat to depth `N`; deduplicate points per location.

On the motivating benchmark this yields a 7-instance refutation chain (`inv_3(0)` → `inv_2(1,0,2,30)` → `inv_0` → `inv_2(1,1,3,30)` → `inv_0` → `inv_2(2,1,4,30)` → safety violation). Each `inv_l(point)` is the same conjunction of ground guarded atoms wherever it occurs, so the contradiction is propositional. This is derived by hand from the clause shapes and must be confirmed by running it.

Seeding is bounded unrolling: it cannot help on satisfiable formulas, and it does not exercise automata. Results with and without it are reported separately.

## Correctness

The procedure is sound, and it terminates whenever no projection falls back to a point instance. Definitions used below: `γ` for key `(i, d)` is the restriction of `B*` to the Bool parameters that occur syntactically in disjunct `d` of `Cᵢ`, so `¬Cᵢᵈ(B*, K, X)` and `¬Cᵢᵈ(γ, K, X)` are the same formula.

**Soundness of UNSAT.** Every element of `G` follows from `∀X ψ` for each fixed `P`:

- A seeded instance is `ψ` at a ground `X`.
- A lemma `γ → Kᵢ ∉ Bad_{i,d,γ}`: if `γ` holds and `Kᵢ ∈ Bad`, some `X` violates `Cᵢ`, so `∀X ψ` is false.
- A synthesis blocking clause says that a set of guards and atoms is inconsistent with the lemmas active under it; it stays valid when later lemmas are added, because lemmas only remove solutions.

So if `G` has no model, neither does `∃P ∀X ψ`.

**Soundness of SAT.** SAT is returned only when verification shows every disjunct of every `¬Cᵢ(P*, X)` empty. The emptiness checks are exact (exhausted early-stop products, or a sound bound refutation), so `∀X ψ(P*, X)` holds.

**Progress.** Suppose verification returns `(i, d, X*)` for `P*` and the key `(i, d, γ)` already has a lemma. Then `γ` holds under `B*`, so `K*ᵢ ∉ Bad`. But `X*` violates `Cᵢᵈ(γ, K*ᵢ, X*)`, so `K*ᵢ ∈ Bad`. Contradiction. Every round therefore learns a lemma with a new key.

**Termination.** There are finitely many keys: clauses × disjuncts × assignments to the Bool parameters of one disjunct. Within one synthesis call, each blocking clause is falsified by the current `B*` and the asserted atoms, so SAT cannot return them again, and there are finitely many such assignments. Hence the outer loop and each synthesis call end.

The guarantee is lost for any key where the state budget forced a `point` fallback; such keys are reported in debug output. Termination says nothing about running time: the number of keys can be exponential in the Bool parameters per disjunct.

## Integration into Amaya

The strategy lives in a new module `amaya/cegqi_automata.py` with entry point `solve_with_cegqi_over_automata`, dispatched next to the other strategies in `run-amaya.py`. Everything else is reused or extended.

| Need | Existing code | Change |
| --- | --- | --- |
| Bool substitution and folding | `amaya/sat_toplevel.py:substitute_bool_vars` | none |
| Int constant substitution | `amaya/preprocessing/unbound_vars.py:_substitute_var_with_value`, `amaya/preprocessing/eval.py:apply_substitution` | generalize to a map of constants over `Relation` nodes |
| Disjunct decision, verification | `amaya/parse.py:try_decide_atom_conjunction_with_early_stop` | none |
| Model decoding | `amaya/parse.py:convert_binary_model_into_decadic` | none |
| Projection, determinization, minimization | default evaluation operations (`-m hopcroft`) | call on one disjunct with a state budget |
| Bound filter | `amaya/dpllt_automata.py:find_bounds_refutation` | return a core, not only a verdict |
| Lazy product with complemented operands | `amaya/dpllt_automata.py:_try_decide_iteration_by_early_stopping_lazy_construction` | accept deterministic operands to complement by final-state swap |
| SAT abstraction | `amaya/sat_toplevel.py:abstract_to_sat_formula` | none |
| Shortest-word model | `mtbdd-backend/src/base.cpp:NFA::find_model` | add BFS variant, expose through `MTBDD_NFA` |

Track handling: every automaton is built over only the tracks it mentions. Lemma automata over `Kᵢ` are stored with their track list, and the lazy product aligns overlapping tracks. If the backend currently forces the run's full alphabet on every automaton, that must change for this strategy; check before milestone 3.

CLI (all help text marked `(EXPERIMENTAL)`, all off by default):

| Flag | Meaning |
| --- | --- |
| `--use-cegqi-automata` | enable the strategy |
| `--cegqi-generalization {disjunct,clause,point}` | lemma granularity, default `disjunct` |
| `--cegqi-projection-state-limit N` | budget before falling back to a point instance |
| `--cegqi-seed-depth N` | forward ground seeding depth, default 0 |
| `--cegqi-shortest-model` | BFS model extraction in synthesis |
| `--cegqi-core-minimization-ms N` | time budget for shrinking blocking clauses |

Debug output per round: candidate bit lengths, the violated `(i, d)`, `|γ|`, sizes of the projected NFA and of `Bad`, synthesis product states explored, and fallbacks taken.

## Risks and open questions

The largest risk is the size of `Bad` after projecting `X` with about 24 `Kᵢ` tracks kept; nothing measured so far bounds it.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| `Bad` blows up in determinization | generalization too slow | `disjunct` granularity; state budget with point fallback; measure first (milestone 2) |
| Many lemma keys (exponential in \` | γ | \`) |
| Synthesis product large on UNSAT | final synthesis call slow | bound filter with cores; seeding; measure product states |
| Coarse blocking clauses | many SAT iterations, as in top-level SAT | core minimization under a time budget |
| Unit-coefficient benchmarks need no automata | weak case for automata in the paper | non-automata baseline; benchmarks with non-unit coefficients and congruences |
| Backend forces full alphabet on every automaton | track locality lost | check in milestone 1; per-automaton track lists |

Open questions:

- [ ] Does the MTBDD backend support automata over a subset of the run's tracks, and products of automata with different track sets?
- [ ] Is `Bad` small enough to determinize on the motivating benchmark, per disjunct and per clause?
- [ ] Does seeding alone refute the motivating benchmark, as the hand derivation predicts?
- [ ] Is `find_bounds_refutation` able to return a core cheaply, or does it need rewriting?
- [ ] Which other `From_T2` instances fit the shape, and how many are satisfiable?

## Evaluation plan

The evaluation must show where automata projection is the reason for success, not only that the strategy solves the benchmark.

Configurations:

| Configuration | Purpose |
| --- | --- |
| Default, top-level SAT, DPLL(T) over automata | existing baselines |
| CEGQI, `point` generalization | effect of projection versus point instances |
| CEGQI, `disjunct` and `clause` generalization | granularity trade-off |
| CEGQI with a non-automata oracle (difference logic or z3 for verification and projection-free synthesis) | isolates the automata contribution |
| Each of the above with seeding on and off | separates shallow propositional refutations |
| z3, cvc5 | external reference |

Benchmark sets:

1. `Integer_Transition_Systems/From_T2` instances that fit the shape, split into SAT and UNSAT.
2. The same generator (`gen_its_hard`) with non-unit template or transition coefficients.
3. Instances with `mod` or congruences in transitions.
4. Instances with large constants (long bit lengths).

Sets 2 to 4 are where exact automata projection should matter; set 1 is reported separately because unit coefficients favor non-automata reasoning.

Metrics per instance: result and time; rounds; lemmas learned and fallbacks; maximum and median size of `Bad`; tracks per automaton; synthesis product states explored; SAT iterations; bit length of candidates with DFS versus BFS model search.

## Implementation milestones

The order front-loads the two measurements that can invalidate the design: per-clause projection size and seeding on the motivating benchmark.

1. **Shape detection and substitution.** Detect `∃P ∀X ⋀ Cᵢ`, compute `Pᵢ`, substitute Bool and Int constants. Confirm the backend can build automata over a subset of tracks. Gate: detection works on the benchmark and reports `max |Pᵢ| / |P|`.
2. **Projection probe (standalone).** For each clause and disjunct of the benchmark and a sample of guard assignments, build `Bad` and record NFA and DFA sizes. Gate: `Bad` fits a practical state budget for `disjunct` granularity; if not, revisit the design before continuing.
3. **Seeding plus SAT.** Implement forward ground seeding and run it with SAT alone. Gate: confirms or refutes the hand-derived refutation.
4. **Verification step.** Disjunct splitting, bound pre-check, early-stop decision, model decoding.
5. **Synthesis step.** SAT abstraction, lemma store, bound filter with cores, lazy product with complemented `Bad` automata, blocking clauses.
6. **Loop with `point` generalization.** End-to-end baseline; check soundness on small hand-made `∃∀` instances with known answers.
7. **Projection lemmas.** `disjunct` and `clause` granularity, state budget, fallback, key deduplication.
8. **BFS model search** in the MTBDD backend.
9. **Core minimization** for blocking clauses.
10. **Evaluation** as planned above, including the non-automata baseline.
