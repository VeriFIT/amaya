"""
Tests for `MTBDDTransitionFn.reduce_nfa_by_forward_simulation` (mtbdd-backend/src/simulation_reduction.cpp):
the reduced automaton accepts the same language, and simulation-equivalent and strictly simulated states
are removed.
"""
import copy
import itertools
import random

import pytest

from amaya.alphabet import LSBF_Alphabet
from amaya.automatons import AutomatonType
from amaya.config import solver_config
from amaya.mtbdd_automatons import MTBDD_NFA
from amaya.mtbdd_transitions import MTBDDTransitionFn
from amaya.relations_structures import Var
from amaya.semantics_tracking import AH_Atom, AH_AtomType


TRACKS = [Var(1), Var(2)]
SYMBOLS = list(itertools.product((0, 1), repeat=len(TRACKS)))


@pytest.fixture
def simulation_reduction_disabled():
    saved_backend_config = copy.deepcopy(solver_config.backend)
    solver_config.backend.reduce_by_simulation_before_determinization = False
    yield
    solver_config.backend = saved_backend_config


def make_nfa(states, initial_states, final_states, transitions) -> MTBDD_NFA:
    nfa = MTBDD_NFA(alphabet=LSBF_Alphabet.from_vars(TRACKS), automaton_type=AutomatonType.NFA,
                    state_semantics=AH_Atom(atom_type=AH_AtomType.CUSTOM, atom=None),
                    states=set(states), initial_states=set(initial_states), final_states=set(final_states),
                    used_variables=list(TRACKS))
    for source, symbol, destination in transitions:
        nfa.update_transition_fn(source, symbol, destination)
    return nfa


def is_language_included(smaller: MTBDD_NFA, larger: MTBDD_NFA) -> bool:
    larger_complement = larger.determinize().complement()
    return smaller.intersection(larger_complement).find_model() is None


def assert_same_language(left: MTBDD_NFA, right: MTBDD_NFA):
    assert is_language_included(left, right)
    assert is_language_included(right, left)


def count_transitions(nfa: MTBDD_NFA) -> int:
    return sum(len(nfa.transition_fn.get_state_post(state)) for state in nfa.states) if nfa.states else 0


def test_simulation_equivalent_states_are_merged(simulation_reduction_disabled):
    # 1 and 2 accept the same language with the same transitions: they are simulation-equivalent.
    nfa = make_nfa(states={0, 1, 2, 3}, initial_states={0}, final_states={3},
                   transitions=[(0, (0, 0), 1), (0, (0, 0), 2), (1, (1, 1), 3), (2, (1, 1), 3), (3, (0, 0), 3)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0)

    assert len(reduced.states) == 3
    assert_same_language(nfa, reduced)


def test_strictly_simulated_destination_is_pruned(simulation_reduction_disabled):
    # 2 accepts (1,1) only, 1 accepts (1,1) and (0,1): 2 < 1, so 0 -(0,0)-> 2 is redundant next to 0 -(0,0)-> 1.
    nfa = make_nfa(states={0, 1, 2, 3}, initial_states={0}, final_states={3},
                   transitions=[(0, (0, 0), 1), (0, (0, 0), 2), (1, (1, 1), 3), (1, (0, 1), 3), (2, (1, 1), 3)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0)

    assert 2 not in reduced.states
    assert_same_language(nfa, reduced)


def test_automaton_above_state_limit_only_loses_useless_states(simulation_reduction_disabled):
    # 4 reaches no final state; with max_states=1 the simulation is skipped.
    nfa = make_nfa(states={0, 1, 2, 3, 4}, initial_states={0}, final_states={3},
                   transitions=[(0, (0, 0), 1), (0, (0, 0), 2), (1, (1, 1), 3), (2, (1, 1), 3), (0, (1, 0), 4)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=1)

    assert reduced.states == {0, 1, 2, 3}
    assert_same_language(nfa, reduced)


def test_reduction_preserves_language_of_random_nfas(simulation_reduction_disabled):
    rng = random.Random(20261121)
    reduced_state_counts = []
    for _ in range(200):
        state_count = rng.randint(1, 8)
        states = list(range(state_count))
        transitions = []
        for source in states:
            for symbol in SYMBOLS:
                for destination in rng.sample(states, rng.randint(0, min(3, state_count))):
                    transitions.append((source, symbol, destination))
        initial_states = rng.sample(states, rng.randint(1, min(2, state_count)))
        final_states = rng.sample(states, rng.randint(0, state_count))
        nfa = make_nfa(states, initial_states, final_states, transitions)

        reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0)

        assert_same_language(nfa, reduced)
        reduced_state_counts.append((state_count, len(reduced.states)))
    assert any(reduced_count < state_count for state_count, reduced_count in reduced_state_counts)


def test_determinization_with_reduction_accepts_the_same_language(simulation_reduction_disabled):
    rng = random.Random(7)
    for _ in range(50):
        states = list(range(6))
        transitions = [(source, symbol, destination) for source in states for symbol in SYMBOLS
                       for destination in rng.sample(states, rng.randint(0, 2))]
        nfa = make_nfa(states, [0], rng.sample(states, 2), transitions)

        dfa_without_reduction = nfa.determinize()
        solver_config.backend.reduce_by_simulation_before_determinization = True
        dfa_with_reduction = nfa.determinize()
        solver_config.backend.reduce_by_simulation_before_determinization = False

        assert dfa_with_reduction.automaton_type & AutomatonType.DFA
        assert_same_language(dfa_without_reduction, dfa_with_reduction)
        assert_same_language(dfa_without_reduction.complement(), dfa_with_reduction.complement())


def _minimal_dfa_state_count(nfa: MTBDD_NFA) -> int:
    return len(MTBDDTransitionFn.minimize_hopcroft(MTBDDTransitionFn.determinize(nfa)).states)


def test_macrostate_member_simulated_by_another_member_is_dropped(simulation_reduction_disabled):
    # 0 and 1 are incomparable initial states. 2 accepts (1,1) only and 3 accepts (1,1) and (1,0), so 2 < 3;
    # the macrostate {2, 3} reached on (0,0) is pruned to {3}, which (1,0) reaches on its own.
    nfa = make_nfa(states={0, 1, 2, 3, 4}, initial_states={0, 1}, final_states={4},
                   transitions=[(0, (0, 0), 2), (0, (1, 1), 4), (1, (0, 0), 3), (1, (0, 1), 4), (1, (1, 0), 3),
                                (2, (1, 1), 4), (3, (1, 1), 4), (3, (1, 0), 4)])

    dfa_without_pruning = MTBDDTransitionFn.determinize(MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0))
    dfa_with_pruning = MTBDDTransitionFn.determinize_pruned_by_simulation(nfa, max_states=0)

    assert len(dfa_with_pruning.states) < len(dfa_without_pruning.states)
    assert_same_language(nfa, dfa_with_pruning)


def test_pruned_determinization_of_random_nfas_has_the_same_minimal_dfa(simulation_reduction_disabled):
    rng = random.Random(20261122)
    smaller_dfa_count = 0
    for _ in range(200):
        state_count = rng.randint(1, 8)
        states = list(range(state_count))
        transitions = [(source, symbol, destination) for source in states for symbol in SYMBOLS
                       for destination in rng.sample(states, rng.randint(0, min(3, state_count)))]
        initial_states = rng.sample(states, rng.randint(1, min(3, state_count)))
        nfa = make_nfa(states, initial_states, rng.sample(states, rng.randint(0, state_count)), transitions)

        dfa_without_pruning = MTBDDTransitionFn.determinize(nfa)
        dfa_with_pruning = MTBDDTransitionFn.determinize_pruned_by_simulation(nfa, max_states=0)

        assert dfa_with_pruning.automaton_type & AutomatonType.DFA
        assert_same_language(dfa_without_pruning, dfa_with_pruning)
        assert_same_language(dfa_without_pruning.complement(), dfa_with_pruning.complement())
        assert _minimal_dfa_state_count(nfa) == len(MTBDDTransitionFn.minimize_hopcroft(dfa_with_pruning).states)
        smaller_dfa_count += len(dfa_with_pruning.states) < len(dfa_without_pruning.states)
    assert smaller_dfa_count > 0


def test_nfa_without_initial_state_reaching_final_state_keeps_an_initial_state(simulation_reduction_disabled):
    # 2 is final but unreachable from the initial state 0, so every state reachable from 0 is useless.
    nfa = make_nfa(states={0, 1, 2}, initial_states={0}, final_states={2},
                   transitions=[(0, (0, 0), 1), (0, (0, 0), 0), (1, (1, 1), 1), (2, (0, 0), 2)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0)

    assert reduced.initial_states and not reduced.final_states
    assert reduced.determinize().minimize_hopcroft().find_model() is None


def test_deterministic_automaton_is_returned_unchanged(simulation_reduction_disabled):
    # 1 and 2 are simulation-equivalent, but the automaton is deterministic and stays as it is.
    nfa = make_nfa(states={0, 1, 2, 3}, initial_states={0}, final_states={3},
                   transitions=[(0, (0, 0), 1), (0, (1, 1), 2), (1, (1, 1), 3), (2, (1, 1), 3)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0)

    assert reduced.states == {0, 1, 2, 3}
    assert_same_language(nfa, reduced)


def test_deterministic_automaton_is_minimized_when_requested(simulation_reduction_disabled):
    # 1, 2 and 4 are equivalent states of a deterministic automaton; the minimal complete DFA has an
    # initial state, the merged state, the final state and a trap state.
    nfa = make_nfa(states={0, 1, 2, 3, 4}, initial_states={0}, final_states={3},
                   transitions=[(0, (0, 0), 1), (0, (1, 1), 2), (0, (0, 1), 4),
                                (1, (1, 1), 3), (2, (1, 1), 3), (4, (1, 1), 3)])

    reduced = MTBDDTransitionFn.reduce_nfa_by_forward_simulation(nfa, max_states=0, minimize_if_deterministic=True)

    assert len(reduced.states) == 4
    assert_same_language(nfa, reduced)


QUANTIFIED_FORMULAS = [
    # (assertion over the free variables a, b, expected satisfiability)
    ('(exists ((x Int) (y Int)) (and (<= (+ x y) a) (= (- x y) 3) (>= x 2) (<= y b)))', True),
    ('(exists ((x Int)) (and (<= a (* 2 x)) (<= (* 2 x) (+ a 1)) (= (mod a 2) 1) (<= x b) (<= b (- x 1))))', False),
    ('(forall ((x Int)) (=> (and (<= 0 x) (<= x a)) (<= (* 3 x) (+ b 7))))', True),
    ('(and (> a 5) (forall ((x Int)) (exists ((y Int)) (and (<= x y) (<= y (+ x a)) (= (mod y 3) b)))))', True),
    ('(and (= b 0) (exists ((x Int) (y Int)) (and (= (+ (* 2 x) (* 4 y)) (+ a 1)) (= (mod a 2) 0))))', False),
]


@pytest.mark.parametrize('assertion, expected_is_sat', QUANTIFIED_FORMULAS)
def test_reduction_before_projection_keeps_verdicts(simulation_reduction_disabled, assertion, expected_is_sat):
    from amaya import parse
    from amaya.config import BackendType, SolutionDomain
    source = ('(set-logic LIA)\n(declare-fun a () Int)\n(declare-fun b () Int)\n'
              f'(assert {assertion})\n(check-sat)\n')
    saved_backend_type, saved_domain = solver_config.backend_type, solver_config.solution_domain
    solver_config.backend_type, solver_config.solution_domain = BackendType.MTBDD, SolutionDomain.INTEGERS
    try:
        verdicts = []
        for is_reduction_enabled in (False, True):
            solver_config.backend.reduce_by_simulation_before_projection = is_reduction_enabled
            solver_config.backend.reduce_by_simulation_before_determinization = is_reduction_enabled
            result = parse.perform_whole_evaluation_on_source_text(source)
            verdicts.append(result.model is not None)
    finally:
        solver_config.backend_type, solver_config.solution_domain = saved_backend_type, saved_domain
    assert verdicts == [expected_is_sat, expected_is_sat]
