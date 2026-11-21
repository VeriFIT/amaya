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
