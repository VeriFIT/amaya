from typing import (
    Dict,
    List,
    Tuple
)

from amaya.alphabet import LSBF_Alphabet
from amaya.automatons import NFA
from amaya.mtbdd_automatons import MTBDD_NFA
from amaya.presburger.constructions.integers import build_nfa_from_linear_inequality
from amaya.relations_structures import Relation, Var
from tests.conftest import ResolutionState

import pytest

alphabet = LSBF_Alphabet.from_vars([Var(1), Var(2)])


@pytest.fixture()
def relation1() -> Relation:
    """Returns relation: x - y <= 0."""
    relation = Relation.new_lin_relation(variable_names=[Var(1), Var(2)], variable_coefficients=[1, -1],
                                         absolute_part=0, predicate_symbol="<=")
    return relation


@pytest.fixture()
def relation2() -> Relation:
    relation = Relation.new_lin_relation(variable_names=[Var(1), Var(2)], variable_coefficients=[1, 1],
                                         absolute_part=1, predicate_symbol="<=")
    return relation


@pytest.mark.parametrize('automaton_cls', (NFA, MTBDD_NFA))
def test_intersection(automaton_cls: NFA, relation1: Relation, relation2: Relation):
    nfa1 = build_nfa_from_linear_inequality(automaton_cls, alphabet, relation1)
    nfa2 = build_nfa_from_linear_inequality(automaton_cls, alphabet, relation2)

    nfa = nfa1.intersection(nfa2)
    nfa.remove_nonfinishing_states()

    assert len(nfa.states) == 8  # We assume that nonfinishing states are removed after the automaton is constructed
    assert len(nfa.initial_states) == 1
    assert len(nfa.final_states) == 1

    state_names = [(0, 1), (-1, -1), (0, 0), (-1, 0), (-1, -2), (0, -1), (0, -2)]
    states: Dict[Tuple[int, int], ResolutionState] = dict((state, ResolutionState()) for state in state_names)

    initial_state = next(iter(nfa.initial_states))
    states[(0, 1)].bind(initial_state)

    expected_transitions = [
        ((0, 1), (0, 0), (0, 0)),
        ((0, 1), (0, 1), (0, 0)),
        ((0, 1), (1, 0), (-1, 0)),
        ((0, 1), (1, 1), (0, -1)),

        ((0, 0), (0, 0), (0, 0)),
        ((0, 0), (0, 1), (0, -1)),
        ((0, 0), (1, 0), (-1, -1)),
        ((0, 0), (1, 1), (0, -1)),

        ((-1, 0), (0, 0), (-1, 0)),
        ((-1, 0), (0, 1), (0, -1)),
        ((-1, 0), (1, 0), (-1, -1)),
        ((-1, 0), (1, 1), (-1, -1)),

        ((-1, -1), (0, 0), (-1, -1)),
        ((-1, -1), (0, 1), (0, -1)),
        ((-1, -1), (1, 0), (-1, -1)),
        ((-1, -1), (1, 1), (-1, -2)),

        ((-1, -2), (0, 0), (-1, -1)),
        ((-1, -2), (0, 1), (0, -2)),
        ((-1, -2), (1, 0), (-1, -2)),
        ((-1, -2), (1, 1), (-1, -2)),

        ((0, -1), (0, 0), (0, -1)),
        ((0, -1), (0, 1), (0, -1)),
        ((0, -1), (1, 0), (-1, -1)),
        ((0, -1), (1, 1), (0, -2)),

        ((0, -2), (0, 0), (0, -1)),
        ((0, -2), (0, 1), (0, -2)),
        ((0, -2), (1, 0), (-1, -2)),
        ((0, -2), (1, 1), (0, -2)),
    ]

    for origin, symbol, destination in expected_transitions:
        _origin = states[origin].get()
        dest_states = set(nfa.get_transition_target(_origin, symbol))
        dest_nonfinal_states = dest_states - nfa.final_states

        assert len(dest_nonfinal_states) == 1
        _destination = states[destination]
        _destination.bind(next(iter(dest_nonfinal_states)))

    all_symbols = {(0, 0), (0, 1), (1, 0), (1, 1)}
    expected_accepting_symbols = [
        ((0, 1), [(0, 0), (1, 0), (1, 1)]),
        ((-1, 0), [(1, 0)]),
        ((0, 0), [(0, 0), (1, 0), (1, 1)]),
        ((0, 0), [(0, 0), (1, 0), (1, 1)]),
        ((-1, -1), [(1, 0)]),
        ((-1, -2), []),
        ((0, -1), [(1, 0), (1, 1)]),
        ((0, -2), [(1, 1)]),
    ]

    for state, accepting_symbols in expected_accepting_symbols:
        _state = states[state].get()
        for accepting_symbol in accepting_symbols:
            dest_states = set(nfa.get_transition_target(_state, accepting_symbol))
            dest_final_states = nfa.final_states.intersection(dest_states)
            assert len(dest_final_states) == 1

        for nonaccepting_symbol in all_symbols.difference(set(accepting_symbols)):
            dest_states = set(nfa.get_transition_target(_state, nonaccepting_symbol))
            dest_final_states = nfa.final_states.intersection(dest_states)
            assert len(dest_final_states) == 0


def test_intersection_of_complete_dfas_is_a_complete_dfa():
    import random
    from amaya.automatons import AutomatonType
    from amaya.mtbdd_transitions import MTBDDTransitionFn
    from tests.test_simulation_reduction import SYMBOLS, assert_same_language, make_nfa

    def make_random_complete_dfa(rng: random.Random) -> MTBDD_NFA:
        states = list(range(rng.randint(1, 6)))
        transitions = [(source, symbol, destination) for source in states for symbol in SYMBOLS
                       for destination in rng.sample(states, rng.randint(0, min(2, len(states))))]
        nfa = make_nfa(states, [0], rng.sample(states, rng.randint(0, len(states))), transitions)
        return MTBDDTransitionFn.determinize(nfa)

    rng = random.Random(20261122)
    for _ in range(100):
        left, right = make_random_complete_dfa(rng), make_random_complete_dfa(rng)

        product = left.intersection(right)

        assert product.automaton_type & AutomatonType.DFA
        determinized_product = MTBDDTransitionFn.determinize(product)
        assert_same_language(product.complement(), determinized_product.complement())
        assert (len(MTBDDTransitionFn.minimize_hopcroft(product).states)
                == len(MTBDDTransitionFn.minimize_hopcroft(determinized_product).states))
