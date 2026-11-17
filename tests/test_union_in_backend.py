"""
`MTBDD_NFA.union` runs in C++ (`mtbdd-backend/src/base.cpp:compute_nfa_union`). It numbers states as the
Python implementation kept as `MTBDD_NFA._union_in_python` does when the operand states are iterated in
ascending order, so the tests compare the two results by their states and their transitions expanded to concrete symbols.
"""
import itertools
import logging

import pytest

from amaya import logger as amaya_logger
from amaya.alphabet import LSBF_Alphabet
from amaya.mtbdd_transitions import MTBDDTransitionFn
from amaya.relations_structures import Congruence, Relation, Var


X, Y, Z = Var(id=1), Var(id=2), Var(id=3)
ALPHABET = LSBF_Alphabet.from_vars([X, Y, Z])


@pytest.fixture(autouse=True)
def quiet_logger():
    saved_log_level = amaya_logger.level
    amaya_logger.setLevel(logging.WARNING)
    yield
    amaya_logger.setLevel(saved_log_level)


def _le(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _eq(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='=')


def _nfa_for(atoms, quantified_vars=()):
    return MTBDDTransitionFn.construct_dfa_for_atom_conjunction(atoms, list(quantified_vars), ALPHABET)


OPERAND_PAIRS = {
    'same-variables': lambda: (_nfa_for([_le([(1, X), (1, Y)], 3)]), _nfa_for([_eq([(2, X), (-1, Y)], 1)])),
    'different-variables': lambda: (_nfa_for([_le([(1, X)], -20)]), _nfa_for([_le([(-1, Z)], -30)])),
    'congruence-and-projection': lambda: (_nfa_for([Congruence(vars=[X, Y], coefs=[1, 3], rhs=2, modulus=7)]),
                                          _nfa_for([_eq([(1, X), (-2, Z)], 0)], (Z,))),
    'operand-that-is-a-union': lambda: (_nfa_for([_le([(1, X)], 4)]).union(_nfa_for([_le([(-1, X)], -9)])),
                                        _nfa_for([_eq([(1, Y)], 5)])),
}


def _symbolic_transitions(nfa):
    """
    The transitions as (origin, destination, concrete symbol over all variables of `ALPHABET`).

    The backend union is over the union of the operand variables, the Python union over all variables of
    the alphabet; a variable missing in a symbol is unconstrained, so both expand to the same set.
    """
    alphabet_var_ids = [var.id for var in ALPHABET.all_vars]
    concrete_transitions = set()
    for origin, dest, symbol in nfa.transition_fn._nfa.get_symbolic_transitions():
        track_value_by_var_id = dict(zip(nfa.transition_fn._nfa.vars, symbol))
        track_choices = [(0, 1) if track_value_by_var_id.get(var_id, 2) == 2 else (track_value_by_var_id[var_id],)
                         for var_id in alphabet_var_ids]
        for concrete_symbol in itertools.product(*track_choices):
            concrete_transitions.add((origin, dest, concrete_symbol))
    return sorted(concrete_transitions)


def _renumber_states_in_ascending_order(nfa):
    """ `_union_in_python` numbers states in the iteration order of a Python set; make it ascending. """
    rename_map = {state: index for index, state in enumerate(sorted(nfa.states))}
    nfa.transition_fn.rename_states(rename_map)
    nfa.states = set(rename_map.values())
    nfa.initial_states = {rename_map[state] for state in nfa.initial_states}
    nfa.final_states = {rename_map[state] for state in nfa.final_states}


@pytest.mark.parametrize('operand_pair_id', sorted(OPERAND_PAIRS))
def test_backend_union_equals_python_union(operand_pair_id):
    left, right = OPERAND_PAIRS[operand_pair_id]()
    left_copy, right_copy = OPERAND_PAIRS[operand_pair_id]()
    _renumber_states_in_ascending_order(left_copy)
    _renumber_states_in_ascending_order(right_copy)

    backend_union = left.union(right)
    python_union = left_copy._union_in_python(right_copy)

    assert backend_union.states == python_union.states
    assert backend_union.initial_states == python_union.initial_states
    assert backend_union.final_states == python_union.final_states
    assert _symbolic_transitions(backend_union) == _symbolic_transitions(python_union)
    assert backend_union.used_variables == python_union.used_variables


def test_backend_union_does_not_modify_the_right_operand():
    left, right = OPERAND_PAIRS['same-variables']()
    right_states_before = set(right.states)
    right_transitions_before = _symbolic_transitions(right)

    left.union(right)

    assert right.states == right_states_before
    assert _symbolic_transitions(right) == right_transitions_before


def test_backend_union_accepts_the_models_of_both_operands():
    left, right = OPERAND_PAIRS['different-variables']()
    union = left.union(right)

    model = union.find_model()
    assert model is not None
