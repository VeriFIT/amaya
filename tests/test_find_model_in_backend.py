"""
`MTBDD_NFA.find_model` runs in C++ (`mtbdd-backend/src/base.cpp:NFA::find_model`). The exploration order
is that of the Python implementation kept as `MTBDD_NFA._find_model_in_python`, so both return the same
word; the tests compare them on automata with one and several initial states, with an empty language,
and in both solution domains.
"""
import logging

import pytest

from amaya import logger as amaya_logger
from amaya.alphabet import LSBF_Alphabet
from amaya.config import SolutionDomain, solver_config
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


@pytest.fixture(params=[SolutionDomain.INTEGERS, SolutionDomain.NATURALS], ids=['integers', 'naturals'])
def solution_domain(request):
    saved_domain = solver_config.solution_domain
    solver_config.solution_domain = request.param
    yield request.param
    solver_config.solution_domain = saved_domain


def _le(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _eq(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='=')


def _nfa_for(atoms, quantified_vars=()):
    return MTBDDTransitionFn.construct_dfa_for_atom_conjunction(atoms, list(quantified_vars), ALPHABET)


def _automata():
    yield 'inequations', _nfa_for([_le([(1, X), (1, Y)], 3), _le([(-1, Y)], -5)])
    yield 'equation', _nfa_for([_eq([(2, X), (-1, Y)], 1), _le([(1, Y)], 5)])
    yield 'congruence', _nfa_for([Congruence(vars=[X, Y], coefs=[1, 3], rhs=2, modulus=7), _le([(-1, X)], -40)])
    yield 'projected-variable', _nfa_for([_eq([(1, X), (-2, Z)], 0), _le([(-1, X)], -9)], (Z,))
    yield 'two-initial-states', _nfa_for([_le([(1, X)], -20)]).union(_nfa_for([_le([(-1, X)], -30)]))
    yield 'intersection', _nfa_for([_le([(1, X), (-1, Y)], 0)]).intersection(_nfa_for([_eq([(1, X), (1, Y)], 17)]))
    yield 'empty-language', _nfa_for([_le([(1, X)], 4)]).intersection(_nfa_for([_le([(-1, X)], -5)]))


@pytest.mark.parametrize('automaton_id', [automaton_id for automaton_id, _ in _automata()])
def test_backend_search_returns_the_word_of_the_python_search(automaton_id, solution_domain):
    nfa = dict(_automata())[automaton_id]

    assert nfa.find_model() == nfa._find_model_in_python()


def test_empty_language_has_no_model():
    nfa = dict(_automata())['empty-language']

    assert nfa.find_model() is None
