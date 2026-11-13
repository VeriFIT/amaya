"""
Tests for the lazy construction stopped at the first accepting macrostate
(`MTBDDTransitionFn.construct_dfa_for_atom_conjunction(..., stop_after_first_accepting_macrostate=True)`,
`mtbdd-backend/src/lazy.cpp:build_nfa_with_formula_entailement`).
"""
import copy
import logging

import pytest

from amaya import logger as amaya_logger
from amaya import parse
from amaya.alphabet import LSBF_Alphabet
from amaya.config import BackendType, SolutionDomain, solver_config
from amaya.mtbdd_transitions import MTBDDTransitionFn
from amaya.relations_structures import Congruence, Relation, Var


X, Y, Z = Var(id=1), Var(id=2), Var(id=3)


@pytest.fixture
def mtbdd_solver_config():
    saved_backend = solver_config.backend_type
    saved_domain = solver_config.solution_domain
    saved_dpllt = copy.deepcopy(solver_config.dpllt_automata)
    saved_log_level = amaya_logger.level

    solver_config.backend_type = BackendType.MTBDD
    solver_config.solution_domain = SolutionDomain.INTEGERS
    amaya_logger.setLevel(logging.WARNING)

    yield solver_config

    solver_config.backend_type = saved_backend
    solver_config.solution_domain = saved_domain
    solver_config.dpllt_automata = saved_dpllt
    amaya_logger.setLevel(saved_log_level)


def _le(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _eq(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='=')


def _holds(atom, assignment) -> bool:
    total = sum(coef * assignment[var] for coef, var in zip(atom.coefs, atom.vars))
    if isinstance(atom, Congruence):
        return (total - atom.rhs) % atom.modulus == 0
    return total == atom.rhs if atom.predicate_symbol == '=' else total <= atom.rhs


SATISFIABLE_QUANTIFIER_FREE_CONJUNCTIONS = [
    [_le([(1, X)], 5), _le([(-1, X)], -3)],
    [_le([(1, X), (1, Y)], 4), _le([(-1, X)], -2), Congruence(vars=[Y], coefs=[1], rhs=1, modulus=3)],
    [_eq([(2, X), (-3, Y)], 1), _le([(1, X)], 20), _le([(-1, Y)], -1)],
    [_le([(-1, X)], -1000), Congruence(vars=[X, Y], coefs=[1, 2], rhs=3, modulus=7)],
]

UNSATISFIABLE_CONJUNCTIONS = [
    [_le([(1, X)], 2), _le([(-1, X)], -3)],
    [_eq([(2, X)], 1)],
    [_le([(1, X)], 4), _le([(-1, X)], -4), Congruence(vars=[X], coefs=[1], rhs=1, modulus=2)],
]


def _construct(atoms, quantified_vars, stop_after_first_accepting_macrostate):
    alphabet = LSBF_Alphabet.from_vars([X, Y, Z])
    return MTBDDTransitionFn.construct_dfa_for_atom_conjunction(
        atoms, quantified_vars, alphabet, stop_after_first_accepting_macrostate=stop_after_first_accepting_macrostate)


@pytest.mark.parametrize('atoms', SATISFIABLE_QUANTIFIER_FREE_CONJUNCTIONS)
def test_model_read_off_the_early_stopped_automaton_satisfies_the_conjunction(atoms, mtbdd_solver_config):
    nfa = _construct(atoms, [], stop_after_first_accepting_macrostate=True)

    binary_model = nfa.find_model()

    assert binary_model is not None
    model = parse.convert_binary_model_into_decadic(binary_model, tuple(nfa.used_variables))
    assert all(_holds(atom, model) for atom in atoms), model


@pytest.mark.parametrize('atoms', SATISFIABLE_QUANTIFIER_FREE_CONJUNCTIONS)
def test_early_stopped_construction_explores_no_more_states_than_the_full_one(atoms, mtbdd_solver_config):
    early_stopped_nfa = _construct(atoms, [], stop_after_first_accepting_macrostate=True)
    full_nfa = _construct(atoms, [], stop_after_first_accepting_macrostate=False)

    assert len(early_stopped_nfa.states) <= len(full_nfa.states)


@pytest.mark.parametrize('atoms', UNSATISFIABLE_CONJUNCTIONS)
def test_early_stopped_construction_of_an_unsatisfiable_conjunction_has_no_model(atoms, mtbdd_solver_config):
    nfa = _construct(atoms, [], stop_after_first_accepting_macrostate=True)

    assert nfa.find_model() is None


def test_model_of_free_variables_with_a_quantified_variable(mtbdd_solver_config):
    # exists y. (x = 3y + 1  and  0 <= y <= 5  and  x >= 7)  - x in {7, 10, 13, 16}
    atoms = [_eq([(1, X), (-3, Y)], 1), _le([(-1, Y)], 0), _le([(1, Y)], 5), _le([(-1, X)], -7)]

    nfa = _construct(atoms, [Y], stop_after_first_accepting_macrostate=True)
    binary_model = nfa.find_model()

    assert binary_model is not None
    x_value = parse.convert_binary_model_into_decadic(binary_model, tuple(nfa.used_variables), (X,))[X]
    assert x_value in {7, 10, 13, 16}
