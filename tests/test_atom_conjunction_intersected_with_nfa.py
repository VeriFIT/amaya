"""
Tests for the construction of the automaton for a conjunction of atoms intersected with an NFA
(`MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa`,
`mtbdd-backend/src/wrapper.cpp:construct_atom_conjunction_intersected_with_nfa`).

The reference is the existing route: the automaton for the conjunction built on its own
(`construct_dfa_for_atom_conjunction`) and intersected with the NFA. The languages are compared on every
assignment of a small grid of values.
"""
import itertools
import logging

import pytest

from amaya import logger as amaya_logger
from amaya import parse
from amaya.alphabet import LSBF_Alphabet
from amaya.mtbdd_transitions import MTBDDTransitionFn
from amaya.relations_structures import Congruence, Relation, Var


X, Y, Z = Var(id=1), Var(id=2), Var(id=3)
ALPHABET = LSBF_Alphabet.from_vars([X, Y, Z])
VALUE_RANGE = range(-6, 7)


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


def _holds(atom, assignment) -> bool:
    total = sum(coef * assignment[var] for coef, var in zip(atom.coefs, atom.vars))
    if isinstance(atom, Congruence):
        return (total - atom.rhs) % atom.modulus == 0
    return total == atom.rhs if atom.predicate_symbol == '=' else total <= atom.rhs


def _nfa_for(atoms, quantified_vars=()):
    return MTBDDTransitionFn.construct_dfa_for_atom_conjunction(atoms, list(quantified_vars), ALPHABET)


def _accepts(nfa, assignment) -> bool:
    restricted = nfa
    for var, value in assignment.items():
        point = MTBDDTransitionFn.construct_nfa_for_eq(_eq([(1, var)], value), ALPHABET)
        restricted = restricted.intersection(point)
    return restricted.find_model() is not None


def _assert_languages_agree_on_grid(actual_nfa, expected_nfa, grid_vars):
    for values in itertools.product(VALUE_RANGE, repeat=len(grid_vars)):
        assignment = dict(zip(grid_vars, values))
        assert _accepts(actual_nfa, assignment) == _accepts(expected_nfa, assignment), assignment


# (conjunction, atoms the NFA is built from, the NFA's quantified variables, the variables of the grid)
CASES = [
    pytest.param([_le([(1, X), (1, Y)], 3), _le([(-1, Y)], 1)],
                 [_le([(-1, X)], 0)], (), [X, Y], id='inequations'),
    pytest.param([_eq([(2, X), (-1, Y)], 1), _le([(1, Y)], 5)],
                 [_le([(1, X)], 4), _le([(-1, X)], 2)], (), [X, Y], id='equation'),
    pytest.param([Congruence(vars=[X, Y], coefs=[1, 3], rhs=2, modulus=4), _le([(1, X)], 5)],
                 [_le([(-1, Y)], 2)], (), [X, Y], id='congruence-even-modulus'),
    pytest.param([Congruence(vars=[X], coefs=[2], rhs=1, modulus=3), Congruence(vars=[Y], coefs=[1], rhs=0, modulus=12)],
                 [_le([(1, X), (-1, Y)], 0)], (), [X, Y], id='two-congruences'),
    pytest.param([_le([(1, X)], 4), _le([(-1, X)], 4)],
                 [_eq([(1, X), (-2, Z)], 0)], (Z,), [X], id='nfa-with-projected-variable'),
    pytest.param([_le([(1, X)], 2)],
                 [_le([(1, Y)], 0)], (), [X, Y], id='disjoint-variables'),
]


@pytest.mark.parametrize('conjunction, nfa_atoms, nfa_quantified_vars, grid_vars', CASES)
def test_language_equals_that_of_the_separately_built_intersection(conjunction, nfa_atoms, nfa_quantified_vars,
                                                                   grid_vars):
    nfa = _nfa_for(nfa_atoms, nfa_quantified_vars)

    actual = MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa(conjunction, nfa)
    expected = _nfa_for(conjunction).intersection(nfa)

    _assert_languages_agree_on_grid(actual, expected, grid_vars)


def test_nondeterministic_nfa_with_two_initial_states():
    # The union of two automata keeps both initial states: (x <= -2) or (x >= 3).
    nfa = _nfa_for([_le([(1, X)], -2)]).union(_nfa_for([_le([(-1, X)], -3)]))
    conjunction = [Congruence(vars=[X], coefs=[1], rhs=0, modulus=2), _le([(1, X), (1, Y)], 1)]

    actual = MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa(conjunction, nfa)
    expected = _nfa_for(conjunction).intersection(nfa)

    _assert_languages_agree_on_grid(actual, expected, [X, Y])


@pytest.mark.parametrize('conjunction, nfa_atoms, nfa_quantified_vars, grid_vars', CASES)
def test_model_of_the_early_stopped_construction_satisfies_both(conjunction, nfa_atoms, nfa_quantified_vars,
                                                                grid_vars):
    nfa = _nfa_for(nfa_atoms, nfa_quantified_vars)

    early_stopped = MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa(
        conjunction, nfa, stop_after_first_accepting_state=True)
    full = MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa(conjunction, nfa)

    binary_model = early_stopped.find_model()
    assert (binary_model is None) == (full.find_model() is None)
    assert len(early_stopped.states) <= len(full.states)

    if binary_model is not None:
        model = parse.convert_binary_model_into_decadic(binary_model, tuple(early_stopped.used_variables), (X, Y, Z))
        assert all(_holds(atom, model) for atom in conjunction), model
        point_constraints = [_eq([(1, var)], model[var]) for var in (X, Y) if var in early_stopped.used_variables]
        assert _nfa_for(point_constraints).intersection(nfa).find_model() is not None, model


def test_unsatisfiable_combination_has_an_empty_language():
    nfa = _nfa_for([_le([(-1, X)], -5)])  # x >= 5
    conjunction = [_le([(1, X)], 4)]

    for stop_after_first_accepting_state in (False, True):
        result = MTBDDTransitionFn.construct_atom_conjunction_intersected_with_nfa(
            conjunction, nfa, stop_after_first_accepting_state=stop_after_first_accepting_state)
        assert result.find_model() is None


# --- The DPLL(T) iteration (`--dpllt-construct-assertion-with-phi`) ------------------------------

@pytest.fixture
def mtbdd_dpllt_config():
    import copy
    from amaya.config import BackendType, DpllTAutomataConfig, SolutionDomain, solver_config

    saved_backend = solver_config.backend_type
    saved_domain = solver_config.solution_domain
    saved_dpllt = copy.deepcopy(solver_config.dpllt_automata)

    solver_config.backend_type = BackendType.MTBDD
    solver_config.solution_domain = SolutionDomain.INTEGERS
    solver_config.dpllt_automata = DpllTAutomataConfig()

    yield solver_config

    solver_config.backend_type = saved_backend
    solver_config.solution_domain = saved_domain
    solver_config.dpllt_automata = saved_dpllt


def test_iteration_is_decided_against_a_general_part_that_is_not_a_conjunction_of_atoms(mtbdd_dpllt_config):
    from amaya import dsl
    from amaya.dpllt_automata import (
        Dpllt_Run_Statistics,
        General_Part,
        _try_decide_iteration_by_constructing_assertion_intersected_with_phi,
    )
    from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
    from amaya.preprocessing.eval import VarInfo
    from amaya.relations_structures import VariableType
    from amaya.solver_core import EvaluationContext

    var_table = {
        X: VarInfo(name='x', type=VariableType.INT, is_formula_param=True),
        Z: VarInfo(name='z', type=VariableType.INT, is_formula_param=False),
    }
    ctx = EvaluationContext(alphabet=LSBF_Alphabet.from_vars(var_table.keys()), var_table=var_table)

    x_is_odd = dsl._neg(dsl._exists((Z,), _eq([(1, X), (-2, Z)], 0)))
    fill_referenced_vars(x_is_odd)
    statistics = Dpllt_Run_Statistics()
    general_part = General_Part((x_is_odd,), ctx, statistics)
    assert general_part.atoms is None

    satisfiable_assertion = dsl._and(_le([(1, X)], 7), _le([(-1, X)], -6))     # 6 <= x <= 7, x odd -> x = 7
    unsatisfiable_assertion = dsl._and(_le([(1, X)], 6), _le([(-1, X)], -6))   # x = 6, x odd

    was_decided, result = _try_decide_iteration_by_constructing_assertion_intersected_with_phi(
        general_part, satisfiable_assertion, ctx, statistics)
    assert was_decided
    assert result is not None and result.model[X] == 7

    was_decided, result = _try_decide_iteration_by_constructing_assertion_intersected_with_phi(
        general_part, unsatisfiable_assertion, ctx, statistics)
    assert was_decided and result is None

    assert statistics.iterations_decided_by_constructing_assertion_intersected_with_phi == 2


DPLLT_SOURCES = [
    # The general part contains a negated quantifier, so it is not a conjunction of atoms.
    ("""
(declare-fun x () Int)
(declare-fun y () Int)
(assert (<= 0 x))
(assert (<= x 10))
(assert (not (exists ((z Int)) (= x (* 2 z)))))
(assert (or (exists ((w Int)) (and (= x (* 3 w)) (<= 2 w)))
            (exists ((v Int)) (and (= y (+ x v)) (<= v 2) (<= 2 v) (<= y 0)))))
(check-sat)
""", True),
    ("""
(declare-fun x () Int)
(assert (<= 0 x))
(assert (<= x 4))
(assert (not (exists ((z Int)) (= x (* 2 z)))))
(assert (or (exists ((w Int)) (= x (* 4 w)))
            (exists ((v Int)) (and (= x (+ v 6)) (<= 0 v)))))
(check-sat)
""", False),
]


@pytest.mark.parametrize('source_text, is_satisfiable', DPLLT_SOURCES)
def test_dpllt_verdict_does_not_depend_on_the_flag(source_text, is_satisfiable, mtbdd_dpllt_config):
    from amaya.dpllt_automata import perform_whole_evaluation_on_source_text_with_dpllt_automata

    verdicts = []
    for flag_value in (False, True):
        mtbdd_dpllt_config.dpllt_automata.construct_assertion_intersected_with_phi = flag_value
        result = perform_whole_evaluation_on_source_text_with_dpllt_automata(source_text)
        verdicts.append(result is not None and result.model is not None)

    assert verdicts == [is_satisfiable, is_satisfiable]
