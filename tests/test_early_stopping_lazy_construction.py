"""
Tests for the lazy construction stopped at the first accepting macrostate
(`MTBDDTransitionFn.construct_dfa_for_atom_conjunction(..., stop_after_first_accepting_macrostate=True)`,
`mtbdd-backend/src/lazy.cpp:build_nfa_with_formula_entailement`) and for the solver paths that use it
under `SolverConfig.decide_atom_conjunctions_by_early_stopping_lazy_construction`.
"""
import copy
import logging

import pytest

from amaya import logger as amaya_logger
from amaya import parse
from amaya.alphabet import LSBF_Alphabet
from amaya.config import BackendType, SolutionDomain, solver_config
from amaya.dpllt_automata import perform_whole_evaluation_on_source_text_with_dpllt_automata
from amaya.mtbdd_transitions import MTBDDTransitionFn
from amaya.relations_structures import Congruence, Relation, Var


X, Y, Z = Var(id=1), Var(id=2), Var(id=3)


@pytest.fixture
def mtbdd_solver_config():
    saved_backend = solver_config.backend_type
    saved_domain = solver_config.solution_domain
    saved_dpllt = copy.deepcopy(solver_config.dpllt_automata)
    saved_flag = solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction
    saved_log_level = amaya_logger.level

    solver_config.backend_type = BackendType.MTBDD
    solver_config.solution_domain = SolutionDomain.INTEGERS
    amaya_logger.setLevel(logging.WARNING)

    yield solver_config

    solver_config.backend_type = saved_backend
    solver_config.solution_domain = saved_domain
    solver_config.dpllt_automata = saved_dpllt
    solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction = saved_flag
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


@pytest.mark.parametrize('last_atom_rhs, is_satisfiable', [(-1, False), (0, True)])
@pytest.mark.parametrize('stop_after_first_accepting_macrostate', [True, False])
def test_equation_left_with_one_variable_by_substitution(last_atom_rhs, is_satisfiable,
                                                         stop_after_first_accepting_macrostate, mtbdd_solver_config):
    # a, b, c are fixed to 0 by their bounds; substituting them leaves d = a, e = b, f = c with one
    # variable each. The simplification used to record such an equation as a hard bound of d, e, f and
    # then read it as an inequation (`mtbdd-backend/src/rewrites.cpp:set_var_value_in_eq`): SIGSEGV/SIGFPE.
    a, b, c, d, e, f = (Var(id=10 + index) for index in range(6))
    atoms = [_le([(-1, a)], 0), _le([(1, a)], 0), _le([(-1, b)], 0), _le([(1, b)], 0), _le([(-1, c)], 0), _le([(1, c)], 0),
             _eq([(-1, a), (1, d)], 0), _eq([(-1, b), (1, e)], 0), _eq([(-1, c), (1, f)], 0),
             _le([(-1, d), (1, e)], last_atom_rhs)]
    alphabet = LSBF_Alphabet.from_vars([a, b, c, d, e, f])
    nfa = MTBDDTransitionFn.construct_dfa_for_atom_conjunction(
        atoms, [], alphabet, stop_after_first_accepting_macrostate=stop_after_first_accepting_macrostate)

    binary_model = nfa.find_model()

    assert (binary_model is not None) == is_satisfiable
    if binary_model is not None:
        model = parse.convert_binary_model_into_decadic(binary_model, tuple(nfa.used_variables))
        assert all(_holds(atom, {var: model.get(var, 0) for var in (a, b, c, d, e, f)}) for atom in atoms), model


def test_model_of_free_variables_with_a_quantified_variable(mtbdd_solver_config):
    # exists y. (x = 3y + 1  and  0 <= y <= 5  and  x >= 7)  - x in {7, 10, 13, 16}
    atoms = [_eq([(1, X), (-3, Y)], 1), _le([(-1, Y)], 0), _le([(1, Y)], 5), _le([(-1, X)], -7)]

    nfa = _construct(atoms, [Y], stop_after_first_accepting_macrostate=True)
    binary_model = nfa.find_model()

    assert binary_model is not None
    x_value = parse.convert_binary_model_into_decadic(binary_model, tuple(nfa.used_variables), (X,))[X]
    assert x_value in {7, 10, 13, 16}


def test_matcher_accepts_conjunctions_of_atoms_under_quantifiers():
    from amaya import dsl

    atoms = (_le([(1, X)], 5), _le([(1, X), (-1, Y)], 0))
    formula = dsl._exists((Y,), dsl._and(*atoms))

    conjunction = parse.match_quantified_atom_conjunction(formula)

    assert conjunction is not None
    assert conjunction.atoms == atoms
    assert conjunction.bound_vars == (Y,)


def test_matcher_rejects_a_conjunction_with_a_disjunction():
    from amaya import dsl

    formula = dsl._and(_le([(1, X)], 5), dsl._or(_le([(1, Y)], 0), _le([(-1, Y)], -3)))

    assert parse.match_quantified_atom_conjunction(formula) is None


CONJUNCTION_SOURCES = [
    ("""
(declare-fun x () Int)
(declare-fun y () Int)
(assert (and (<= 3 x) (<= x 5) (= (mod (+ x y) 4) 1) (<= (+ x y) 9)))
(check-sat)
""", True),
    ("""
(declare-fun x () Int)
(assert (exists ((y Int)) (and (= x (+ (* 3 y) 1)) (<= 0 y) (<= y 5) (<= 7 x))))
(check-sat)
""", True),
    ("""
(declare-fun x () Int)
(assert (and (<= 4 x) (<= x 4) (= (mod x 2) 1)))
(check-sat)
""", False),
]


def _is_sat(source_text: str) -> bool:
    result = parse.perform_whole_evaluation_on_source_text(source_text)
    return result is not None and result.model is not None


@pytest.mark.parametrize('source_text, is_satisfiable', CONJUNCTION_SOURCES)
def test_whole_formula_verdict_does_not_depend_on_the_flag(source_text, is_satisfiable, mtbdd_solver_config):
    verdicts = []
    for flag_value in (False, True):
        solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction = flag_value
        verdicts.append(_is_sat(source_text))

    assert verdicts == [is_satisfiable, is_satisfiable]


DPLLT_SOURCES = [
    # phi is a conjunction of atoms; chi has a disjunction, so DPLL(T) enumerates implicants.
    ("""
(declare-fun x () Int)
(declare-fun y () Int)
(assert (<= 0 x))
(assert (<= x 10))
(assert (or (exists ((z Int)) (and (= x (* 4 z)) (<= 3 z)))
            (exists ((w Int)) (and (= y (+ x w)) (<= w 2) (<= 2 w) (= (mod y 3) 0)))))
(check-sat)
""", True),
    ("""
(declare-fun x () Int)
(assert (<= 0 x))
(assert (<= x 3))
(assert (or (exists ((z Int)) (and (= x (* 4 z)) (<= 1 z)))
            (exists ((w Int)) (and (= x (+ w 7)) (<= 0 w)))))
(check-sat)
""", False),
]


@pytest.mark.parametrize('source_text, is_satisfiable', DPLLT_SOURCES)
def test_dpllt_verdict_does_not_depend_on_the_flag(source_text, is_satisfiable, mtbdd_solver_config):
    verdicts = []
    for flag_value in (False, True):
        solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction = flag_value
        result = perform_whole_evaluation_on_source_text_with_dpllt_automata(source_text)
        verdicts.append(result is not None and result.model is not None)

    assert verdicts == [is_satisfiable, is_satisfiable]
