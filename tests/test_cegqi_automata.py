"""
Tests for `amaya.cegqi_automata` (the experimental exists-forall strategy, see
`docs/CEGQI with automata.md`).

The end-to-end tests compare the strategy's verdict with the verdict known by construction, and, for
SAT verdicts, check the reported parameter values: the formula with the parameters fixed to them is
handed to the default evaluator, which must report SAT.
"""
import copy
import logging
from typing import Dict

import pytest

from amaya import parse
from amaya import logger as amaya_logger
from amaya.cegqi_automata import (
    Unsupported_Formula_Shape,
    build_bad_values_automaton,
    convert_to_negation_normal_form,
    detect_exists_forall_shape,
    eliminate_universal_vars_exactly,
    evaluate_negation_normal_form_at_point,
    evaluate_prepared_formula_with_cegqi_automata,
    extract_implicant_at_point,
    substitute_parameter_values,
)
from amaya.alphabet import LSBF_Alphabet
from amaya.config import BackendType, CegqiAutomataConfig, SolutionDomain, solver_config
from amaya.parse import convert_binary_model_into_decadic
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import (
    AST_Connective,
    AST_Negation,
    BoolLiteral,
    Congruence,
    Connective_Type,
    Relation,
    Var,
    VariableType,
)
from amaya.solver_core import EvaluationContext


@pytest.fixture(params=[False, True], ids=['projection', 'lazy-construction'])
def mtbdd_solver_config(request):
    """
    MTBDD over the integers. Parametrized by `do_lazy_evaluation`: the lazy construction keeps projected
    variables in `used_variables`, the projection path removes them, and the strategy must handle both.
    """
    saved_backend = solver_config.backend_type
    saved_domain = solver_config.solution_domain
    saved_cegqi = copy.deepcopy(solver_config.cegqi_automata)
    saved_optimizations = copy.deepcopy(solver_config.optimizations)
    saved_log_level = amaya_logger.level

    solver_config.backend_type = BackendType.MTBDD
    solver_config.solution_domain = SolutionDomain.INTEGERS
    solver_config.optimizations.do_lazy_evaluation = request.param
    solver_config.cegqi_automata = CegqiAutomataConfig(enabled=True, max_rounds=500)
    amaya_logger.setLevel(logging.WARNING)

    yield solver_config

    solver_config.backend_type = saved_backend
    solver_config.solution_domain = saved_domain
    solver_config.cegqi_automata = saved_cegqi
    solver_config.optimizations = saved_optimizations
    amaya_logger.setLevel(saved_log_level)


def make_source(parameter_declarations: Dict[str, str], assertion: str, extra_assertions=()) -> str:
    declarations = '\n'.join(f'(declare-fun {name} () {sort})' for name, sort in parameter_declarations.items())
    extra = '\n'.join(f'(assert {extra_assertion})' for extra_assertion in extra_assertions)
    return f'{declarations}\n(assert {assertion})\n{extra}\n(check-sat)\n'


def solve_with_cegqi(source: str) -> parse.Evaluation_Result:
    result = parse.perform_whole_evaluation_on_source_text(
        source, evaluate_prepared_formula=evaluate_prepared_formula_with_cegqi_automata)
    assert result is not None
    return result


def assert_model_satisfies_formula(parameter_declarations: Dict[str, str], assertion: str,
                                   result: parse.Evaluation_Result):
    model_by_name = {result.var_table[var].name: value for var, value in result.model.items()}
    fixing_assertions = []
    for name, sort in parameter_declarations.items():
        value = model_by_name.get(name, 0)
        if sort == 'Bool':
            fixing_assertions.append(name if value else f'(not {name})')
        else:
            fixing_assertions.append(f'(= {name} {value})' if value >= 0 else f'(= {name} (- {-value}))')
    check_source = make_source(parameter_declarations, assertion, fixing_assertions)
    check_result = parse.perform_whole_evaluation_on_source_text(check_source)
    assert check_result is not None
    assert check_result.model is not None, f'Reported model {model_by_name} does not satisfy the formula'


# (parameters, assertion, expected verdict)
EXISTS_FORALL_CASES = {
    'unit_lower_bound_sat': (
        {'c': 'Int'},
        '(forall ((x Int)) (=> (>= x 0) (>= (+ x c) 0)))',
        True,
    ),
    'unbounded_universal_unsat': (
        {'c': 'Int'},
        '(forall ((x Int)) (<= x c))',
        False,
    ),
    'guard_must_be_off_sat': (
        {'e': 'Bool', 'c': 'Int'},
        '(and (forall ((x Int)) (=> e (<= x c))) (or e (>= c 5)))',
        True,
    ),
    'guard_must_be_on_unsat': (
        {'e': 'Bool', 'c': 'Int'},
        '(and (forall ((x Int)) (=> e (<= x c))) e)',
        False,
    ),
    'non_unit_coefficient_parity_sat': (
        {'c': 'Int'},
        '(forall ((x Int)) (not (= (* 2 x) c)))',
        True,
    ),
    'non_unit_coefficient_parity_unsat': (
        {'c': 'Int'},
        '(forall ((x Int)) (and (not (= (* 2 x) c)) (not (= (+ (* 2 x) 1) c))))',
        False,
    ),
    'transition_template_sat': (
        {'c1': 'Int', 'c2': 'Int'},
        '(and (forall ((x Int) (y Int)) (=> (and (<= c1 x) (= y (+ x 1))) (<= c2 y))) (<= c2 3) (<= 7 c1))',
        True,
    ),
    'transition_template_unsat': (
        {'c1': 'Int', 'c2': 'Int'},
        '(and (forall ((x Int) (y Int)) (=> (and (<= c1 x) (= y (+ x 1))) (<= c2 y))) (>= c2 (+ c1 2)))',
        False,
    ),
    'guarded_template_sat': (
        {'e1': 'Bool', 'e2': 'Bool', 'c1': 'Int', 'c2': 'Int'},
        ('(and (forall ((x Int) (y Int)) (=> (and (=> e1 (<= c1 x)) (= y (* 3 x))) (=> e2 (<= c2 y))))'
         ' (or e2 (> c2 100)) (>= c2 0))'),
        True,
    ),
    'congruence_sat': (
        {'c': 'Int'},
        '(forall ((x Int)) (=> (= (mod x 3) 1) (not (= x c))))',
        True,
    ),
    # Regression: the `Bad` automaton of the first negated clause keeps the projected `x` among its
    # tracks; decoding a candidate must not assign `x`, or the second negated clause, which shares `x`,
    # is verified at that single value of `x` only.
    'shared_universal_var_unsat': (
        {'c': 'Int', 'd': 'Int'},
        '(forall ((x Int)) (and (not (and (<= d x) (<= x c) (<= (* 2 x) (+ c d)))) (not (= x 7))))',
        False,
    ),
    'two_clause_interval_unsat': (
        {'lo': 'Int', 'hi': 'Int'},
        ('(and (forall ((x Int)) (=> (and (<= lo x) (<= x hi)) (<= x 10)))'
         ' (forall ((x Int)) (=> (and (<= lo x) (<= x hi)) (>= x 0))) (<= lo 5) (>= hi 12))'),
        False,
    ),
}


@pytest.mark.parametrize('case_name', sorted(EXISTS_FORALL_CASES))
def test_cegqi_verdict_and_model(mtbdd_solver_config, monkeypatch, case_name):
    parameter_declarations, assertion, expected_is_sat = EXISTS_FORALL_CASES[case_name]
    source = make_source(parameter_declarations, assertion)

    def fail_on_fall_through(*args, **kwargs):
        raise AssertionError('The formula fell through to the default evaluator')

    with monkeypatch.context() as patch:
        patch.setattr(parse, 'evaluate_prepared_formula_with_automata', fail_on_fall_through)
        result = solve_with_cegqi(source)
    assert (result.model is not None) == expected_is_sat
    if expected_is_sat:
        assert_model_satisfies_formula(parameter_declarations, assertion, result)


@pytest.mark.parametrize('case_name', sorted(EXISTS_FORALL_CASES))
def test_cegqi_verdict_agrees_with_default_evaluator(mtbdd_solver_config, case_name):
    parameter_declarations, assertion, expected_is_sat = EXISTS_FORALL_CASES[case_name]
    source = make_source(parameter_declarations, assertion)
    default_result = parse.perform_whole_evaluation_on_source_text(source)
    assert default_result is not None
    assert (default_result.model is not None) == expected_is_sat


def test_shape_detection_rejects_deeper_alternation(mtbdd_solver_config):
    source = make_source({'c': 'Int'}, '(forall ((x Int)) (exists ((y Int)) (and (<= x y) (<= y (+ x c)))))')
    # Falls through to the default evaluator, which decides it.
    result = solve_with_cegqi(source)
    assert result.model is not None


# ---------------------------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------------------------

X, Y, C, E = Var(1), Var(2), Var(3), Var(4)


def _relation(vars, coefs, rhs, predicate_symbol='<='):
    return Relation(vars=list(vars), coefs=list(coefs), rhs=rhs, predicate_symbol=predicate_symbol)


def test_negation_normal_form_of_equation_is_two_inequalities():
    nnf = convert_to_negation_normal_form(_relation([X], [1], 3, '='), is_negated=True)
    for value in range(-2, 8):
        assert evaluate_negation_normal_form_at_point(nnf, {}, {X: value}) == (value != 3)


def test_negation_normal_form_of_congruence():
    congruence = Congruence(vars=[X], coefs=[1], rhs=1, modulus=3)
    nnf = convert_to_negation_normal_form(congruence, is_negated=True)
    for value in range(-6, 7):
        assert evaluate_negation_normal_form_at_point(nnf, {}, {X: value}) == ((value - 1) % 3 != 0)


def test_substitution_folds_parameter_only_atoms():
    node = AST_Connective(referenced_vars=(X, C, E), type=Connective_Type.AND, children=(
        _relation([C], [1], 5),
        AST_Connective(referenced_vars=(X, C, E), type=Connective_Type.OR, children=(
            AST_Negation(referenced_vars=(E,), child=E),
            _relation([X, C], [1, -1], 0),
        )),
    ))
    assert substitute_parameter_values(node, {E: True}, {C: 7}) == BoolLiteral(False)
    substituted = substitute_parameter_values(node, {E: True}, {C: 2})
    assert substituted == _relation([X], [1], 2)
    assert substitute_parameter_values(node, {E: False}, {C: 2}) == BoolLiteral(True)


def test_implicant_prefers_atoms_over_guard_literals():
    # (NOT e OR x - c <= 0) AND (x >= 10); at e=False, c=20, x=10 both disjuncts of the first conjunct hold.
    node = AST_Connective(referenced_vars=(X, C, E), type=Connective_Type.AND, children=(
        AST_Connective(referenced_vars=(X, C, E), type=Connective_Type.OR, children=(
            AST_Negation(referenced_vars=(E,), child=E),
            _relation([X, C], [1, -1], 0),
        )),
        _relation([X], [-1], -10),
    ))
    implicant = extract_implicant_at_point(node, {E: False}, {X: 10, C: 20})
    assert implicant.guard_literals == frozenset()
    assert len(implicant.atoms) == 2

    implicant = extract_implicant_at_point(node, {E: False}, {X: 10, C: 0})
    assert implicant.guard_literals == frozenset({(E, False)})
    assert len(implicant.atoms) == 1


def test_exact_elimination_by_unit_equation_and_one_sided_variable():
    # exists x y. y = x + 1 AND c <= x AND y <= 10   ==>   c <= 9
    atoms = [_relation([X, Y], [-1, 1], 1, '='), _relation([X, C], [-1, 1], 0), _relation([Y], [1], 10)]
    result = eliminate_universal_vars_exactly(atoms, [X, Y])
    assert result is not None
    remaining_atoms, eliminated_var_count = result
    # x is substituted by y - 1; y is then bounded on both sides and stays for the automaton.
    assert eliminated_var_count == 1
    for c_value in range(0, 15):
        expected = c_value <= 9
        is_satisfiable = any(
            all(evaluate_negation_normal_form_at_point(atom, {}, {C: c_value, Y: y_value}) for atom in remaining_atoms)
            for y_value in range(-30, 30))
        assert is_satisfiable == expected


def test_one_sided_elimination_drops_relations():
    # exists x. c <= x AND 2c <= 3x: x occurs with negative coefficients only.
    atoms = [_relation([X, C], [-1, 1], 0), _relation([X, C], [-3, 2], 0)]
    remaining_atoms, eliminated_var_count = eliminate_universal_vars_exactly(atoms, [X])
    assert remaining_atoms == []
    assert eliminated_var_count == 1


def test_bad_values_automaton_and_its_complement(mtbdd_solver_config):
    # Bad = { c | exists x. 2x = c AND 0 <= x <= 3 } = {0, 2, 4, 6}
    var_table = {X: VarInfo(name='x', type=VariableType.INT), C: VarInfo(name='c', type=VariableType.INT)}
    alphabet = LSBF_Alphabet.from_vars(var_table.keys())
    ctx = EvaluationContext(alphabet=alphabet, var_table=var_table)
    atoms = [_relation([X, C], [2, -1], 0, '='), _relation([X], [-1], 0), _relation([X], [1], 3)]
    bad = build_bad_values_automaton(atoms, [X], ctx)
    complemented_bad = bad.complement()
    from amaya.mtbdd_transitions import MTBDDTransitionFn
    for c_value in range(-3, 10):
        c_automaton = MTBDDTransitionFn.construct_nfa_for_eq(_relation([C], [1], c_value, '='), alphabet)
        is_bad = c_value in (0, 2, 4, 6)
        assert (bad.intersection(c_automaton).find_model() is not None) == is_bad
        assert (complemented_bad.intersection(c_automaton).find_model() is not None) == (not is_bad)


# ---------------------------------------------------------------------------------------------
# Forward ground seeding
# ---------------------------------------------------------------------------------------------

def make_counter_transition_system(bad_value: int) -> str:
    """
    One location with the interval template `e_lo -> c_lo <= v` and `e_hi -> v <= c_hi`, initial state
    0, transition `x <= 2 -> x' = x + 1`, safety `x != bad_value`. The reachable states are 0..3, so an
    invariant exists iff `bad_value` is outside 0..3.
    """
    def invariant(var: str) -> str:
        return f'(and (=> e_lo (<= c_lo {var})) (=> e_hi (<= {var} c_hi)))'
    return (
        '(assert (exists ((e_lo Bool) (c_lo Int) (e_hi Bool) (c_hi Int)) (forall ((x_0 Int) (y_0 Int)) (and'
        f' (=> (= y_0 0) {invariant("y_0")})'
        f' (=> (and {invariant("x_0")} (<= x_0 2) (= y_0 (+ x_0 1))) {invariant("y_0")})'
        f' (=> {invariant("x_0")} (not (= x_0 {bad_value})))))))\n(check-sat)\n'
    )


@pytest.mark.parametrize('seed_depth', [0, 2, 5])
@pytest.mark.parametrize('bad_value, expected_is_sat', [(3, False), (5, True), (-1, True)])
def test_cegqi_with_seeding_on_counter_system(mtbdd_solver_config, seed_depth, bad_value, expected_is_sat):
    solver_config.cegqi_automata.seed_depth = seed_depth
    result = solve_with_cegqi(make_counter_transition_system(bad_value))
    assert (result.model is not None) == expected_is_sat


def test_seeding_alone_refutes_counter_system(mtbdd_solver_config):
    """ With depth 5 the seeds cover states 0..3; the first synthesis call has no candidate. """
    solver_config.cegqi_automata.seed_depth = 5
    solver_config.cegqi_automata.max_rounds = 1
    result = solve_with_cegqi(make_counter_transition_system(3))
    assert result.model is None


def test_counter_system_verdicts_agree_with_default_evaluator(mtbdd_solver_config):
    for bad_value, expected_is_sat in [(3, False), (5, True)]:
        result = parse.perform_whole_evaluation_on_source_text(make_counter_transition_system(bad_value))
        assert (result.model is not None) == expected_is_sat


@pytest.mark.parametrize('generalization', ['point', 'disjunct-and-point'])
@pytest.mark.parametrize('counterexample_seed_depth', [0, 3])
@pytest.mark.parametrize('bad_value, expected_is_sat', [(3, False), (5, True)])
def test_cegqi_point_generalization_and_counterexample_seeding(mtbdd_solver_config, generalization,
                                                               counterexample_seed_depth, bad_value, expected_is_sat):
    solver_config.cegqi_automata.generalization = generalization
    solver_config.cegqi_automata.counterexample_seed_depth = counterexample_seed_depth
    result = solve_with_cegqi(make_counter_transition_system(bad_value))
    assert (result.model is not None) == expected_is_sat


@pytest.mark.parametrize('case_name', sorted(EXISTS_FORALL_CASES))
def test_cegqi_disjunct_and_point_generalization(mtbdd_solver_config, case_name):
    solver_config.cegqi_automata.generalization = 'disjunct-and-point'
    parameter_declarations, assertion, expected_is_sat = EXISTS_FORALL_CASES[case_name]
    result = solve_with_cegqi(make_source(parameter_declarations, assertion))
    assert (result.model is not None) == expected_is_sat
    if expected_is_sat:
        assert_model_satisfies_formula(parameter_declarations, assertion, result)
