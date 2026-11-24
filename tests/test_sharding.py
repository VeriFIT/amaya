"""
Tests of top-level sharding (`--shard`, `OptimizationsConfig.allow_sharding`):
`amaya.parse.split_conjunction_to_shards` and `amaya.parse.evaluate_using_sharding`.

Regression: `split_conjunction_to_shards` assigned no shard to a conjunct that is a Boolean variable
(`Var`) or that references no variable (`BoolLiteral`, ground atom). `evaluate_using_sharding` then
decided the conjunction without that conjunct, and `--shard` reported sat for unsatisfiable formulae
such as `b and (not b) and x > 0`.
"""
import copy
import logging
import pathlib
import subprocess
import sys

import pytest

from amaya import logger as amaya_logger
from amaya import parse
from amaya.alphabet import LSBF_Alphabet
from amaya.config import BackendType, SolutionDomain, solver_config
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import (
    AST_Connective,
    AST_Negation,
    BoolLiteral,
    Connective_Type,
    Relation,
    Var,
    VariableType,
)
from amaya.solver_core import EvaluationContext


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SOLVER_ENTRY_POINT = REPOSITORY_ROOT / 'run-amaya.py'
PER_FORMULA_TIMEOUT_SECONDS = 60

BOOL_VAR_B = Var(id=1)
INT_VAR_X = Var(id=2)
INT_VAR_Y = Var(id=3)


def _less_or_equal(coefs_and_vars, rhs) -> Relation:
    return Relation(vars=[var for _, var in coefs_and_vars], coefs=[coef for coef, _ in coefs_and_vars],
                    rhs=rhs, predicate_symbol='<=')


def _conjunction(*children) -> AST_Connective:
    referenced_vars = sorted({var for child in children for var in _vars_of_conjunct(child)})
    return AST_Connective(referenced_vars=tuple(referenced_vars), type=Connective_Type.AND, children=tuple(children))


def _vars_of_conjunct(conjunct):
    match conjunct:
        case Var():
            return (conjunct,)
        case Relation():
            return tuple(conjunct.vars)
        case BoolLiteral():
            return ()
        case _:
            return tuple(conjunct.referenced_vars)


B_IS_FALSE = AST_Negation(referenced_vars=(BOOL_VAR_B,), child=BOOL_VAR_B)
X_IS_POSITIVE = _less_or_equal([(-1, INT_VAR_X)], -1)                       # x >= 1
X_IS_LESS_THAN_Y = _less_or_equal([(1, INT_VAR_X), (-1, INT_VAR_Y)], -1)    # x < y


def _assert_every_child_is_in_exactly_one_shard(and_expr, shard_child_indices):
    all_assigned_indices = sorted(index for shard in shard_child_indices for index in shard)
    assert all_assigned_indices == list(range(len(and_expr.children)))


def test_boolean_variable_conjunct_is_in_the_shard_of_its_variable():
    and_expr = _conjunction(X_IS_POSITIVE, BOOL_VAR_B, B_IS_FALSE, X_IS_LESS_THAN_Y)

    shard_child_indices, shard_vars = parse.split_conjunction_to_shards(and_expr)

    _assert_every_child_is_in_exactly_one_shard(and_expr, shard_child_indices)
    shard_of_vars = dict(zip(shard_vars, shard_child_indices))
    assert shard_of_vars[(BOOL_VAR_B,)] == (1, 2)
    assert shard_of_vars[(INT_VAR_X, INT_VAR_Y)] == (0, 3)


def test_conjuncts_without_variables_form_one_shard_without_variables():
    and_expr = _conjunction(X_IS_POSITIVE, BoolLiteral(value=False), BoolLiteral(value=True))

    shard_child_indices, shard_vars = parse.split_conjunction_to_shards(and_expr)

    _assert_every_child_is_in_exactly_one_shard(and_expr, shard_child_indices)
    shard_of_vars = dict(zip(shard_vars, shard_child_indices))
    assert shard_of_vars[()] == (1, 2)
    assert shard_of_vars[(INT_VAR_X,)] == (0,)


@pytest.fixture
def mtbdd_solver_config():
    """ MTBDD backend over integers with default optimizations, restoring the global config afterwards. """
    saved_backend = solver_config.backend_type
    saved_domain = solver_config.solution_domain
    saved_optimizations = copy.deepcopy(solver_config.optimizations)
    saved_log_level = amaya_logger.level

    solver_config.backend_type = BackendType.MTBDD
    solver_config.solution_domain = SolutionDomain.INTEGERS
    amaya_logger.setLevel(logging.WARNING)

    yield solver_config

    solver_config.backend_type = saved_backend
    solver_config.solution_domain = saved_domain
    solver_config.optimizations = saved_optimizations
    amaya_logger.setLevel(saved_log_level)


def _make_evaluation_context() -> EvaluationContext:
    var_table = {
        BOOL_VAR_B: VarInfo(name='b', type=VariableType.BOOL, is_formula_param=True),
        INT_VAR_X: VarInfo(name='x', type=VariableType.INT, is_formula_param=True),
        INT_VAR_Y: VarInfo(name='y', type=VariableType.INT, is_formula_param=True),
    }
    return EvaluationContext(alphabet=LSBF_Alphabet.from_vars(var_table.keys()), var_table=var_table)


@pytest.mark.parametrize('bool_literal_value, expected_is_sat', [(False, False), (True, True)])
def test_shard_without_variables_decides_the_conjunction(mtbdd_solver_config, bool_literal_value, expected_is_sat):
    and_expr = _conjunction(X_IS_POSITIVE, BoolLiteral(value=bool_literal_value))

    model = parse.evaluate_using_sharding(and_expr, _make_evaluation_context())

    assert (model is not None) == expected_is_sat
    if model is not None:
        assert model[INT_VAR_X] >= 1


SHARDING_FORMULAE = [
    pytest.param(
        """
        (declare-fun b () Bool)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (and b (not b) (> x 0) (> y x)))
        (check-sat)
        """,
        'unsat',
        id='boolean-variable-and-its-negation',
    ),
    pytest.param(
        # Preprocessing rewrites (=> b (< x 0)) to (not b) using x > 0.
        """
        (declare-fun b () Bool)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (and b (=> b (< x 0)) (> x 0) (> y 3)))
        (check-sat)
        """,
        'unsat',
        id='boolean-variable-and-guard-falsified-by-bound',
    ),
    pytest.param(
        """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (and (> x 0) (< x 5) (> y 10) (< y 3)))
        (check-sat)
        """,
        'unsat',
        id='two-independent-shards-one-unsat',
    ),
    pytest.param(
        """
        (declare-fun x () Int)
        (declare-fun y () Int)
        (declare-fun z () Int)
        (assert (and (> x 0) (< x 5) (> y 10) (< y 30) (= z (+ y y))))
        (check-sat)
        """,
        'sat',
        id='two-independent-shards-both-sat',
    ),
]


@pytest.mark.parametrize('solver_options', [('--fast',), ('--fast', '-O', 'all')], ids=['fast', 'fast-O-all'])
@pytest.mark.parametrize('source_text, expected_verdict', SHARDING_FORMULAE)
def test_sharded_verdict_matches_expected_verdict(tmp_path, source_text, expected_verdict, solver_options):
    """ Runs the solver CLI in its own process; the CLI sets the global solver configuration. """
    formula_path = tmp_path / 'formula.smt2'
    formula_path.write_text('(set-logic LIA)\n' + source_text)

    completed_process = subprocess.run(
        [sys.executable, str(SOLVER_ENTRY_POINT), *solver_options, '--shard', 'get-sat', str(formula_path)],
        capture_output=True,
        text=True,
        timeout=PER_FORMULA_TIMEOUT_SECONDS,
        cwd=REPOSITORY_ROOT,
    )

    output_lines = completed_process.stdout.strip().splitlines()
    solver_verdict = output_lines[-1].strip() if output_lines else ''
    assert solver_verdict == expected_verdict, completed_process.stderr[-2000:]
