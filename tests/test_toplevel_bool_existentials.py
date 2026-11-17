"""
`amaya/preprocessing/toplevel_bool_existentials.py`: the Boolean variables of positive-position
existential quantifiers become free, and the consumers treat them as assignable.
"""
from amaya import dsl
from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.preprocessing.theory_reasoning import (
    Variable_Use_Info,
    remove_atoms_satisfied_by_unconstrained_vars,
    scan_variable_use,
)
from amaya.preprocessing.toplevel_bool_existentials import lift_bool_vars_of_toplevel_existentials
from amaya.relations_structures import AST_Connective, AST_Negation, AST_Quantifier, BoolLiteral, Relation, Var, VariableType
from amaya.sat_toplevel import is_assignable_free_bool_var


B, C, X, Y = Var(1), Var(2), Var(3), Var(4)


def _var_table() -> dict:
    return {
        B: VarInfo(name='b', type=VariableType.BOOL),
        C: VarInfo(name='c', type=VariableType.BOOL),
        X: VarInfo(name='x', type=VariableType.INT),
        Y: VarInfo(name='y', type=VariableType.INT, is_formula_param=True),
    }


def _le(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _strip_referenced_vars(node):
    match node:
        case AST_Quantifier():
            return dsl._exists(node.bound_vars, _strip_referenced_vars(node.child))
        case AST_Negation():
            return dsl._neg(_strip_referenced_vars(node.child))
        case AST_Connective():
            return AST_Connective(referenced_vars=(), type=node.type,
                                  children=tuple(_strip_referenced_vars(child) for child in node.children))
        case _:
            return node


def test_bool_variables_of_a_toplevel_existential_become_free_and_int_variables_stay_bound():
    var_table = _var_table()
    body = dsl._and(B, _le([(1, X), (-1, Y)], 0))
    formula = dsl._exists((B, X), body)

    result = lift_bool_vars_of_toplevel_existentials(formula, var_table)

    assert _strip_referenced_vars(result) == dsl._exists((X,), body)
    assert var_table[B].is_lifted_from_toplevel_existential
    assert not var_table[B].is_formula_param
    assert not var_table[X].is_lifted_from_toplevel_existential


def test_quantifier_binding_only_bool_variables_is_removed():
    var_table = _var_table()
    body = dsl._or(B, C)
    formula = dsl._and(_le([(1, Y)], 3), dsl._exists((B, C), body))

    result = lift_bool_vars_of_toplevel_existentials(formula, var_table)

    assert _strip_referenced_vars(result) == dsl._and(_le([(1, Y)], 3), body)


def test_existential_below_a_negation_is_not_lifted():
    var_table = _var_table()
    formula = dsl._neg(dsl._exists((B,), dsl._and(B, _le([(1, Y)], 0))))

    result = lift_bool_vars_of_toplevel_existentials(formula, var_table)

    assert _strip_referenced_vars(result) == _strip_referenced_vars(formula)
    assert not var_table[B].is_lifted_from_toplevel_existential


def test_existential_below_an_existential_and_a_disjunction_is_lifted():
    var_table = _var_table()
    inner = dsl._exists((B,), dsl._and(B, _le([(1, X)], 0)))
    formula = dsl._exists((X,), dsl._or(inner, _le([(1, X), (-1, Y)], 0)))

    result = lift_bool_vars_of_toplevel_existentials(formula, var_table)

    assert _strip_referenced_vars(result) == dsl._exists((X,), dsl._or(dsl._and(B, _le([(1, X)], 0)),
                                                                      _le([(1, X), (-1, Y)], 0)))


def test_lifted_bool_variable_is_assignable_by_the_toplevel_sat_loop():
    var_table = _var_table()
    lift_bool_vars_of_toplevel_existentials(dsl._exists((B,), B), var_table)

    assert is_assignable_free_bool_var(B, var_table)
    assert not is_assignable_free_bool_var(C, var_table)


def test_pure_literal_rule_fixes_a_lifted_variable_but_not_another_free_variable():
    """ `not (b and y <= 0) and (c or y >= 5)`: `b` is lifted and occurs only negatively, `c` is free. """
    formula = dsl._and(dsl._neg(dsl._and(B, _le([(1, Y)], 0))), dsl._or(C, _le([(-1, Y)], -5)))
    fill_referenced_vars(formula)

    var_use_info = Variable_Use_Info()
    scan_variable_use(formula, var_use_info, free_vars_that_may_be_fixed=frozenset({B}))
    result = remove_atoms_satisfied_by_unconstrained_vars(formula, var_use_info, desired_polarity=True)

    assert var_use_info.get_bool_var_desired_value(B) is False
    assert var_use_info.get_bool_var_desired_value(C) is None
    assert _strip_referenced_vars(result) == dsl._or(C, _le([(-1, Y)], -5))


def test_skeleton_contains_a_lifted_variable_that_occurs_only_below_a_quantifier():
    """
    `exists x. (b and x <= y)` with `b` lifted: the quantifier is one opaque atom of the abstraction, and
    `b` is added to it as an unconstrained variable that the SAT loop assigns.
    """
    from types import SimpleNamespace
    from amaya.sat_toplevel import extract_bool_skeleton, isolated_sat_formula_context

    var_table = _var_table()
    formula = lift_bool_vars_of_toplevel_existentials(
        dsl._exists((B, X), dsl._and(B, _le([(1, X), (-1, Y)], 0))), var_table)

    with isolated_sat_formula_context():
        skeleton = extract_bool_skeleton(formula, SimpleNamespace(var_table=var_table))

        assert set(skeleton.free_bool_var_atoms) == {B}
        assert B in skeleton.solver_var_id_of_free_bool_var


def _with_optimizations(**enabled_optimizations):
    import contextlib
    from amaya.config import solver_config

    @contextlib.contextmanager
    def manager():
        saved_values = {name: getattr(solver_config.optimizations, name) for name in enabled_optimizations}
        for name, value in enabled_optimizations.items():
            setattr(solver_config.optimizations, name, value)
        try:
            yield
        finally:
            for name, value in saved_values.items():
                setattr(solver_config.optimizations, name, value)
    return manager()


def test_residual_optimization_eliminates_a_quantified_variable_of_unguarded_bounds():
    """ The residual `exists x. (y <= x and x <= 3)` becomes `y <= 3` by unit-coefficient FM. """
    from types import SimpleNamespace
    from amaya.sat_toplevel import _optimize_residual

    var_table = _var_table()
    residual = dsl._exists((X,), dsl._and(_le([(-1, X), (1, Y)], 0), _le([(1, X)], 3)))
    fill_referenced_vars(residual)

    with _with_optimizations(eliminate_unit_coefficient_vars_by_fourier_motzkin=True):
        result = _optimize_residual(residual, SimpleNamespace(var_table=var_table))

    assert _strip_referenced_vars(result) == _le([(1, Y)], 3)


def test_residual_optimization_allocating_variables_is_discarded():
    """
    Miniscoping `exists x. (x <= y or x >= y + 5)` binds a fresh copy of `x` in the second disjunct; the
    copy is not in the alphabet of the run, so the residual is returned unchanged and the copy removed.
    """
    from types import SimpleNamespace
    from amaya.sat_toplevel import _optimize_residual

    var_table = _var_table()
    residual = dsl._exists((X,), dsl._or(_le([(1, X), (-1, Y)], 0), _le([(-1, X), (1, Y)], -5)))
    fill_referenced_vars(residual)
    var_table_before = dict(var_table)

    with _with_optimizations(do_miniscoping=True):
        result = _optimize_residual(residual, SimpleNamespace(var_table=var_table))

    assert result is residual
    assert var_table == var_table_before
