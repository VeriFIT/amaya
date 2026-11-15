from amaya import dsl
from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.preprocessing.unit_coefficient_fourier_motzkin import eliminate_unit_coefficient_vars_by_fourier_motzkin
from amaya.relations_structures import (
    AST_Negation,
    BoolLiteral,
    Congruence,
    Relation,
    Var,
    VariableType,
)
from tests.test_conditional_equality_resolution import assert_semantically_equivalent


X, Y, C, R, V, W = (Var(i) for i in range(1, 7))


def _var_table(*int_vars: Var) -> dict:
    return {var: VarInfo(name=f'v{var.id}', type=VariableType.INT) for var in int_vars}


def _le(var_coef_pairs: list[tuple[int, Var]], rhs: int) -> Relation:
    """ sum(coef * var) <= rhs, with the variables sorted by id as `_linear_combo` produces them. """
    sorted_pairs = sorted(var_coef_pairs, key=lambda coef_var_pair: coef_var_pair[1])
    return Relation(vars=[var for _coef, var in sorted_pairs], coefs=[coef for coef, _var in sorted_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _eliminate(formula, *int_vars: Var, quantified_vars_are_nonnegative: bool = False):
    fill_referenced_vars(formula)
    return eliminate_unit_coefficient_vars_by_fourier_motzkin(formula, _var_table(*int_vars), quantified_vars_are_nonnegative)


def test_lower_bounds_are_paired_with_upper_bound():
    # exists v. (v >= 5  and  v >= 2y + 5  and  v <= x)   --->   (x >= 5  and  x >= 2y + 5)
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], -5),
        _le([(-1, V), (2, Y)], -5),
        _le([(1, V), (-1, X)], 0),
    ))

    result = _eliminate(formula, X, Y, V)

    expected = dsl._and(
        _le([(-1, X)], -5),
        _le([(-1, X), (2, Y)], -5),
    )
    fill_referenced_vars(expected)
    assert result == expected
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X, Y])


def test_unit_coefficient_equation_contributes_both_bounds():
    # exists v. (v - 5c - r = 0  and  v <= -1  and  v <= x)   --->   (5c + r <= -1  and  5c + r - x <= 0)
    formula = dsl._exists((V,), dsl._and(
        dsl._eq([(1, V), (-5, C), (-1, R)], 0),
        _le([(1, V)], -1),
        _le([(1, V), (-1, X)], 0),
    ))

    result = _eliminate(formula, X, C, R, V)

    expected = dsl._and(
        _le([(5, C), (1, R)], -1),
        _le([(-1, X), (5, C), (1, R)], 0),
    )
    fill_referenced_vars(expected)
    assert result == expected
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X, C, R])


def test_equation_with_negative_coefficient_at_eliminated_var():
    # exists v. (-v + 5c = 0  and  0 <= v  and  v <= x)  <=>  (0 <= 5c  and  5c <= x)
    formula = dsl._exists((V,), dsl._and(
        dsl._eq([(-1, V), (5, C)], 0),
        _le([(-1, V)], 0),
        _le([(1, V), (-1, X)], 0),
    ))

    result = _eliminate(formula, X, C, V)

    assert V not in result.referenced_vars
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X, C])


def test_squeezed_variable_under_negation_with_outer_quantifier():
    # Conjunct 3 of Problem15_label00_false-unreach-call.c_17.smt2 with small constants:
    # not exists c. (c <= 2  and  y <= c  and  exists v. (0 <= v  and  v <= x  and  5c <= v  and  v <= 5c + 4))
    inner = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        _le([(1, V), (-1, X)], 0),
        _le([(5, C), (-1, V)], 0),
        _le([(1, V), (-5, C)], 4),
    ))
    formula = dsl._neg(dsl._exists((C,), dsl._and(
        _le([(1, C)], 2),
        _le([(1, Y), (-1, C)], 0),
        inner,
    )))

    result = _eliminate(formula, X, Y, C, V)

    assert V not in result.referenced_vars
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X, Y])


def test_variable_with_only_lower_bounds_is_dropped_with_its_bounds():
    # exists v. (v >= x  and  v >= y  and  x <= 3)   --->   x <= 3
    x_le_3 = _le([(1, X)], 3)
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V), (1, X)], 0),
        _le([(-1, V), (1, Y)], 0),
        x_le_3,
    ))

    result = _eliminate(formula, X, Y, V)

    assert result == x_le_3


def test_contradictory_constant_bounds_yield_false():
    # exists v. (v >= 3  and  v <= 2)   --->   False
    formula = dsl._exists((V,), dsl._and(_le([(-1, V)], -3), _le([(1, V)], 2)))

    result = _eliminate(formula, V)

    assert result == BoolLiteral(False)


def test_eliminating_one_variable_enables_another():
    # exists v, w. (-v + w <= 0  and  v - 2w <= 0  and  w <= x)
    # w has coefficient -2 in the second atom; it becomes eliminable once v is eliminated: -w <= 0.
    formula = dsl._exists((V, W), dsl._and(
        _le([(-1, V), (1, W)], 0),
        _le([(1, V), (-2, W)], 0),
        _le([(1, W), (-1, X)], 0),
    ))

    result = _eliminate(formula, X, V, W)

    assert result == _le([(-1, X)], 0)
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X])


def test_combined_atom_is_divided_by_gcd_with_floor_of_rhs():
    # exists v. (0 <= v  and  v <= 5w + 7)   --->   -5w <= 7   --->   -w <= floor(7/5) = 1
    formula = dsl._exists((V,), dsl._and(_le([(-1, V)], 0), _le([(1, V), (-5, W)], 7)))

    result = _eliminate(formula, W, V)

    assert result == _le([(-1, W)], 1)
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[W])


def test_gcd_division_makes_another_variable_eliminable():
    # exists v, w. (0 <= v  and  v <= 5w + 300007  and  w <= -3  and  w <= x - 449582)
    # Eliminating v gives -5w <= 300007, i.e. -w <= 60001; w is then eliminable: x >= 389581.
    formula = dsl._exists((V, W), dsl._and(
        _le([(-1, V)], 0),
        _le([(1, V), (-5, W)], 300007),
        _le([(1, W)], -3),
        _le([(1, W), (-1, X)], -449582),
    ))

    result = _eliminate(formula, X, V, W)

    assert result == _le([(-1, X)], -389581)


def test_non_unit_coefficient_is_left_unchanged():
    formula = dsl._exists((V,), dsl._and(_le([(-2, V)], 0), _le([(1, V), (-1, X)], 0)))

    result = _eliminate(formula, X, V)

    assert result == formula


def test_variable_in_congruence_is_left_unchanged():
    congruence = Congruence(vars=[V], coefs=[1], rhs=3, modulus=5)
    formula = dsl._exists((V,), dsl._and(congruence, _le([(-1, V)], 0), _le([(1, V), (-1, X)], 0)))

    result = _eliminate(formula, X, V)

    assert result == formula


def test_variable_in_negated_equation_is_left_unchanged():
    disequation = AST_Negation(referenced_vars=(), child=dsl._eq([(1, V), (-1, Y)], 0))
    formula = dsl._exists((V,), dsl._and(disequation, _le([(-1, V)], 0), _le([(1, V), (-1, X)], 0)))

    result = _eliminate(formula, X, Y, V)

    assert result == formula


def test_rewrite_increasing_the_number_of_atoms_is_not_applied():
    # 2 lower bounds and 3 upper bounds: 6 combined atoms replace 5.
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        _le([(-1, V), (1, Y)], 0),
        _le([(1, V), (-1, X)], 0),
        _le([(1, V), (-1, C)], 0),
        _le([(1, V)], 10),
    ))

    result = _eliminate(formula, X, Y, C, V)

    assert result == formula


def test_natural_numbers_domain_adds_nonnegativity_of_eliminated_var():
    # exists v. v <= x   --->   True over the integers, 0 <= x over the natural numbers
    formula_over_integers = dsl._exists((V,), _le([(1, V), (-1, X)], 0))
    formula_over_naturals = dsl._exists((V,), _le([(1, V), (-1, X)], 0))

    assert _eliminate(formula_over_integers, X, V) == BoolLiteral(True)
    assert _eliminate(formula_over_naturals, X, V, quantified_vars_are_nonnegative=True) == _le([(-1, X)], 0)


# --- Bounds in a nested existential quantifier ---------------------------------------------------

def test_bound_in_nested_quantifier_is_used_and_both_quantifiers_are_eliminated():
    # exists v. (0 <= v  and  v <= 6  and  exists w. (w <= -1  and  w <= x - 2  and  v <= 5w + 12))
    # v: 0 <= 6 (outside), -5w <= 12 -> -w <= 2 (inside); then w: -2 <= -1, -2 <= x - 2  ->  x >= 0
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        _le([(1, V)], 6),
        dsl._exists((W,), dsl._and(
            _le([(1, W)], -1),
            _le([(1, W), (-1, X)], -2),
            _le([(1, V), (-5, W)], 12),
        )),
    ))

    result = _eliminate(formula, X, V, W)

    assert result == _le([(-1, X)], 0)
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X])


def test_nested_quantifier_is_kept_when_its_variable_is_not_eliminable():
    # exists v. (0 <= v  and  exists w. (v <= w + 1  and  2w <= x))
    #   --->  exists w. (2w <= x  and  -w <= 1)            (w has coefficient 2 in 2w <= x)
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        dsl._exists((W,), dsl._and(
            _le([(1, V), (-1, W)], 1),
            _le([(-1, X), (2, W)], 0),
        )),
    ))

    result = _eliminate(formula, X, V, W)

    expected = dsl._exists((W,), dsl._and(
        _le([(-1, X), (2, W)], 0),
        _le([(-1, W)], 1),
    ))
    fill_referenced_vars(expected)
    assert result == expected
    assert_semantically_equivalent(formula, result, bool_vars=[], int_vars=[X])


def test_original_constants_of_the_two_level_shape():
    # exists v. (0 <= v  and  v <= 299992  and  exists w. (w <= -3  and  w <= x - 449582  and  v <= 5w + 300007))
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        _le([(1, V)], 299992),
        dsl._exists((W,), dsl._and(
            _le([(1, W)], -3),
            _le([(1, W), (-1, X)], -449582),
            _le([(1, V), (-5, W)], 300007),
        )),
    ))

    result = _eliminate(formula, X, V, W)

    assert result == _le([(-1, X)], -389581)


def test_nested_bound_var_occurring_in_another_conjunct_blocks_the_rewrite():
    # The free w in `w <= 3` and the w bound by the nested quantifier are the same Var.
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        _le([(1, W)], 3),
        dsl._exists((W,), dsl._and(
            _le([(1, V), (-1, W)], 0),
            _le([(-1, X), (2, W)], 0),
        )),
    ))
    fill_referenced_vars(formula)

    result = _eliminate(formula, X, V, W)

    assert result == formula


def test_bounds_in_two_sibling_quantifiers_block_the_rewrite():
    formula = dsl._exists((V,), dsl._and(
        dsl._exists((W,), dsl._and(
            _le([(1, V), (-1, W)], 0),
            _le([(-1, X), (2, W)], 0),
        )),
        dsl._exists((C,), dsl._and(
            _le([(-1, V), (1, C)], 0),
            _le([(-2, C), (-1, Y)], 0),
        )),
    ))
    fill_referenced_vars(formula)

    result = _eliminate(formula, X, Y, V, W, C)

    assert result == formula


def test_eliminated_var_rebound_by_nested_quantifier_blocks_the_rewrite():
    formula = dsl._exists((V,), dsl._and(
        _le([(-1, V)], 0),
        dsl._exists((V,), dsl._and(
            _le([(1, V), (-1, X)], 0),
            _le([(-2, V), (1, Y)], 0),
        )),
    ))
    fill_referenced_vars(formula)

    result = _eliminate(formula, X, Y, V)

    assert result == formula


def test_bool_variable_is_not_eliminated():
    formula = dsl._exists((V,), dsl._and(V, _le([(1, X)], 0)))
    fill_referenced_vars(formula)
    var_table = {X: VarInfo(name='x', type=VariableType.INT), V: VarInfo(name='v', type=VariableType.BOOL)}

    result = eliminate_unit_coefficient_vars_by_fourier_motzkin(formula, var_table)

    assert result == formula
