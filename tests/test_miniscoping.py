"""
`antiprenexing.miniscope_quantifiers`: distribution of an existential quantifier over a disjunction
(`antiprenexing._distribute_quantifier_over_disjunction`).
"""
from amaya import dsl
from amaya.preprocessing.antiprenexing import miniscope_quantifiers
from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.relations_structures import AST_Connective, AST_Quantifier, Connective_Type, Relation, Var


def _le(var_coef_pairs, rhs: int) -> Relation:
    return Relation(vars=[var for _, var in var_coef_pairs], coefs=[coef for coef, _ in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


def _miniscope(formula):
    fill_referenced_vars(formula)
    return miniscope_quantifiers(formula)


def _strip_referenced_vars(node):
    """Rebuild `node` with empty `referenced_vars` so that results can be compared structurally."""
    match node:
        case AST_Quantifier():
            return dsl._exists(node.bound_vars, _strip_referenced_vars(node.child))
        case AST_Connective():
            return AST_Connective(referenced_vars=(), type=node.type,
                                  children=tuple(_strip_referenced_vars(child) for child in node.children))
        case _:
            return node


def test_variable_occurring_in_every_disjunct_is_bound_in_each_disjunct():
    x, y, s = Var(1), Var(2), Var(3)
    first_disjunct = _le([(1, x), (1, y), (-1, s)], 0)
    second_disjunct = _le([(-1, x), (1, y), (1, s)], -3)
    formula = dsl._exists((x, y), dsl._or(first_disjunct, second_disjunct))

    result = _strip_referenced_vars(_miniscope(formula))

    assert result == dsl._or(dsl._exists((x, y), first_disjunct),
                             dsl._exists((x, y), second_disjunct))


def test_disjunct_without_bound_variables_is_not_quantified():
    x, s = Var(1), Var(2)
    quantified_disjunct = _le([(1, x), (-1, s)], 0)
    free_disjunct = _le([(1, s)], 3)
    formula = dsl._exists((x,), dsl._or(quantified_disjunct, free_disjunct))

    result = _strip_referenced_vars(_miniscope(formula))

    assert result == dsl._or(dsl._exists((x,), quantified_disjunct), free_disjunct)


def test_each_disjunct_binds_only_the_variables_it_references():
    x, y, s = Var(1), Var(2), Var(3)
    disjunct_with_x = _le([(1, x), (-1, s)], 0)
    disjunct_with_x_and_y = _le([(1, x), (1, y), (-1, s)], 1)
    formula = dsl._exists((x, y), dsl._or(disjunct_with_x, disjunct_with_x_and_y))

    result = _strip_referenced_vars(_miniscope(formula))

    assert result == dsl._or(dsl._exists((x,), disjunct_with_x),
                             dsl._exists((x, y), disjunct_with_x_and_y))


def test_conjunction_inside_a_disjunct_is_miniscoped_after_distribution():
    """`∃x,y. ((y = 4 ∧ x <= s) ∨ x >= s)`: in the first disjunct `x` and `y` occur in different conjuncts."""
    x, y, s = Var(1), Var(2), Var(3)
    equation_on_y = dsl._eq([(1, y)], 4)
    bound_on_x = _le([(1, x), (-1, s)], 0)
    second_disjunct = _le([(-1, x), (1, s)], 0)
    formula = dsl._exists((x, y), dsl._or(dsl._and(equation_on_y, bound_on_x), second_disjunct))

    result = _strip_referenced_vars(_miniscope(formula))

    assert result == dsl._or(dsl._and(dsl._exists((x,), bound_on_x), dsl._exists((y,), equation_on_y)),
                             dsl._exists((x,), second_disjunct))


def test_quantifier_under_negation_is_distributed_over_the_disjunction_below_it():
    x, s = Var(1), Var(2)
    first_disjunct = _le([(1, x), (-1, s)], 0)
    second_disjunct = _le([(-1, x), (1, s)], -5)
    formula = dsl._neg(dsl._exists((x,), dsl._or(first_disjunct, second_disjunct)))

    result = _miniscope(formula)

    assert result.child.type == Connective_Type.OR
    assert all(isinstance(child, AST_Quantifier) and child.bound_vars == (x,) for child in result.child.children)
