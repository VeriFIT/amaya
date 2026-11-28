import pytest

from amaya import dsl
from amaya.converters import format_smt2_attribute_value, format_smt2_symbol, generate_optimization_problem, write_ast_in_smt2
from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import AST_Quantifier, BoolLiteral, Relation, Var, VariableType


X, X_PRIMED, B = Var(1), Var(2), Var(3)


def _le(var_coef_pairs, rhs):
    return Relation(vars=[var for _coef, var in var_coef_pairs], coefs=[coef for coef, _var in var_coef_pairs],
                    rhs=rhs, predicate_symbol='<=')


@pytest.mark.parametrize('name', ['x', 'c_~a1~0', 'v_prenex_247', '<=x', 'a.b?c'])
def test_simple_symbol_is_written_unchanged(name):
    assert format_smt2_symbol(name) == name


@pytest.mark.parametrize('name', ['c_old(~a1~0)', 'main_#t~nondet0', 'congruence replacement', '1x', 'exists', ''])
def test_non_simple_symbol_or_reserved_word_is_quoted(name):
    assert format_smt2_symbol(name) == f'|{name}|'


@pytest.mark.parametrize('name', ['a|b', 'a\\b'])
def test_name_not_representable_as_quoted_symbol_raises(name):
    with pytest.raises(ValueError):
        format_smt2_symbol(name)


def test_declarations_binders_and_atoms_use_quoted_names():
    var_table = {
        X: VarInfo(name='c_old(~a1~0)', type=VariableType.INT, is_formula_param=True),
        B: VarInfo(name='main_#t~nondet0', type=VariableType.INT),
    }
    formula = dsl._exists((B,), _le([(1, X), (-1, B)], 0))
    fill_referenced_vars(formula)

    output = write_ast_in_smt2(formula, [(X, var_table[X])], var_table, smt_info={})

    assert '(declare-fun |c_old(~a1~0)| () Int)' in output
    assert '(exists ((|main_#t~nondet0| Int))' in output
    assert '(+ (* 1 |c_old(~a1~0)|) (* (- 1) |main_#t~nondet0|))' in output


def test_primed_copies_do_not_reuse_names_of_existing_variables():
    # The input has both `x` and `x_primed` as parameters; the primed copy of `x` must not be named `x_primed`.
    var_table = {
        X: VarInfo(name='x', type=VariableType.INT, is_formula_param=True),
        X_PRIMED: VarInfo(name='x_primed', type=VariableType.INT, is_formula_param=True),
        Var(7): VarInfo(name='bound', type=VariableType.INT),  # var ids are not contiguous
    }
    formula = _le([(1, X), (-1, X_PRIMED)], 0)

    result = generate_optimization_problem(formula, var_table, [X, X_PRIMED])

    optimality_quantifier = result.children[1].child
    assert isinstance(optimality_quantifier, AST_Quantifier)
    primed_vars = optimality_quantifier.bound_vars
    assert all(primed_var.id > 7 for primed_var in primed_vars)
    primed_names = [var_table[primed_var].name for primed_var in primed_vars]
    assert primed_names == ['x_primed_1', 'x_primed_primed']
    all_names = [var_info.name for var_info in var_table.values()]
    assert len(all_names) == len(set(all_names))


@pytest.mark.parametrize(('attribute_value', 'expected_text'), [
    ('unsat', 'unsat'),
    ('2.6', '2.6'),
    ('https://creativecommons.org/licenses/by/4.0/', '"https://creativecommons.org/licenses/by/4.0/"'),
    ('say "hi"', '"say ""hi"""'),
])
def test_attribute_value_is_symbol_numeral_or_string_literal(attribute_value, expected_text):
    assert format_smt2_attribute_value(attribute_value) == expected_text


def test_preserved_status_is_written_as_symbol():
    smt2_text = write_ast_in_smt2(BoolLiteral(True), [], {}, {'status': 'unsat'})
    assert '(set-info :status unsat)' in smt2_text
