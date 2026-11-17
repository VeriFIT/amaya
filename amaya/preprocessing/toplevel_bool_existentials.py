"""
(EXPERIMENTAL, disabled by default; enabled by `--lift-toplevel-bool-existentials`.) Make the Boolean
variables of positive-position existential quantifiers free variables of the formula.

A quantifier `exists X. psi` is in positive position if every node on the path from the root to it is
an `AST_Connective` of type `AND` or `OR` or an `AST_Quantifier`. Every bound variable of the formula
has its own `Var` id (`eval.Scoper`), so all such quantifiers can be moved to the front of the formula,
and a formula `exists X. phi` is satisfiable if and only if `phi` with `X` free is satisfiable. The pass
removes the Boolean variables from the bound variables of these quantifiers and removes a quantifier
left without bound variables. Integer variables stay bound.

A lifted variable is marked by `VarInfo.is_lifted_from_toplevel_existential`. Its `is_formula_param`
stays False, so it is not part of a reported model (`parse.py`, `dpllt_automata.py` and
`sat_toplevel.py` build models from the formula parameters only). Two consumers treat it as a free
variable that may be assigned:

| consumer | effect |
|---|---|
| `sat_toplevel.is_assignable_free_bool_var` | the top-level SAT loop (`--use-toplevel-sat`) assigns the variable and substitutes the assignment, including below negated quantifiers |
| `theory_reasoning.scan_variable_use` | the pure-literal rule fixes the variable if all its occurrences have one polarity |

The DPLL(T) driver (`--use-dpllt-automata`) abstracts free Boolean variables only in the conjuncts
without negated quantifiers (`chi` in `dpllt_automata.py`); an occurrence below a negated quantifier stays
a track of the automaton for `phi`.
"""
from __future__ import annotations

from typing import Dict

from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import (
    AST_Connective,
    AST_Quantifier,
    ASTp_Node,
    Connective_Type,
    Var,
    VariableType,
)


def _lift_bool_vars_of_positive_existentials(node: ASTp_Node, var_table: Dict[Var, VarInfo]) -> ASTp_Node:
    match node:
        case AST_Quantifier():
            new_child = _lift_bool_vars_of_positive_existentials(node.child, var_table)

            lifted_vars = [var for var in node.bound_vars if var_table[var].type == VariableType.BOOL]
            for lifted_var in lifted_vars:
                var_table[lifted_var].is_lifted_from_toplevel_existential = True

            remaining_bound_vars = tuple(var for var in node.bound_vars if var not in lifted_vars)
            if not remaining_bound_vars:
                return new_child
            return AST_Quantifier(referenced_vars=node.referenced_vars, bound_vars=remaining_bound_vars, child=new_child)

        case AST_Connective() if node.type in (Connective_Type.AND, Connective_Type.OR):
            new_children = tuple(_lift_bool_vars_of_positive_existentials(child, var_table) for child in node.children)
            return AST_Connective(referenced_vars=node.referenced_vars, type=node.type, children=new_children,
                                  variable_bounds=node.variable_bounds)

        case _:
            return node  # a negation, an EQUIV connective or an atom: quantifiers below are not in positive position


def lift_bool_vars_of_toplevel_existentials(root: ASTp_Node, var_table: Dict[Var, VarInfo]) -> ASTp_Node:
    """
    Return `root` with the Boolean variables of its positive-position existential quantifiers free, and
    mark them in `var_table` (see the module docstring).
    """
    result = _lift_bool_vars_of_positive_existentials(root, var_table)
    fill_referenced_vars(result)
    return result
