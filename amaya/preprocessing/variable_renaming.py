"""
Renaming of variables in a formula: allocation of fresh `Var` ids recorded in the variable table, and
renaming of bound variables apart.

Moved from `amaya/dpllt_automata.py`, which re-exports the names, so that preprocessing passes can use
them without importing the DPLL(T) module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import (
    AST_Connective,
    AST_Negation,
    AST_Quantifier,
    ASTp_Node,
    BoolLiteral,
    Congruence,
    Relation,
    Var,
)


@dataclass
class Fresh_Variable_Allocator:
    """ Hands out `Var` ids not present in `var_table`, recording each one in it. """

    var_table: Dict[Var, VarInfo]
    next_var_id: int

    @staticmethod
    def for_var_table(var_table: Dict[Var, VarInfo]) -> 'Fresh_Variable_Allocator':
        next_var_id = max((var.id for var in var_table), default=0) + 1
        return Fresh_Variable_Allocator(var_table=var_table, next_var_id=next_var_id)

    def allocate_copy_of_variable(self, original_var: Var) -> Var:
        fresh_var = Var(id=self.next_var_id)
        self.next_var_id += 1

        original_var_info = self.var_table.get(original_var)
        if original_var_info is None:
            raise ValueError(f'Cannot rename a variable that is not present in the var table: {original_var}')

        self.var_table[fresh_var] = VarInfo(name=f'{original_var_info.name}#{fresh_var.id}',
                                            type=original_var_info.type,
                                            comment=original_var_info.comment,
                                            is_formula_param=False)
        return fresh_var


def _substitute_vars_in_atom_terms(atom: Relation | Congruence,
                                   substitution: Dict[Var, Var]) -> Relation | Congruence:
    """
    Rebuild `atom` with its variables renamed, keeping its linear terms sorted by variable.

    Never mutates `atom`: `Relation` and `Congruence` are mutable dataclasses shared with the tree the
    caller was handed. The re-sorting matters because fresh ids are larger than the ones they replace,
    so a plain substitution would leave the term list unsorted and break
    `Relation.is_in_canoical_form`, which several passes and constructions rely on.
    """
    renamed_terms = sorted((substitution.get(var, var), coef) for coef, var in zip(atom.coefs, atom.vars))
    renamed_vars = [var for var, _ in renamed_terms]
    renamed_coefs = [coef for _, coef in renamed_terms]

    if isinstance(atom, Relation):
        return Relation(vars=renamed_vars, coefs=renamed_coefs, rhs=atom.rhs,
                        predicate_symbol=atom.predicate_symbol)
    return Congruence(vars=renamed_vars, coefs=renamed_coefs, rhs=atom.rhs, modulus=atom.modulus)


def _freshen_bound_variables(node: ASTp_Node,
                             substitution: Dict[Var, Var],
                             allocator: Fresh_Variable_Allocator) -> ASTp_Node:
    """ `freshen_bound_variables_in_subformula`'s recursion; returns `node` itself when nothing changes. """
    match node:
        case Var():
            return substitution.get(node, node)

        case BoolLiteral():
            return node

        case Relation() | Congruence():
            if not any(var in substitution for var in node.vars):
                return node
            return _substitute_vars_in_atom_terms(node, substitution)

        case AST_Negation():
            renamed_child = _freshen_bound_variables(node.child, substitution, allocator)
            if renamed_child is node.child:
                return node
            return AST_Negation(referenced_vars=node.referenced_vars, child=renamed_child)

        case AST_Connective():
            renamed_children = tuple(_freshen_bound_variables(child, substitution, allocator)
                                     for child in node.children)
            if all(renamed is original for renamed, original in zip(renamed_children, node.children)):
                return node
            return AST_Connective(referenced_vars=node.referenced_vars, type=node.type,
                                  children=renamed_children)

        case AST_Quantifier():
            # A binder always allocates, so that two `AST_Quantifier` nodes sharing a `Var` id (which
            # miniscoping can produce) end up binding different variables. The extended substitution
            # shadows the outer one for the duration of the subtree, as scoping requires.
            substitution_inside_binder = dict(substitution)
            fresh_bound_vars: List[Var] = []
            for bound_var in node.bound_vars:
                fresh_bound_var = allocator.allocate_copy_of_variable(bound_var)
                substitution_inside_binder[bound_var] = fresh_bound_var
                fresh_bound_vars.append(fresh_bound_var)

            renamed_child = _freshen_bound_variables(node.child, substitution_inside_binder, allocator)
            return AST_Quantifier(referenced_vars=node.referenced_vars,
                                  bound_vars=tuple(fresh_bound_vars),
                                  child=renamed_child)

    raise ValueError(f'Unhandled formula node while renaming bound variables apart: {type(node)}')


def freshen_bound_variables_in_subformula(node: ASTp_Node, var_table: Dict[Var, VarInfo]) -> ASTp_Node:
    """
    Return `node` with every binder rebound to freshly allocated variables, extending `var_table` with
    one entry per allocated variable (`is_formula_param=False`, type copied from the original).

    Needed because `build_assertion_formula_for_atom_ids` hoists the binders of the asserted literals
    into a single existential prefix. `Scoper` (`amaya/preprocessing/eval.py`) gives every binder a
    globally fresh id, but nothing re-establishes that after `optimize_formula_structure`: pushing a
    quantifier into a disjunction duplicates the binder, leaving two `AST_Quantifier` nodes that bind
    the same `Var` id and mean different variables. Hoisting those into one prefix would force them to
    take the same value.
    """
    allocator = Fresh_Variable_Allocator.for_var_table(var_table)
    freshened_node = _freshen_bound_variables(node, {}, allocator)
    fill_referenced_vars(freshened_node)
    return freshened_node


def substitute_free_occurrences_of_vars(node: ASTp_Node, substitution: Dict[Var, Var]) -> ASTp_Node:
    """
    Return `node` with every free occurrence of a variable `v` in `substitution` replaced by
    `substitution[v]`. An occurrence below a quantifier binding `v` is bound there and is kept.
    Returns `node` itself when nothing changes. Does not update `referenced_vars`.
    """
    if not substitution:
        return node

    match node:
        case Var():
            return substitution.get(node, node)

        case BoolLiteral():
            return node

        case Relation() | Congruence():
            if not any(var in substitution for var in node.vars):
                return node
            return _substitute_vars_in_atom_terms(node, substitution)

        case AST_Negation():
            substituted_child = substitute_free_occurrences_of_vars(node.child, substitution)
            if substituted_child is node.child:
                return node
            return AST_Negation(referenced_vars=node.referenced_vars, child=substituted_child)

        case AST_Connective():
            substituted_children = tuple(substitute_free_occurrences_of_vars(child, substitution)
                                         for child in node.children)
            if all(substituted is original for substituted, original in zip(substituted_children, node.children)):
                return node
            return AST_Connective(referenced_vars=node.referenced_vars, type=node.type,
                                  children=substituted_children, variable_bounds=node.variable_bounds)

        case AST_Quantifier():
            substitution_inside_binder = {var: replacement for var, replacement in substitution.items()
                                          if var not in node.bound_vars}
            substituted_child = substitute_free_occurrences_of_vars(node.child, substitution_inside_binder)
            if substituted_child is node.child:
                return node
            return AST_Quantifier(referenced_vars=node.referenced_vars, bound_vars=node.bound_vars,
                                  child=substituted_child)

    raise ValueError(f'Unhandled formula node while substituting variables: {type(node)}')
