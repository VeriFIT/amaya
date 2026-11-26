"""
Renaming of variables in a formula: allocation of fresh `Var` ids recorded in the variable table, and
renaming of bound variables apart.

Moved from `amaya/dpllt_automata.py`, which re-exports the names, so that preprocessing passes can use
them without importing the DPLL(T) module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

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


@dataclass
class _Crossed_Binder_Stack:
    """
    Substitutions introduced by the quantifiers crossed on the path from the root to the current position
    of a depth-first traversal, outermost first.

    Each crossed quantifier pushes one substitution mapping the variables it binds to their replacements
    and pops it when the traversal leaves the quantifier. A variable is renamed by the substitution of the
    innermost crossed quantifier that binds it (`find_replacement` searches the stack from the top).
    """

    substitutions: List[Dict[Var, Var]]

    def push_binder(self, bound_var_replacements: Dict[Var, Var]) -> None:
        self.substitutions.append(bound_var_replacements)

    def pop_binder(self) -> None:
        self.substitutions.pop()

    def find_replacement(self, var: Var) -> Var:
        """ Returns `var` itself when no crossed quantifier binds it (`var` is free at the current position). """
        for substitution in reversed(self.substitutions):
            replacement_var = substitution.get(var)
            if replacement_var is not None:
                return replacement_var
        return var


def _make_bound_variables_unique(node: ASTp_Node,
                                 crossed_binders: _Crossed_Binder_Stack,
                                 already_bound_vars: Set[Var],
                                 allocator: Fresh_Variable_Allocator) -> ASTp_Node:
    """ `make_bound_variables_unique`'s recursion; returns `node` itself when nothing changes. """
    match node:
        case Var():
            return crossed_binders.find_replacement(node)

        case BoolLiteral():
            return node

        case Relation() | Congruence():
            substitution_for_atom: Dict[Var, Var] = {}
            for var in node.vars:
                replacement_var = crossed_binders.find_replacement(var)
                if replacement_var != var:
                    substitution_for_atom[var] = replacement_var
            if not substitution_for_atom:
                return node
            return _substitute_vars_in_atom_terms(node, substitution_for_atom)

        case AST_Negation():
            renamed_child = _make_bound_variables_unique(node.child, crossed_binders, already_bound_vars, allocator)
            if renamed_child is node.child:
                return node
            return AST_Negation(referenced_vars=node.referenced_vars, child=renamed_child)

        case AST_Connective():
            renamed_children = tuple(
                _make_bound_variables_unique(child, crossed_binders, already_bound_vars, allocator)
                for child in node.children
            )
            if all(renamed is original for renamed, original in zip(renamed_children, node.children)):
                return node
            return AST_Connective(referenced_vars=node.referenced_vars, type=node.type, children=renamed_children)

        case AST_Quantifier():
            bound_var_replacements: Dict[Var, Var] = {}
            for bound_var in node.bound_vars:
                if bound_var in already_bound_vars:
                    replacement_var = allocator.allocate_copy_of_variable(bound_var)
                else:
                    replacement_var = bound_var
                already_bound_vars.add(replacement_var)
                # Also recorded when the variable is kept: it shadows a renaming of the same variable made
                # by an enclosing quantifier.
                bound_var_replacements[bound_var] = replacement_var

            crossed_binders.push_binder(bound_var_replacements)
            renamed_child = _make_bound_variables_unique(node.child, crossed_binders, already_bound_vars, allocator)
            crossed_binders.pop_binder()

            new_bound_vars = tuple(bound_var_replacements[bound_var] for bound_var in node.bound_vars)
            if renamed_child is node.child and new_bound_vars == tuple(node.bound_vars):
                return node
            return AST_Quantifier(referenced_vars=node.referenced_vars, bound_vars=new_bound_vars, child=renamed_child)

    raise ValueError(f'Unhandled formula node while making bound variables unique: {type(node)}')


def make_bound_variables_unique(node: ASTp_Node, var_table: Dict[Var, VarInfo]) -> ASTp_Node:
    """
    Return `node` in which no two `AST_Quantifier` nodes bind the same variable.

    The traversal is depth-first, left to right. The first quantifier binding a variable keeps it; every
    later one, nested or not, binds a fresh variable recorded in `var_table`. Occurrences are renamed
    according to the innermost crossed binder (`_Crossed_Binder_Stack`), so scoping is preserved and the
    result is equivalent to `node`. Returns `node` itself when no binder is renamed.
    """
    allocator = Fresh_Variable_Allocator.for_var_table(var_table)
    crossed_binders = _Crossed_Binder_Stack(substitutions=[])
    renamed_node = _make_bound_variables_unique(node, crossed_binders, set(), allocator)
    if renamed_node is not node:
        fill_referenced_vars(renamed_node)
    return renamed_node
