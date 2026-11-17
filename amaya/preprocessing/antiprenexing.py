from collections import defaultdict
import functools

from typing import Dict, Iterable, List, Optional, Set
from amaya.preprocessing.conditional_equality_resolution import fill_referenced_vars
from amaya.preprocessing.eval import VarInfo
from amaya.preprocessing.variable_renaming import Fresh_Variable_Allocator, substitute_free_occurrences_of_vars
from amaya.relations_structures import (
    AST_Connective,
    AST_Negation,
    AST_Quantifier,
    ASTp_Leaf_Type_List,
    ASTp_Node,
    BoolLiteral,
    Congruence,
    Connective_Type,
    Relation,
    Var,
    pprint_formula
)


def get_referenced_vars(ast: ASTp_Node) -> Iterable[Var]:
    match ast:
        case Var():
            return (ast,)
        case Relation() | Congruence():
            return ast.vars
        case BoolLiteral():
            return tuple()
        case _:
            return ast.referenced_vars


def _perform_miniscoping(quantif_node: AST_Quantifier, var_table: Optional[Dict[Var, VarInfo]] = None) -> ASTp_Node:
    if isinstance(quantif_node.child, AST_Negation):
        # The quantifier cannot move below the negation, but quantifiers inside the negation can be miniscoped.
        return AST_Quantifier(referenced_vars=quantif_node.referenced_vars,
                              bound_vars=quantif_node.bound_vars,
                              child=miniscope_quantifiers(quantif_node.child, var_table))

    if isinstance(quantif_node.child, AST_Connective) and quantif_node.child.type == Connective_Type.EQUIV:
        # @Incomplete: we would be passing a quantifier through a negation if we tried to push the quantifier
        #              inside, or?
        return quantif_node

    if isinstance(quantif_node.child, AST_Quantifier):
        bound_vars = sorted(set(quantif_node.bound_vars).union(quantif_node.child.bound_vars))
        merged_quantifier = AST_Quantifier(referenced_vars=quantif_node.referenced_vars, bound_vars=tuple(bound_vars), child=quantif_node.child.child)
        return _perform_miniscoping(merged_quantifier, var_table)

    if isinstance(quantif_node.child, ASTp_Leaf_Type_List):
        return quantif_node

    connective: AST_Connective = quantif_node.child

    if connective.type == Connective_Type.OR:
        return _distribute_quantifier_over_disjunction(quantif_node.bound_vars, connective, var_table)

    bound_vars = set(quantif_node.bound_vars)

    scope_range = list(miniscope_quantifiers(child, var_table) for child in connective.children)

    min_var_covers_entire_scope = False
    while not min_var_covers_entire_scope and bound_vars:
        # Select var with min scope
        var_to_child_indices: Dict[Var, List[int]] = defaultdict(list)
        for child_idx, child in enumerate(scope_range):
            for var in get_referenced_vars(child):
                if var not in bound_vars:
                    continue
                var_to_child_indices[var].append(child_idx)

        if not var_to_child_indices:
            break

        min_scope_var = min(var_to_child_indices, key=lambda var: (len(var_to_child_indices[var]), var))

        if len(var_to_child_indices[min_scope_var]) == len(scope_range):
            min_var_covers_entire_scope = True
            break

        # Make a new node capturing the min scope
        min_scope = tuple(scope_range[child_idx] for child_idx in var_to_child_indices[min_scope_var])
        _referenced_vars: Set[Var] = functools.reduce(set.union, tuple(get_referenced_vars(child) for child in min_scope), set())
        referenced_vars = tuple(sorted(_referenced_vars))

        if len(min_scope) > 1:
            miniscoped_subtree = AST_Connective(referenced_vars=referenced_vars,
                                                type=connective.type,
                                                children=min_scope,
                                                variable_bounds=connective.variable_bounds)
        else:
            miniscoped_subtree = min_scope[0]

        miniscoped_node = AST_Quantifier(referenced_vars=referenced_vars,
                                         bound_vars=(min_scope_var,),
                                         child=miniscoped_subtree)

        # Modify tree by removing the min_scope nodes that were put into one new node
        untouched_subtrees = [child for child_idx, child in enumerate(scope_range) if child_idx not in var_to_child_indices[min_scope_var]]
        scope_range = untouched_subtrees + [miniscoped_node]

        bound_vars.remove(min_scope_var)

    if len(scope_range) > 1:
        scope = AST_Connective(referenced_vars=quantif_node.referenced_vars, type=connective.type, children=tuple(scope_range))
        if not bound_vars:
            return scope
        result = AST_Quantifier(referenced_vars=quantif_node.referenced_vars, bound_vars=tuple(sorted(bound_vars)), child=scope)
        return result

    # Scope size is 1
    return scope_range[0]


def _distribute_quantifier_over_disjunction(bound_vars: Iterable[Var], disjunction: AST_Connective,
                                            var_table: Optional[Dict[Var, VarInfo]]) -> ASTp_Node:
    """
    Rewrite `∃X. (D_1 ∨ ... ∨ D_n)` into `(∃X_1. D_1) ∨ ... ∨ (∃X_n. D_n)`, where `X_i` are the variables
    of `X` referenced in `D_i`. The rewrite is an equivalence for any `D_i`.

    Each `∃X_i. D_i` is miniscoped further, so a variable that occurs in every disjunct ends up bound
    separately in each disjunct, where passes working on one quantifier body (e.g. the substitution of
    defining equations in `theory_reasoning.simplify_formula_using_model_properties`) can eliminate it.
    The first disjunct referencing a variable keeps its `Var` id; every further disjunct binds a fresh
    copy allocated in `var_table`, so that each `Var` id is bound by at most one quantifier, as `Scoper`
    (`amaya/preprocessing/eval.py`) establishes and other passes assume (e.g. the monotonicity analysis in
    `unbound_vars._determine_monotonicity_of_variables`). If `var_table` is None, the copies keep the
    original `Var` id.
    """
    bound_var_set = set(bound_vars)
    vars_bound_in_an_earlier_disjunct: Set[Var] = set()
    fresh_variable_allocator = Fresh_Variable_Allocator.for_var_table(var_table) if var_table is not None else None

    new_disjuncts: List[ASTp_Node] = []
    for disjunct in disjunction.children:
        disjunct_referenced_vars = tuple(sorted(set(get_referenced_vars(disjunct))))
        disjunct_bound_vars = tuple(var for var in disjunct_referenced_vars if var in bound_var_set)

        if not disjunct_bound_vars:
            new_disjuncts.append(miniscope_quantifiers(disjunct, var_table))
            continue

        if fresh_variable_allocator is not None:
            renaming = {var: fresh_variable_allocator.allocate_copy_of_variable(var)
                        for var in disjunct_bound_vars if var in vars_bound_in_an_earlier_disjunct}
            vars_bound_in_an_earlier_disjunct.update(disjunct_bound_vars)
            if renaming:
                disjunct = substitute_free_occurrences_of_vars(disjunct, renaming)
                fill_referenced_vars(disjunct)  # `_perform_miniscoping` reads the annotation of the children
                disjunct_referenced_vars = tuple(sorted(renaming.get(var, var) for var in disjunct_referenced_vars))
                disjunct_bound_vars = tuple(sorted(renaming.get(var, var) for var in disjunct_bound_vars))

        quantified_disjunct = AST_Quantifier(referenced_vars=disjunct_referenced_vars,
                                             bound_vars=disjunct_bound_vars,
                                             child=disjunct)
        new_disjuncts.append(_perform_miniscoping(quantified_disjunct, var_table))

    referenced_vars = functools.reduce(set.union, (set(get_referenced_vars(disjunct)) for disjunct in new_disjuncts), set())
    return AST_Connective(referenced_vars=tuple(sorted(referenced_vars)),
                          type=Connective_Type.OR,
                          children=tuple(new_disjuncts),
                          variable_bounds=disjunction.variable_bounds)


def miniscope_quantifiers(ast: ASTp_Node, var_table: Optional[Dict[Var, VarInfo]] = None):
    """
    Move quantifiers towards the atoms. `var_table` receives the fresh variables allocated when a quantifier
    is distributed over a disjunction (see `_distribute_quantifier_over_disjunction`).
    """
    match ast:
        case Var() | Congruence() | Relation() | BoolLiteral():
            return ast
        case AST_Quantifier():
            result = _perform_miniscoping(ast, var_table)
            return result
        case AST_Connective():
            children = tuple(miniscope_quantifiers(child, var_table) for child in ast.children)
            return AST_Connective(referenced_vars=ast.referenced_vars, type=ast.type, children=children)
        case AST_Negation():
            child = miniscope_quantifiers(ast.child, var_table)
            return AST_Negation(referenced_vars=ast.referenced_vars, child=child)
        case _:
            raise NotImplementedError(f'Node {ast} is not handled when miniscoping quantifiers.')
