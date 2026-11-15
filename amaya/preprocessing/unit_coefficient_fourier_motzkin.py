"""
Fourier-Motzkin elimination of existentially quantified integer variables whose every occurrence
has coefficient +1 or -1 (unit-coefficient Fourier-Motzkin, UCFM).

Let `v` be an existentially quantified integer variable and let the body of its quantifier be a
conjunction in which every conjunct mentioning `v` is a linear atom with coefficient +1 or -1 at `v`:

    exists v. (and  L_1 <= v ... L_p <= v   v <= U_1 ... v <= U_q   rest)       (v not in rest)

Every `L_i` and `U_j` is a linear term with integer coefficients, so it takes integer values, and an
integer `v` with `max_i L_i <= v <= min_j U_j` exists if and only if `L_i <= U_j` holds for every pair
`(i, j)`. The rewrite

    exists v. (...)   --->   (and  L_i <= U_j for every (i, j)   rest)

is therefore an equivalence over the integers. If `p = 0` or `q = 0`, no pair exists and every conjunct
mentioning `v` is removed. An equation `v + T = k` contributes both `v <= k - T` and `k - T <= v`.

Bounds in a nested existential quantifier. A conjunct of the body may be an existential quantifier
`exists Y. B` in whose body `v` occurs:

    exists v. (and  A_1 ... A_n   exists Y. B)
      ==  exists Y. exists v. (and  A_1 ... A_n  B)                        (Y not free in any A_i)
      ==  (and  A_rest   pairs(A, A)   exists Y. (and  B_rest   pairs(A, B)   pairs(B, B)))

where `A_rest`, `B_rest` are the conjuncts not mentioning `v`, and `pairs(S, T)` are the combined atoms
of a bound from `S` and a bound from `T` in either role. A combined atom is placed outside `exists Y`
only if both of its bounds come from the `A_i`; such an atom does not mention `Y`. The quantifier
structure is kept: `exists v` is removed and `exists Y` stays in place with a modified body, on which
`_eliminate_vars_of_quantifier` is run again (the pass is bottom-up, so `exists Y` was processed before
its body changed). Side conditions: `v` is not in `Y`, no variable of `Y` occurs in another conjunct of
the body, and `v` occurs in at most one such quantifier (bounds from two sibling quantifiers would
require merging them). The occurrences of `v` in `B` are subject to the same restrictions as those in
the `A_i`.

Occurrence shapes that abort the rewrite for `v` (the formula is left unchanged for `v`):
    - a coefficient of `v` other than +1, -1;
    - `v` in a congruence, in a negated equation, or in a nested connective or negation;
    - `v` in a nested quantifier that violates the side conditions above, or in a quantifier nested inside it;
    - more than `|L| + |U|` atoms produced, i.e. `|L| * |U| > |L| + |U|` (see `_produces_no_more_atoms`).
      The bounds from a nested quantifier are counted with the others.

When the solution domain is the natural numbers, `v >= 0` is added to the lower bounds of `v`.

Every combined atom `sum a_i.x_i <= k` with `g = gcd(a_i) > 1` is replaced by
`sum (a_i / g).x_i <= floor(k / g)`, which is an equivalence for integer-valued `x_i` (both solution
domains). The division can give another variable of the same quantifier a unit coefficient, making it
eliminable in the next round of `_eliminate_vars_of_quantifier`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from amaya.preprocessing.conditional_equality_resolution import (
    _extract_referenced_vars,
    _make_and,
    _referenced_vars_of_children,
)
from amaya.preprocessing.eval import VarInfo
from amaya.preprocessing.inner_quantifier_squeeze_elimination import (
    _fold_relation,
    _linear_combo,
    _references_squeezed_var as _references_var,
)
from amaya.relations_structures import (
    ASTp_Node,
    AST_Connective,
    AST_Negation,
    AST_Quantifier,
    BoolLiteral,
    Congruence,
    Connective_Type,
    Relation,
    Var,
    VariableType,
)


@dataclass
class Unit_Coefficient_Bound:
    """
    One bound on the eliminated variable `v`, stored as the terms and right-hand side of the atom
    `coef_of_v * v + sum(terms) <= rhs` with `v` removed. `coef_of_v` is -1 for a lower bound and +1
    for an upper bound.
    """
    terms: Dict[Var, int]
    rhs: int
    is_from_nested_quantifier: bool = False
    """ The atom is a conjunct of the body of the nested quantifier (see the module docstring). """


@dataclass
class Bounds_On_Eliminated_Var:
    lower_bounds: List[Unit_Coefficient_Bound] = field(default_factory=list)
    upper_bounds: List[Unit_Coefficient_Bound] = field(default_factory=list)


def _terms_without_var(relation: Relation, eliminated_var: Var) -> Dict[Var, int]:
    return {var: coef for var, coef in zip(relation.vars, relation.coefs) if var != eliminated_var}


def _coefficient_of_var(relation: Relation, eliminated_var: Var) -> int:
    if eliminated_var not in relation.vars:
        return 0
    return relation.coefs[relation.vars.index(eliminated_var)]


def _classify_conjunct_into_bounds(conjunct: ASTp_Node, eliminated_var: Var, bounds: Bounds_On_Eliminated_Var,
                                   is_from_nested_quantifier: bool = False) -> bool:
    """
    Add the bounds on `eliminated_var` expressed by `conjunct` to `bounds`. Returns False if the shape of
    `conjunct` is not admissible (see the module docstring), in which case `bounds` must be discarded.
    """
    lower_count_before = len(bounds.lower_bounds)
    upper_count_before = len(bounds.upper_bounds)
    is_admissible = _classify_conjunct_into_untagged_bounds(conjunct, eliminated_var, bounds)
    if is_admissible:
        added_bounds = bounds.lower_bounds[lower_count_before:] + bounds.upper_bounds[upper_count_before:]
        for bound in added_bounds:
            bound.is_from_nested_quantifier = is_from_nested_quantifier
    return is_admissible


def _classify_conjunct_into_untagged_bounds(conjunct: ASTp_Node, eliminated_var: Var,
                                            bounds: Bounds_On_Eliminated_Var) -> bool:
    if not isinstance(conjunct, Relation):
        return False

    coef_of_var = _coefficient_of_var(conjunct, eliminated_var)
    if abs(coef_of_var) != 1:
        return False

    terms = _terms_without_var(conjunct, eliminated_var)
    rhs = conjunct.rhs

    if conjunct.predicate_symbol == '<=':
        bound = Unit_Coefficient_Bound(terms=terms, rhs=rhs)
        if coef_of_var == 1:
            bounds.upper_bounds.append(bound)   # v + T <= k
        else:
            bounds.lower_bounds.append(bound)   # -v + T <= k
        return True

    if conjunct.predicate_symbol == '=':
        # c*v + T = k  <=>  (v + c*T <= c*k)  and  (-v - c*T <= -c*k),  for c in {+1, -1}
        terms_with_v_coef_one = {var: coef_of_var * coef for var, coef in terms.items()}
        bounds.upper_bounds.append(Unit_Coefficient_Bound(terms=terms_with_v_coef_one, rhs=coef_of_var * rhs))
        bounds.lower_bounds.append(Unit_Coefficient_Bound(terms={var: -coef for var, coef in terms_with_v_coef_one.items()},
                                                          rhs=-coef_of_var * rhs))
        return True

    return False


def _produces_no_more_atoms(bounds: Bounds_On_Eliminated_Var) -> bool:
    """
    Fourier-Motzkin replaces `|L| + |U|` atoms by `|L| * |U|` atoms. The rewrite is restricted to the
    cases in which the number of atoms does not increase.
    """
    lower_count = len(bounds.lower_bounds)
    upper_count = len(bounds.upper_bounds)
    return lower_count * upper_count <= lower_count + upper_count


def _combine_lower_and_upper_bound(lower_bound: Unit_Coefficient_Bound, upper_bound: Unit_Coefficient_Bound) -> ASTp_Node:
    """
    (-v + T_lower <= k_lower) + (v + T_upper <= k_upper)  =  (T_lower + T_upper <= k_lower + k_upper),
    divided by the gcd of its coefficients (see the module docstring).
    """
    vars_, coefs = _linear_combo(lower_bound.terms, 1, upper_bound.terms, 1)
    rhs = lower_bound.rhs + upper_bound.rhs
    if coefs:
        coef_gcd = math.gcd(*coefs)
        if coef_gcd > 1:
            coefs = [coef // coef_gcd for coef in coefs]
            rhs = rhs // coef_gcd  # floor, exact for integers of any size
    combined = Relation(vars=vars_, coefs=coefs, rhs=rhs, predicate_symbol='<=')
    return _fold_relation(combined)


def _conjuncts_of(node: ASTp_Node) -> Tuple[ASTp_Node, ...]:
    if isinstance(node, AST_Connective) and node.type == Connective_Type.AND:
        return node.children
    return (node,)


def _is_nested_quantifier_admissible(nested_quantifier: AST_Quantifier, eliminated_var: Var,
                                     other_conjuncts: List[ASTp_Node]) -> bool:
    """
    The side conditions under which the bounds on `eliminated_var` in the body of `nested_quantifier`
    may be used (see the module docstring): `eliminated_var` is not rebound, and no variable bound by
    `nested_quantifier` occurs in the other conjuncts of the enclosing body.
    """
    if eliminated_var in nested_quantifier.bound_vars:
        return False
    for conjunct in other_conjuncts:
        if any(_references_var(conjunct, nested_bound_var) for nested_bound_var in nested_quantifier.bound_vars):
            return False
    return True


def _make_quantifier(bound_vars: Tuple[Var, ...], body: ASTp_Node) -> ASTp_Node:
    if not bound_vars or isinstance(body, BoolLiteral):
        return body
    return AST_Quantifier(referenced_vars=_extract_referenced_vars(body), bound_vars=bound_vars, child=body)


def _try_eliminate_var_by_fourier_motzkin(eliminated_var: Var, body: ASTp_Node, var_table: Dict[Var, VarInfo],
                                          quantified_vars_are_nonnegative: bool) -> Optional[ASTp_Node]:
    """
    Returns the body with `eliminated_var` eliminated, or None if some occurrence of `eliminated_var` is
    not admissible or the rewrite would increase the number of atoms.

    The bounds may come from the conjuncts of `body` and from the top-level conjuncts of the body of at
    most one existential quantifier that is itself a conjunct of `body`. Combined atoms with a bound from
    that quantifier are placed in its body, which is then processed by `_eliminate_vars_of_quantifier`
    again (see the module docstring).
    """
    conjuncts = _conjuncts_of(body)

    bounds = Bounds_On_Eliminated_Var()
    if quantified_vars_are_nonnegative:
        bounds.lower_bounds.append(Unit_Coefficient_Bound(terms={}, rhs=0))  # -v <= 0

    retained_conjuncts: List[ASTp_Node] = []
    nested_quantifier: Optional[AST_Quantifier] = None
    retained_conjuncts_of_nested_body: List[ASTp_Node] = []

    for conjunct_idx, conjunct in enumerate(conjuncts):
        if not _references_var(conjunct, eliminated_var):
            retained_conjuncts.append(conjunct)
            continue

        if isinstance(conjunct, AST_Quantifier):
            other_conjuncts = [other for other_idx, other in enumerate(conjuncts) if other_idx != conjunct_idx]
            if nested_quantifier is not None:
                return None  # Bounds in two sibling quantifiers would require merging them
            if not _is_nested_quantifier_admissible(conjunct, eliminated_var, other_conjuncts):
                return None
            nested_quantifier = conjunct
            for nested_conjunct in _conjuncts_of(conjunct.child):
                if not _references_var(nested_conjunct, eliminated_var):
                    retained_conjuncts_of_nested_body.append(nested_conjunct)
                    continue
                if not _classify_conjunct_into_bounds(nested_conjunct, eliminated_var, bounds,
                                                      is_from_nested_quantifier=True):
                    return None
            continue

        if not _classify_conjunct_into_bounds(conjunct, eliminated_var, bounds):
            return None

    if not _produces_no_more_atoms(bounds):
        return None

    for lower_bound in bounds.lower_bounds:
        for upper_bound in bounds.upper_bounds:
            combined_atom = _combine_lower_and_upper_bound(lower_bound, upper_bound)
            if lower_bound.is_from_nested_quantifier or upper_bound.is_from_nested_quantifier:
                retained_conjuncts_of_nested_body.append(combined_atom)
            else:
                retained_conjuncts.append(combined_atom)

    if nested_quantifier is not None:
        new_nested_body, remaining_nested_bound_vars = _eliminate_vars_of_quantifier(
            nested_quantifier.bound_vars, _make_and(retained_conjuncts_of_nested_body), var_table,
            quantified_vars_are_nonnegative)
        retained_conjuncts.append(_make_quantifier(remaining_nested_bound_vars, new_nested_body))

    return _make_and(retained_conjuncts)


def _eliminate_vars_of_quantifier(bound_vars: Tuple[Var, ...], body: ASTp_Node, var_table: Dict[Var, VarInfo],
                                  quantified_vars_are_nonnegative: bool) -> Tuple[ASTp_Node, Tuple[Var, ...]]:
    """
    Eliminates bound variables until none of the remaining ones is admissible: eliminating one variable
    can change the coefficients of another (e.g. `-v + w <= 0` and `v - w + u <= 0` combine into `u <= 0`).
    """
    remaining_bound_vars = list(bound_vars)
    current_body = body

    made_progress = True
    while made_progress and not isinstance(current_body, BoolLiteral):
        made_progress = False
        for var in list(remaining_bound_vars):
            if var_table[var].type != VariableType.INT:
                continue

            rewritten_body = _try_eliminate_var_by_fourier_motzkin(var, current_body, var_table,
                                                                   quantified_vars_are_nonnegative)
            if rewritten_body is None:
                continue

            current_body = rewritten_body
            remaining_bound_vars.remove(var)
            made_progress = True
            if isinstance(current_body, BoolLiteral):
                break

    return current_body, tuple(remaining_bound_vars)


def eliminate_unit_coefficient_vars_by_fourier_motzkin(root: ASTp_Node, var_table: Dict[Var, VarInfo],
                                                       quantified_vars_are_nonnegative: bool = False) -> ASTp_Node:
    """
    Bottom-up pass applying unit-coefficient Fourier-Motzkin elimination to every existential quantifier.
    See the module docstring. `quantified_vars_are_nonnegative` is True when the solution domain is the
    natural numbers.
    """
    match root:
        case Relation() | Congruence() | BoolLiteral() | Var():
            return root

        case AST_Negation():
            new_child = eliminate_unit_coefficient_vars_by_fourier_motzkin(root.child, var_table, quantified_vars_are_nonnegative)
            if isinstance(new_child, BoolLiteral):
                return BoolLiteral(not new_child.value)
            return AST_Negation(referenced_vars=_extract_referenced_vars(new_child), child=new_child)

        case AST_Connective():
            new_children = tuple(
                eliminate_unit_coefficient_vars_by_fourier_motzkin(child, var_table, quantified_vars_are_nonnegative)
                for child in root.children
            )
            # `replace_children` copies `root.referenced_vars`, which still lists the variables eliminated below.
            new_node = AST_Connective(referenced_vars=_referenced_vars_of_children(list(new_children)), type=root.type,
                                      children=new_children, variable_bounds=root.variable_bounds)

            new_node = new_node.simplify_on_anihilators()
            if not isinstance(new_node, AST_Connective):
                return new_node

            return new_node.remove_idempotent_children()

        case AST_Quantifier():
            new_child = eliminate_unit_coefficient_vars_by_fourier_motzkin(root.child, var_table, quantified_vars_are_nonnegative)
            new_child, remaining_bound_vars = _eliminate_vars_of_quantifier(root.bound_vars, new_child, var_table,
                                                                           quantified_vars_are_nonnegative)

            return _make_quantifier(remaining_bound_vars, new_child)

        case _:
            raise ValueError(f'Unhandled node while eliminating unit-coefficient variables by Fourier-Motzkin: {root}')
