"""
(EXPERIMENTAL, disabled by default) Exists-forall solving with automata projection lemmas (CEGQI).
See `docs/CEGQI with automata.md`.

Decides formulae of the shape `exists P. forall X. AND_i C_i(P_i, X)`, where `P = B + K` are Bool
parameters `B` and Int parameters `K`, `X` are Int universal variables and every `C_i` is quantifier
free. The formula reaches this module after preprocessing, where a universal quantifier is
`NOT exists X. NOT C`, so the module works with the negated clauses `N_i = NOT C_i` throughout.

The loop:

1. Synthesis (`Candidate_Synthesizer`): a SAT solver over `B` chooses `B*`; the lemmas active under
   `B*` constrain `K`; `K*` is read off the product of the complements of their `Bad` automata.
   An empty product blocks the union of the active lemmas' guard literals in the SAT solver.
2. Verification (`find_counterexample_to_candidate`): substitute `(B*, K*)` into every `N_i` and
   decide `exists X. N_i(B*, K*, X)` per disjunct of its DNF with the early-stopping lazy
   construction. No counterexample: the formula is SAT with model `(B*, K*)`.
3. Generalization (`generalize_counterexample_into_lemma`): from the counterexample point
   `(B*, K*, X*)` extract an implicant of `N_i`: a set of Bool literals `gamma` true under `B*` and a set
   of atoms `A` true at `(K*, X*)` with `gamma AND A -> N_i`. The lemma is
   `gamma -> K_i not in Bad`, `Bad = { K_i | exists X. AND A }`, with `Bad` built by projecting the `X`
   tracks of the atom-conjunction automaton and then determinized and minimized.

Implemented subset of the design document (status at the point of description):

| Design element | Status |
| --- | --- |
| Shape detection, `max |P_i| / |P|` | implemented |
| Verification with early stop | implemented; bound pre-check not implemented |
| `disjunct` generalization | implemented, with the implicant chosen at the counterexample point |
| `point` generalization, alone or with `disjunct` | implemented |
| `clause` generalization | not implemented |
| State budget with point fallback | not implemented |
| Synthesis by eager product of complemented `Bad` automata | implemented; lazy product not implemented |
| Bound filter in synthesis, core minimization | not implemented; the blocking clause is the union of the guards of the lemmas intersected before the product became empty |
| Forward ground seeding, from the zero state and from counterexamples | implemented for states named `x_<d>` / `y_<d>` (`Forward_Ground_Seeder`); off by default |
| Breadth-first (shortest) model search | not implemented |
| Bool universal variables | not supported; such formulae fall through |
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from amaya import logger
from amaya import parse
from amaya.config import BackendType, SolutionDomain, solver_config
from amaya.mtbdd_automatons import MTBDD_NFA
from amaya.parse import (
    Evaluation_Result,
    Quantified_Atom_Conjunction,
    convert_binary_model_into_decadic,
    try_decide_atom_conjunction_with_early_stop,
)
from amaya.preprocessing.eval import VarInfo
from amaya.relations_structures import (
    AST_Connective,
    AST_Negation,
    AST_Quantifier,
    ASTp_Node,
    BoolLiteral,
    Congruence,
    Connective_Type,
    Relation,
    Var,
    VariableType,
)
from amaya.solver_core import EvaluationContext


class Unsupported_Formula_Shape(Exception):
    """ The formula is outside the class this strategy decides; the caller falls through. """


MAX_MODULUS_FOR_NEGATED_CONGRUENCE_EXPANSION = 64
""" `NOT (t = r mod m)` is expanded into the disjunction of the other `m - 1` residues up to this `m`. """

MAX_DISJUNCTS_IN_VERIFICATION_DNF = 4096
""" Verification expands a substituted negated clause into DNF only up to this many disjuncts. """


# ---------------------------------------------------------------------------------------------
# Node construction helpers
# ---------------------------------------------------------------------------------------------

def collect_vars_of_node(node: ASTp_Node) -> Set[Var]:
    """ Variables occurring in `node`, recomputed from the tree (bound variables included). """
    match node:
        case Relation() | Congruence():
            return set(node.vars)
        case Var():
            return {node}
        case BoolLiteral():
            return set()
        case AST_Negation():
            return collect_vars_of_node(node.child)
        case AST_Quantifier():
            return collect_vars_of_node(node.child) | set(node.bound_vars)
        case AST_Connective():
            vars_of_children: Set[Var] = set()
            for child in node.children:
                vars_of_children |= collect_vars_of_node(child)
            return vars_of_children
    raise ValueError(f'Unhandled node type: {type(node)}')


def make_connective_with_folding(connective_type: Connective_Type, children: Iterable[ASTp_Node]) -> ASTp_Node:
    """ AND/OR over `children` with Bool literals folded, nested connectives of the same type flattened. """
    assert connective_type in (Connective_Type.AND, Connective_Type.OR)
    annihilator = BoolLiteral(connective_type == Connective_Type.OR)
    identity = BoolLiteral(connective_type == Connective_Type.AND)

    flattened_children: List[ASTp_Node] = []
    for child in children:
        if child == annihilator:
            return annihilator
        if child == identity:
            continue
        if isinstance(child, AST_Connective) and child.type == connective_type:
            flattened_children.extend(child.children)
        else:
            flattened_children.append(child)

    if not flattened_children:
        return identity
    if len(flattened_children) == 1:
        return flattened_children[0]

    referenced_vars: Set[Var] = set()
    for child in flattened_children:
        referenced_vars |= collect_vars_of_node(child)
    return AST_Connective(referenced_vars=tuple(sorted(referenced_vars)), type=connective_type,
                          children=tuple(flattened_children))


def make_negated_bool_var(var: Var) -> AST_Negation:
    return AST_Negation(referenced_vars=(var,), child=var)


def make_relation_with_sorted_vars(vars: Sequence[Var], coefs: Sequence[int], rhs: int, predicate_symbol: str) -> ASTp_Node:
    """ A relation with zero coefficients dropped and variables sorted; folded to a literal if no variable remains. """
    terms = sorted(((var, coef) for var, coef in zip(vars, coefs) if coef != 0), key=lambda term: term[0])
    if not terms:
        if predicate_symbol == '=':
            return BoolLiteral(rhs == 0)
        assert predicate_symbol == '<='
        return BoolLiteral(0 <= rhs)
    return Relation(vars=[var for var, _ in terms], coefs=[coef for _, coef in terms], rhs=rhs,
                    predicate_symbol=predicate_symbol)


def make_congruence_with_sorted_vars(vars: Sequence[Var], coefs: Sequence[int], rhs: int, modulus: int) -> ASTp_Node:
    terms = sorted(((var, coef % modulus) for var, coef in zip(vars, coefs) if coef % modulus != 0),
                   key=lambda term: term[0])
    if not terms:
        return BoolLiteral(rhs % modulus == 0)
    return Congruence(vars=[var for var, _ in terms], coefs=[coef for _, coef in terms], rhs=rhs % modulus,
                      modulus=modulus)


# ---------------------------------------------------------------------------------------------
# Negation normal form
# ---------------------------------------------------------------------------------------------

def negate_relation(relation: Relation) -> ASTp_Node:
    """ `NOT relation` as a `<=` relation, or as a disjunction of two for an equation. """
    if relation.predicate_symbol == '<=':
        return make_relation_with_sorted_vars(relation.vars, [-coef for coef in relation.coefs], -relation.rhs - 1, '<=')
    if relation.predicate_symbol == '=':
        # a.x != b  <=>  a.x <= b - 1  OR  -a.x <= -b - 1
        less_than = make_relation_with_sorted_vars(relation.vars, relation.coefs, relation.rhs - 1, '<=')
        greater_than = make_relation_with_sorted_vars(relation.vars, [-coef for coef in relation.coefs],
                                                      -relation.rhs - 1, '<=')
        return make_connective_with_folding(Connective_Type.OR, (less_than, greater_than))
    raise Unsupported_Formula_Shape(f'Relation with predicate symbol {relation.predicate_symbol}: {relation}')


def negate_congruence(congruence: Congruence) -> ASTp_Node:
    """ `NOT (t = r mod m)` as the disjunction of `t = r' mod m` over the other residues `r'`. """
    if congruence.modulus > MAX_MODULUS_FOR_NEGATED_CONGRUENCE_EXPANSION:
        raise Unsupported_Formula_Shape(f'Negated congruence with modulus {congruence.modulus}')
    other_residues = (residue for residue in range(congruence.modulus) if residue != congruence.rhs % congruence.modulus)
    return make_connective_with_folding(Connective_Type.OR, (
        make_congruence_with_sorted_vars(congruence.vars, congruence.coefs, residue, congruence.modulus)
        for residue in other_residues
    ))


def convert_to_negation_normal_form(node: ASTp_Node, is_negated: bool = False) -> ASTp_Node:
    """
    The quantifier-free `node` (negated if `is_negated`) with negations only directly above Bool
    variables. Negated atoms are rewritten into atoms (`negate_relation`, `negate_congruence`), and
    `EQUIV` is expanded. Raises `Unsupported_Formula_Shape` on a quantifier.
    """
    match node:
        case BoolLiteral():
            return BoolLiteral(node.value != is_negated)
        case Var():
            return make_negated_bool_var(node) if is_negated else node
        case Relation():
            if not is_negated:
                return make_relation_with_sorted_vars(node.vars, node.coefs, node.rhs, node.predicate_symbol)
            return negate_relation(node)
        case Congruence():
            if not is_negated:
                return make_congruence_with_sorted_vars(node.vars, node.coefs, node.rhs, node.modulus)
            return negate_congruence(node)
        case AST_Negation():
            return convert_to_negation_normal_form(node.child, not is_negated)
        case AST_Connective():
            if node.type == Connective_Type.EQUIV:
                if len(node.children) != 2:
                    raise Unsupported_Formula_Shape('EQUIV with more than two children')
                left, right = node.children
                if not is_negated:  # (l AND r) OR (NOT l AND NOT r)
                    both_true = make_connective_with_folding(Connective_Type.AND, (
                        convert_to_negation_normal_form(left), convert_to_negation_normal_form(right)))
                    both_false = make_connective_with_folding(Connective_Type.AND, (
                        convert_to_negation_normal_form(left, True), convert_to_negation_normal_form(right, True)))
                    return make_connective_with_folding(Connective_Type.OR, (both_true, both_false))
                left_only = make_connective_with_folding(Connective_Type.AND, (
                    convert_to_negation_normal_form(left), convert_to_negation_normal_form(right, True)))
                right_only = make_connective_with_folding(Connective_Type.AND, (
                    convert_to_negation_normal_form(left, True), convert_to_negation_normal_form(right)))
                return make_connective_with_folding(Connective_Type.OR, (left_only, right_only))

            connective_type = node.type
            if is_negated:
                connective_type = Connective_Type.OR if node.type == Connective_Type.AND else Connective_Type.AND
            return make_connective_with_folding(connective_type, (
                convert_to_negation_normal_form(child, is_negated) for child in node.children))
        case AST_Quantifier():
            raise Unsupported_Formula_Shape('Quantifier below the universal block')
    raise ValueError(f'Unhandled node type: {type(node)}')


# ---------------------------------------------------------------------------------------------
# Shape detection
# ---------------------------------------------------------------------------------------------

@dataclass
class Negated_Clause:
    """ `N_i = NOT C_i`, in negation normal form; the clause is violated exactly where `N_i` holds. """
    clause_index: int
    negated_clause_in_negation_normal_form: ASTp_Node
    universal_vars: Tuple[Var, ...]
    """ The universal variables occurring in this clause. """
    bool_parameters: FrozenSet[Var]
    int_parameters: FrozenSet[Var]


@dataclass
class Exists_Forall_Shape:
    bool_parameters: Tuple[Var, ...]
    int_parameters: Tuple[Var, ...]
    universal_vars: Tuple[Var, ...]
    negated_clauses: List[Negated_Clause]

    def compute_maximal_clause_parameter_ratio(self) -> float:
        parameter_count = len(self.bool_parameters) + len(self.int_parameters)
        if parameter_count == 0:
            return 0.0
        maximal_clause_parameter_count = max(
            (len(clause.bool_parameters) + len(clause.int_parameters) for clause in self.negated_clauses), default=0)
        return maximal_clause_parameter_count / parameter_count


def _split_into_conjuncts(node: ASTp_Node) -> List[ASTp_Node]:
    if isinstance(node, AST_Connective) and node.type == Connective_Type.AND:
        return [conjunct for child in node.children for conjunct in _split_into_conjuncts(child)]
    return [node]


def _split_into_disjuncts(node: ASTp_Node) -> List[ASTp_Node]:
    if isinstance(node, AST_Connective) and node.type == Connective_Type.OR:
        return [disjunct for child in node.children for disjunct in _split_into_disjuncts(child)]
    return [node]


def _strip_existential_quantifiers(node: ASTp_Node) -> Tuple[List[Var], ASTp_Node]:
    bound_vars: List[Var] = []
    while isinstance(node, AST_Quantifier):
        bound_vars.extend(node.bound_vars)
        node = node.child
    return bound_vars, node


def detect_exists_forall_shape(root: ASTp_Node, var_table: Dict[Var, VarInfo]) -> Exists_Forall_Shape:
    """
    Match `exists P. AND_j G_j`, where every `G_j` is either `NOT exists X_j. D_j` with `D_j` quantifier
    free, or quantifier free itself (a constraint on `P` only, with no universal variables).

    Each disjunct of each `D_j` becomes one negated clause, since `NOT exists X. (D' OR D'')` equals
    `(NOT exists X. D') AND (NOT exists X. D'')`. A quantifier-free `G_j` becomes the negated clause
    `NOT G_j`. Raises `Unsupported_Formula_Shape` for any other formula.
    """
    outer_bound_vars, body = _strip_existential_quantifiers(root)
    universal_vars: Set[Var] = set()
    negated_clause_bodies: List[Tuple[ASTp_Node, Tuple[Var, ...]]] = []

    for conjunct in _split_into_conjuncts(body):
        if isinstance(conjunct, AST_Negation) and isinstance(conjunct.child, AST_Quantifier):
            inner_bound_vars, negated_body = _strip_existential_quantifiers(conjunct.child)
            for var in inner_bound_vars:
                if var_table[var].type != VariableType.INT:
                    raise Unsupported_Formula_Shape(f'Universal variable {var} is not of sort Int')
            universal_vars.update(inner_bound_vars)
            # Negation normal form first: without the -O pipeline the body may still be `NOT AND(...)`.
            for disjunct in _split_into_disjuncts(convert_to_negation_normal_form(negated_body)):
                negated_clause_bodies.append((disjunct, tuple(inner_bound_vars)))
        else:
            negated_clause_bodies.append((convert_to_negation_normal_form(conjunct, is_negated=True), tuple()))

    all_vars_of_clauses: Set[Var] = set()
    negated_clauses: List[Negated_Clause] = []
    for clause_index, (negated_clause_body, clause_universal_vars) in enumerate(negated_clause_bodies):
        clause_vars = collect_vars_of_node(negated_clause_body)
        all_vars_of_clauses |= clause_vars
        clause_parameters = clause_vars - set(clause_universal_vars)
        negated_clauses.append(Negated_Clause(
            clause_index=clause_index,
            negated_clause_in_negation_normal_form=negated_clause_body,
            universal_vars=tuple(sorted(set(clause_universal_vars) & clause_vars)),
            bool_parameters=frozenset(var for var in clause_parameters if var_table[var].type == VariableType.BOOL),
            int_parameters=frozenset(var for var in clause_parameters if var_table[var].type == VariableType.INT),
        ))

    parameters = (set(outer_bound_vars) | (all_vars_of_clauses - universal_vars))
    for var in parameters:
        if var_table[var].type not in (VariableType.BOOL, VariableType.INT):
            raise Unsupported_Formula_Shape(f'Parameter {var} has unsupported sort {var_table[var].type}')

    return Exists_Forall_Shape(
        bool_parameters=tuple(sorted(var for var in parameters if var_table[var].type == VariableType.BOOL)),
        int_parameters=tuple(sorted(var for var in parameters if var_table[var].type == VariableType.INT)),
        universal_vars=tuple(sorted(universal_vars)),
        negated_clauses=negated_clauses,
    )


# ---------------------------------------------------------------------------------------------
# Substitution and evaluation at a point
# ---------------------------------------------------------------------------------------------

def substitute_parameter_values(node: ASTp_Node,
                                bool_values: Dict[Var, bool],
                                int_values: Dict[Var, int]) -> ASTp_Node:
    """
    Substitute constants into a negation-normal-form `node` and fold. Equivalence preserving: the result
    has exactly the models of `node` that agree with the substituted values. Does not mutate `node`.
    """
    match node:
        case BoolLiteral():
            return node
        case Var():
            value = bool_values.get(node)
            return node if value is None else BoolLiteral(value)
        case AST_Negation():
            assert isinstance(node.child, Var), 'Expected negation normal form'
            value = bool_values.get(node.child)
            return node if value is None else BoolLiteral(not value)
        case Relation():
            if not any(var in int_values for var in node.vars):
                return node
            rhs = node.rhs
            remaining_vars: List[Var] = []
            remaining_coefs: List[int] = []
            for coef, var in zip(node.coefs, node.vars):
                if var in int_values:
                    rhs -= coef * int_values[var]
                else:
                    remaining_vars.append(var)
                    remaining_coefs.append(coef)
            return make_relation_with_sorted_vars(remaining_vars, remaining_coefs, rhs, node.predicate_symbol)
        case Congruence():
            if not any(var in int_values for var in node.vars):
                return node
            rhs = node.rhs
            remaining_vars = []
            remaining_coefs = []
            for coef, var in zip(node.coefs, node.vars):
                if var in int_values:
                    rhs -= coef * int_values[var]
                else:
                    remaining_vars.append(var)
                    remaining_coefs.append(coef)
            return make_congruence_with_sorted_vars(remaining_vars, remaining_coefs, rhs, node.modulus)
        case AST_Connective():
            return make_connective_with_folding(node.type, (
                substitute_parameter_values(child, bool_values, int_values) for child in node.children))
    raise ValueError(f'Unhandled node type in negation normal form: {type(node)}')


def evaluate_atom_at_point(atom: Relation | Congruence, int_values: Dict[Var, int]) -> bool:
    left_hand_side_value = sum(coef * int_values[var] for coef, var in zip(atom.coefs, atom.vars))
    if isinstance(atom, Congruence):
        return (left_hand_side_value - atom.rhs) % atom.modulus == 0
    if atom.predicate_symbol == '=':
        return left_hand_side_value == atom.rhs
    assert atom.predicate_symbol == '<='
    return left_hand_side_value <= atom.rhs


def evaluate_negation_normal_form_at_point(node: ASTp_Node,
                                           bool_values: Dict[Var, bool],
                                           int_values: Dict[Var, int]) -> bool:
    match node:
        case BoolLiteral():
            return node.value
        case Var():
            return bool_values[node]
        case AST_Negation():
            return not bool_values[node.child]
        case Relation() | Congruence():
            return evaluate_atom_at_point(node, int_values)
        case AST_Connective():
            child_values = (evaluate_negation_normal_form_at_point(child, bool_values, int_values)
                            for child in node.children)
            return all(child_values) if node.type == Connective_Type.AND else any(child_values)
    raise ValueError(f'Unhandled node type in negation normal form: {type(node)}')


@dataclass(frozen=True)
class Implicant_At_Point:
    """
    `guard_literals AND atoms -> node` holds propositionally (atoms treated as opaque), the guard
    literals hold under the point's Bool values and the atoms hold at the point's Int values.
    """
    guard_literals: FrozenSet[Tuple[Var, bool]]
    atoms: Tuple[Relation | Congruence, ...]


IMPLICANT_PREFERENCE_FEWER_GUARDS = 'fewer-guards'
IMPLICANT_PREFERENCE_FEWER_ATOMS = 'fewer-atoms'
IMPLICANT_PREFERENCES = (IMPLICANT_PREFERENCE_FEWER_GUARDS, IMPLICANT_PREFERENCE_FEWER_ATOMS)


def extract_implicant_at_point(node: ASTp_Node,
                               bool_values: Dict[Var, bool],
                               int_values: Dict[Var, int],
                               preference: str = IMPLICANT_PREFERENCE_FEWER_GUARDS) -> Implicant_At_Point:
    """
    An implicant of the negation-normal-form `node` that holds at the given point, which must satisfy
    `node`. For a disjunction, one true child is chosen:

    | `preference` | Chosen child | Effect on the lemma |
    | --- | --- | --- |
    | `fewer-guards` | fewest guard literals, then fewest atoms | smaller `gamma`: active under more Bool assignments; more atoms: larger `Bad` automata |
    | `fewer-atoms` | fewest atoms, then fewest guard literals | fewer Int parameter tracks per `Bad` automaton; larger `gamma` |
    """
    match node:
        case BoolLiteral():
            assert node.value
            return Implicant_At_Point(guard_literals=frozenset(), atoms=())
        case Var():
            assert bool_values[node]
            return Implicant_At_Point(guard_literals=frozenset({(node, True)}), atoms=())
        case AST_Negation():
            assert not bool_values[node.child]
            return Implicant_At_Point(guard_literals=frozenset({(node.child, False)}), atoms=())
        case Relation() | Congruence():
            assert evaluate_atom_at_point(node, int_values)
            return Implicant_At_Point(guard_literals=frozenset(), atoms=(node,))
        case AST_Connective():
            if node.type == Connective_Type.AND:
                guard_literals: Set[Tuple[Var, bool]] = set()
                atoms: List[Relation | Congruence] = []
                for child in node.children:
                    child_implicant = extract_implicant_at_point(child, bool_values, int_values, preference)
                    guard_literals |= child_implicant.guard_literals
                    atoms.extend(atom for atom in child_implicant.atoms if atom not in atoms)
                return Implicant_At_Point(guard_literals=frozenset(guard_literals), atoms=tuple(atoms))

            implicants_of_true_children = [
                extract_implicant_at_point(child, bool_values, int_values, preference)
                for child in node.children
                if evaluate_negation_normal_form_at_point(child, bool_values, int_values)
            ]
            assert implicants_of_true_children, 'The point does not satisfy the disjunction'
            if preference == IMPLICANT_PREFERENCE_FEWER_ATOMS:
                return min(implicants_of_true_children,
                           key=lambda implicant: (len(implicant.atoms), len(implicant.guard_literals)))
            return min(implicants_of_true_children,
                       key=lambda implicant: (len(implicant.guard_literals), len(implicant.atoms)))
    raise ValueError(f'Unhandled node type in negation normal form: {type(node)}')


def expand_into_atom_conjunctions(node: ASTp_Node, max_disjuncts: int) -> Optional[List[Tuple[Relation | Congruence, ...]]]:
    """
    DNF of a negation-normal-form `node` without Bool variables, as a list of atom conjunctions.
    `[]` is False, `[()]` is True. None if more than `max_disjuncts` disjuncts would be produced.
    """
    match node:
        case BoolLiteral():
            return [()] if node.value else []
        case Relation() | Congruence():
            return [(node,)]
        case AST_Connective():
            child_expansions = []
            for child in node.children:
                child_expansion = expand_into_atom_conjunctions(child, max_disjuncts)
                if child_expansion is None:
                    return None
                child_expansions.append(child_expansion)
            if node.type == Connective_Type.OR:
                result = [conjunction for expansion in child_expansions for conjunction in expansion]
                return result if len(result) <= max_disjuncts else None
            product_size = 1
            for expansion in child_expansions:
                product_size *= len(expansion)
                if product_size > max_disjuncts:
                    return None
            return [tuple(atom for conjunction in combination for atom in conjunction)
                    for combination in itertools.product(*child_expansions)]
    raise Unsupported_Formula_Shape(f'Unexpected node in a substituted negated clause: {node}')


# ---------------------------------------------------------------------------------------------
# Automata helpers
# ---------------------------------------------------------------------------------------------

def _reset_encoding_cache(ctx: EvaluationContext):
    """ `ctx.enc_table` is keyed by `id(node)`; the trees built here are short-lived. See `amaya.sat_toplevel`. """
    if getattr(ctx, 'enc_table', None) is not None:
        ctx.enc_table = {}


def decide_atom_conjunction_and_find_model(atoms: Sequence[Relation | Congruence],
                                           ctx: EvaluationContext) -> Optional[Dict[Var, int]]:
    """ A model of the quantifier-free atom conjunction over its variables, or None if it is unsatisfiable. """
    if not atoms:
        return {}
    conjunction = Quantified_Atom_Conjunction(atoms=tuple(atoms), bound_vars=())
    nfa = try_decide_atom_conjunction_with_early_stop(conjunction, ctx)
    if nfa is None:
        _reset_encoding_cache(ctx)
        nfa = parse.run_evaluation_procedure(make_connective_with_folding(Connective_Type.AND, atoms), ctx)
    binary_model = nfa.find_model()
    if binary_model is None:
        return None
    conjunction_vars = tuple(sorted(set(var for atom in atoms for var in atom.vars)))
    return convert_binary_model_into_decadic(binary_model, tuple(nfa.used_variables), conjunction_vars)


def make_minimal_deterministic_automaton(nfa: MTBDD_NFA) -> MTBDD_NFA:
    if not (nfa.automaton_type & parse.AutomatonType.DFA):
        nfa = nfa.determinize()
    return nfa.minimize_hopcroft()


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------

@dataclass
class Counterexample:
    negated_clause: Negated_Clause
    universal_values: Dict[Var, int]


def find_counterexample_for_clause(negated_clause: Negated_Clause,
                                   bool_values: Dict[Var, bool],
                                   int_values: Dict[Var, int],
                                   ctx: EvaluationContext,
                                   statistics: Cegqi_Statistics) -> Optional[Dict[Var, int]]:
    """ Values of the clause's universal variables violating it under the candidate, or None. """
    substituted_negated_clause = substitute_parameter_values(
        negated_clause.negated_clause_in_negation_normal_form, bool_values, int_values)

    if isinstance(substituted_negated_clause, BoolLiteral):
        if not substituted_negated_clause.value:
            return None
        return {var: 0 for var in negated_clause.universal_vars}

    atom_conjunctions = expand_into_atom_conjunctions(substituted_negated_clause, MAX_DISJUNCTS_IN_VERIFICATION_DNF)
    if atom_conjunctions is None:
        raise Unsupported_Formula_Shape(f'Clause {negated_clause.clause_index} has more than '
                                        f'{MAX_DISJUNCTS_IN_VERIFICATION_DNF} disjuncts after substitution')

    for atom_conjunction in atom_conjunctions:
        statistics.verification_atom_conjunctions_decided += 1
        model = decide_atom_conjunction_and_find_model(atom_conjunction, ctx)
        if model is not None:
            universal_values = {var: 0 for var in negated_clause.universal_vars}
            universal_values.update(model)
            return universal_values
    return None


def find_counterexample_to_candidate(shape: Exists_Forall_Shape,
                                     clause_order: List[int],
                                     bool_values: Dict[Var, bool],
                                     int_values: Dict[Var, int],
                                     ctx: EvaluationContext,
                                     statistics: Cegqi_Statistics) -> Optional[Counterexample]:
    """
    A clause and universal values violating it under the candidate, or None if the candidate satisfies
    every clause for all universal values. Clauses are tried in `clause_order`; the violated clause is
    moved to its front, so the clause that failed most recently is tried first in the next round.
    """
    for position, clause_index in enumerate(clause_order):
        negated_clause = shape.negated_clauses[clause_index]
        universal_values = find_counterexample_for_clause(negated_clause, bool_values, int_values, ctx, statistics)
        if universal_values is not None:
            clause_order.insert(0, clause_order.pop(position))
            return Counterexample(negated_clause=negated_clause, universal_values=universal_values)
    return None


# ---------------------------------------------------------------------------------------------
# Generalization by projection
# ---------------------------------------------------------------------------------------------

Lemma_Key = Tuple[int, FrozenSet[Tuple[Var, bool]], Tuple[str, ...]]


@dataclass
class Component_Constraint:
    """
    A theory constraint on the Int parameters, represented by a SAT variable; the synthesis checks the
    conjunction of the constraints set to true by intersecting their `allowed_values_automaton`s. Two
    kinds:

    | Kind | Meaning | `allowed_values_automaton` |
    | --- | --- | --- |
    | projection component `T_j` | the parameters on `tracks` are not in `Bad_j = { K_j \| exists X_j. AND atoms_j }`, for one component of a lemma's atoms (`split_atoms_into_components_connected_by_universal_vars`) | complement of `bad_values_automaton` |
    | parameter atom | an atom over Int parameters only, from a ground instance (`Candidate_Synthesizer.add_ground_instance`) | the atom's automaton; `bad_values_automaton` is None |

    Both kinds occur only positively in the SAT clauses, so a constraint set to false imposes nothing.
    """
    constraint_id: int
    sat_var_id: int
    tracks: FrozenSet[Var]
    bad_values_automaton: Optional[MTBDD_NFA]
    allowed_values_automaton: MTBDD_NFA


@dataclass
class Projection_Lemma:
    """
    `AND guard_literals -> OR_j T_j` over `component_constraints`, which is
    `AND guard_literals -> K not in AND_j Bad_j = Bad`. An empty `component_constraints` is the clause
    `NOT AND guard_literals`: every component's `Bad_j` is all values.
    """
    lemma_id: int
    key: Lemma_Key
    guard_literals: FrozenSet[Tuple[Var, bool]]
    component_constraints: Tuple[Component_Constraint, ...]


def make_lemma_key(clause_index: int, implicant: Implicant_At_Point) -> Lemma_Key:
    return (clause_index, implicant.guard_literals, tuple(sorted(str(atom) for atom in implicant.atoms)))


def _substitute_linear_term_into_atom(atom: Relation | Congruence,
                                      var: Var,
                                      substituted_terms: Dict[Var, int],
                                      substituted_constant: int) -> ASTp_Node:
    """ `atom` with `var` replaced by `sum(coef * v for v, coef in substituted_terms) + substituted_constant`. """
    coef_of_var = dict(zip(atom.vars, atom.coefs))
    var_coef = coef_of_var.pop(var, 0)
    if var_coef == 0:
        return atom
    for term_var, term_coef in substituted_terms.items():
        coef_of_var[term_var] = coef_of_var.get(term_var, 0) + var_coef * term_coef
    rhs = atom.rhs - var_coef * substituted_constant
    if isinstance(atom, Congruence):
        return make_congruence_with_sorted_vars(list(coef_of_var), list(coef_of_var.values()), rhs, atom.modulus)
    return make_relation_with_sorted_vars(list(coef_of_var), list(coef_of_var.values()), rhs, atom.predicate_symbol)


def eliminate_universal_vars_exactly(atoms: Sequence[Relation | Congruence],
                                     vars_to_eliminate: Iterable[Var],
                                     ) -> Optional[Tuple[List[Relation | Congruence], int]]:
    """
    Rewrite `exists vars_to_eliminate. AND atoms` into an equivalent `exists X'. AND atoms'` with
    fewer bound variables, by two rules that are exact over the integers for any coefficients:

    | Rule | Applies to a bound `x` | Rewrite |
    | --- | --- | --- |
    | unit equation | an equation `+-x + t = b` with `x` not in `t` | substitute `x := +-(b - t)` into the other atoms, drop the equation |
    | one-sided | `x` occurs only in `<=` relations, all with coefficients of one sign | drop those relations (`x` can be taken arbitrarily far in the other direction) |

    Returns the rewritten atoms and the number of eliminated variables, or None if an atom folds to
    False (then the conjunction is unsatisfiable).
    """
    remaining_atoms: List[ASTp_Node] = list(atoms)
    bound_vars = set(vars_to_eliminate)
    eliminated_var_count = 0

    made_progress = True
    while made_progress:
        made_progress = False
        if any(atom == BoolLiteral(False) for atom in remaining_atoms):
            return None
        remaining_atoms = [atom for atom in remaining_atoms if not isinstance(atom, BoolLiteral)]

        for var in sorted(bound_vars):
            defining_equation = next((
                atom for atom in remaining_atoms
                if isinstance(atom, Relation) and atom.predicate_symbol == '='
                and abs(dict(zip(atom.vars, atom.coefs)).get(var, 0)) == 1
            ), None)
            if defining_equation is not None:
                coef_of_var = dict(zip(defining_equation.vars, defining_equation.coefs))
                var_coef = coef_of_var.pop(var)
                # var_coef * var + t = b  ==>  var = var_coef * (b - t), since var_coef = +-1
                substituted_terms = {term_var: -var_coef * term_coef for term_var, term_coef in coef_of_var.items()}
                substituted_constant = var_coef * defining_equation.rhs
                remaining_atoms = [
                    _substitute_linear_term_into_atom(atom, var, substituted_terms, substituted_constant)
                    for atom in remaining_atoms if atom is not defining_equation
                ]
                bound_vars.discard(var)
                eliminated_var_count += 1
                made_progress = True
                break

            atoms_with_var = [atom for atom in remaining_atoms if var in atom.vars]
            if not atoms_with_var:
                bound_vars.discard(var)
                made_progress = True
                break
            if all(isinstance(atom, Relation) and atom.predicate_symbol == '<=' for atom in atoms_with_var):
                coef_signs = set(dict(zip(atom.vars, atom.coefs))[var] > 0 for atom in atoms_with_var)
                if len(coef_signs) == 1:
                    remaining_atoms = [atom for atom in remaining_atoms if var not in atom.vars]
                    bound_vars.discard(var)
                    eliminated_var_count += 1
                    made_progress = True
                    break

    if any(atom == BoolLiteral(False) for atom in remaining_atoms):
        return None
    return [atom for atom in remaining_atoms if not isinstance(atom, BoolLiteral)], eliminated_var_count


def split_atoms_into_components_connected_by_universal_vars(atoms: Sequence[Relation | Congruence],
                                                            universal_vars: Set[Var],
                                                            ) -> List[List[Relation | Congruence]]:
    """
    Partition `atoms` so that atoms sharing a universal variable are in one part. An atom without a
    universal variable is a part of its own. Then `exists X. AND atoms` equals the conjunction over the
    parts of `exists X_part. AND part`, because no universal variable occurs in two parts.
    """
    component_of_atom_index: List[int] = list(range(len(atoms)))

    def find_component(atom_index: int) -> int:
        while component_of_atom_index[atom_index] != atom_index:
            component_of_atom_index[atom_index] = component_of_atom_index[component_of_atom_index[atom_index]]
            atom_index = component_of_atom_index[atom_index]
        return atom_index

    first_atom_index_with_var: Dict[Var, int] = {}
    for atom_index, atom in enumerate(atoms):
        for var in atom.vars:
            if var not in universal_vars:
                continue
            if var in first_atom_index_with_var:
                component_of_atom_index[find_component(atom_index)] = find_component(first_atom_index_with_var[var])
            else:
                first_atom_index_with_var[var] = atom_index

    atoms_by_component: Dict[int, List[Relation | Congruence]] = {}
    for atom_index, atom in enumerate(atoms):
        atoms_by_component.setdefault(find_component(atom_index), []).append(atom)
    return list(atoms_by_component.values())


class Component_Constraint_Store:
    """
    Builds and caches `Component_Constraint`s by the atoms of their component; lemmas of one clause
    typically share most components, and a shared component is one SAT variable.
    """

    def __init__(self, ctx: EvaluationContext, statistics: Cegqi_Statistics, first_sat_var_id: int):
        self.ctx = ctx
        self.statistics = statistics
        self.next_sat_var_id = first_sat_var_id
        self.constraints: List[Component_Constraint] = []
        self.constraint_by_atoms: Dict[Tuple, Optional[Component_Constraint]] = {}
        """ None marks a component whose `Bad_j` is all values; its constraint is unsatisfiable. """

    def _build_bad_values_automaton_for_component(self,
                                                  atoms: Sequence[Relation | Congruence],
                                                  universal_vars: Set[Var]) -> MTBDD_NFA:
        elimination_result = eliminate_universal_vars_exactly(atoms, universal_vars)
        if elimination_result is None:
            return MTBDD_NFA.trivial_nonaccepting(self.ctx.get_alphabet())
        remaining_atoms, eliminated_var_count = elimination_result
        self.statistics.universal_vars_eliminated_without_automata += eliminated_var_count
        if not remaining_atoms:
            return MTBDD_NFA.trivial_accepting(self.ctx.get_alphabet())

        atom_vars = set(var for atom in remaining_atoms for var in atom.vars)
        vars_to_project = tuple(sorted(atom_vars & universal_vars))
        self.statistics.universal_vars_projected_by_automata += len(vars_to_project)
        conjunction = make_connective_with_folding(Connective_Type.AND, remaining_atoms)
        formula = conjunction
        if vars_to_project:
            formula = AST_Quantifier(referenced_vars=tuple(sorted(atom_vars)), bound_vars=vars_to_project,
                                     child=conjunction)
        _reset_encoding_cache(self.ctx)
        return make_minimal_deterministic_automaton(parse.run_evaluation_procedure(formula, self.ctx))

    def get_parameter_atom_constraint(self, atom: Relation | Congruence) -> Component_Constraint:
        """ The constraint for an atom over Int parameters only. """
        cache_key = ('atom', str(atom))
        cached = self.constraint_by_atoms.get(cache_key)
        if cached is not None:
            return cached
        _reset_encoding_cache(self.ctx)
        atom_automaton = make_minimal_deterministic_automaton(parse.run_evaluation_procedure(atom, self.ctx))
        constraint = Component_Constraint(constraint_id=len(self.constraints),
                                          sat_var_id=self.next_sat_var_id,
                                          tracks=frozenset(atom.vars),
                                          bad_values_automaton=None,
                                          allowed_values_automaton=atom_automaton)
        self.next_sat_var_id += 1
        self.constraints.append(constraint)
        self.constraint_by_atoms[cache_key] = constraint
        return constraint

    def allocate_auxiliary_sat_var(self) -> int:
        sat_var_id = self.next_sat_var_id
        self.next_sat_var_id += 1
        return sat_var_id

    def get_component_constraint(self,
                                 atoms: Sequence[Relation | Congruence],
                                 universal_vars: Set[Var]) -> Optional[Component_Constraint]:
        """ The constraint for the component, or None if its `Bad_j` is all values. """
        cache_key = tuple(sorted(str(atom) for atom in atoms))
        if cache_key in self.constraint_by_atoms:
            self.statistics.component_automaton_cache_hits += 1
            return self.constraint_by_atoms[cache_key]

        bad_values_automaton = self._build_bad_values_automaton_for_component(atoms, universal_vars)
        # An automaton from the lazy construction keeps projected variables in `used_variables` as
        # tracks it does not constrain, so the parameter tracks are those minus the universal variables.
        parameter_tracks = frozenset(bad_values_automaton.used_variables) - universal_vars
        constraint: Optional[Component_Constraint] = None
        if not parameter_tracks:
            # The atoms hold at the counterexample, so `Bad_j` is nonempty; without tracks it is all values.
            assert bad_values_automaton.final_states, 'A component satisfied at the counterexample has an empty Bad'
        else:
            constraint = Component_Constraint(constraint_id=len(self.constraints),
                                              sat_var_id=self.next_sat_var_id,
                                              tracks=parameter_tracks,
                                              bad_values_automaton=bad_values_automaton,
                                              allowed_values_automaton=bad_values_automaton.complement())
            self.next_sat_var_id += 1
            self.constraints.append(constraint)
            self.statistics.component_automaton_sizes.append(len(bad_values_automaton.states))
        self.constraint_by_atoms[cache_key] = constraint
        return constraint


def build_bad_values_automaton(atoms: Sequence[Relation | Congruence],
                               universal_vars: Iterable[Var],
                               ctx: EvaluationContext) -> MTBDD_NFA:
    """
    The minimal DFA for `{ K | exists X. AND atoms }`, built as the intersection of the components'
    `Bad_j`. Not used by the loop, which keeps the components separate; exists for testing.
    """
    universal_vars = set(universal_vars)
    store = Component_Constraint_Store(ctx, Cegqi_Statistics(), first_sat_var_id=1)
    result: Optional[MTBDD_NFA] = None
    for component in split_atoms_into_components_connected_by_universal_vars(atoms, universal_vars):
        component_automaton = store._build_bad_values_automaton_for_component(component, universal_vars)
        result = component_automaton if result is None else make_minimal_deterministic_automaton(
            result.intersection(component_automaton))
    return result if result is not None else MTBDD_NFA.trivial_accepting(ctx.get_alphabet())


def generalize_counterexample_into_lemma(counterexample: Counterexample,
                                         bool_values: Dict[Var, bool],
                                         int_values: Dict[Var, int],
                                         lemma_id: int,
                                         component_constraint_store: Component_Constraint_Store,
                                         statistics: Cegqi_Statistics) -> Projection_Lemma:
    negated_clause = counterexample.negated_clause
    point_int_values = dict(int_values)
    point_int_values.update(counterexample.universal_values)
    implicant = extract_implicant_at_point(negated_clause.negated_clause_in_negation_normal_form,
                                           bool_values, point_int_values,
                                           preference=solver_config.cegqi_automata.implicant_preference)

    universal_vars = set(negated_clause.universal_vars)
    projection_start_ns = time.perf_counter_ns()
    component_constraints: List[Component_Constraint] = []
    for component in split_atoms_into_components_connected_by_universal_vars(implicant.atoms, universal_vars):
        constraint = component_constraint_store.get_component_constraint(component, universal_vars)
        if constraint is not None and constraint not in component_constraints:
            component_constraints.append(constraint)
    statistics.projection_time_ns += time.perf_counter_ns() - projection_start_ns

    lemma = Projection_Lemma(lemma_id=lemma_id,
                             key=make_lemma_key(negated_clause.clause_index, implicant),
                             guard_literals=implicant.guard_literals,
                             component_constraints=tuple(component_constraints))

    logger.info('CEGQI: lemma %d from clause %d: |gamma|=%d, %d atoms, components with Bad_j states %s.',
                lemma_id, negated_clause.clause_index, len(implicant.guard_literals), len(implicant.atoms),
                [len(constraint.bad_values_automaton.states) for constraint in component_constraints
                 if constraint.bad_values_automaton is not None])
    logger.debug('CEGQI: lemma %d atoms: %s, gamma: %s', lemma_id, implicant.atoms, sorted(implicant.guard_literals))
    return lemma


# ---------------------------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------------------------

@dataclass
class Cegqi_Statistics:
    rounds: int = 0
    sat_calls: int = 0
    synthesis_blocking_clauses: int = 0
    verification_atom_conjunctions_decided: int = 0
    universal_vars_eliminated_without_automata: int = 0
    seeded_ground_instances: int = 0
    point_instances: int = 0
    seeding_time_ns: int = 0
    universal_vars_projected_by_automata: int = 0
    component_automaton_cache_hits: int = 0
    component_automaton_sizes: List[int] = field(default_factory=list)
    synthesis_product_sizes: List[int] = field(default_factory=list)
    projection_time_ns: int = 0
    synthesis_time_ns: int = 0
    verification_time_ns: int = 0

    def format_report(self) -> str:
        component_sizes = sorted(self.component_automaton_sizes)
        product_sizes = self.synthesis_product_sizes
        return (f'rounds={self.rounds} sat_calls={self.sat_calls} '
                f'seeded_ground_instances={self.seeded_ground_instances} '
                f'point_instances={self.point_instances} '
                f'seeding_s={self.seeding_time_ns / 1e9:.3f} '
                f'synthesis_conflict_clauses={self.synthesis_blocking_clauses} '
                f'verification_atom_conjunctions_decided={self.verification_atom_conjunctions_decided} '
                f'universal_vars_eliminated_without_automata={self.universal_vars_eliminated_without_automata} '
                f'universal_vars_projected_by_automata={self.universal_vars_projected_by_automata} '
                f'component_constraints={len(component_sizes)} '
                f'max_component_bad_states={component_sizes[-1] if component_sizes else "-"} '
                f'median_component_bad_states={component_sizes[len(component_sizes) // 2] if component_sizes else "-"} '
                f'component_cache_hits={self.component_automaton_cache_hits} '
                f'max_synthesis_product_states={max(product_sizes) if product_sizes else "-"} '
                f'projection_s={self.projection_time_ns / 1e9:.3f} '
                f'synthesis_s={self.synthesis_time_ns / 1e9:.3f} '
                f'verification_s={self.verification_time_ns / 1e9:.3f}')


@dataclass
class Candidate:
    bool_values: Dict[Var, bool]
    int_values: Dict[Var, int]


def group_constraints_by_shared_tracks(constraints: Sequence[Component_Constraint]) -> List[List[Component_Constraint]]:
    """ Partition `constraints` so that constraints sharing a track are in one part. """
    group_of_track: Dict[Var, int] = {}
    groups: Dict[int, List[Component_Constraint]] = {}
    for constraint in constraints:
        merged_group_ids = sorted(set(group_of_track[track] for track in constraint.tracks if track in group_of_track))
        if merged_group_ids:
            target_group_id = merged_group_ids[0]
        else:
            target_group_id = constraint.constraint_id
            groups[target_group_id] = []
        for other_group_id in merged_group_ids[1:]:
            for other_constraint in groups.pop(other_group_id):
                groups[target_group_id].append(other_constraint)
                for track in other_constraint.tracks:
                    group_of_track[track] = target_group_id
        groups[target_group_id].append(constraint)
        for track in constraint.tracks:
            group_of_track[track] = target_group_id
    return list(groups.values())


class Candidate_Synthesizer:
    """
    Finds `(B*, K*)` satisfying every lemma, or reports that none exists. A DPLL(T) loop:

    - SAT variables: one per Bool parameter, one per `Component_Constraint` (`T_j`).
    - Clauses: one per lemma, `NOT gamma OR T_1 OR ... OR T_m`, and theory conflict clauses.
    - Theory check: the constraints set to true are grouped by shared tracks; per group the
      `allowed_values_automaton`s are intersected. An empty intersection adds the conflict clause
      `NOT T_a OR ... OR NOT T_b` over the constraints intersected until it became empty; the clause
      mentions no Bool parameter, so it holds in every later round.

    The solver prefers setting `T_j` to false, so the theory check sees only the constraints the
    lemma clauses force.
    """

    def __init__(self, bool_parameters: Sequence[Var], int_parameters: Sequence[Var],
                 ctx: EvaluationContext, statistics: Cegqi_Statistics):
        import pysat.solvers
        self.bool_parameters = tuple(bool_parameters)
        self.int_parameters = tuple(int_parameters)
        self.sat_var_id_of_bool_parameter = {var: index + 1 for index, var in enumerate(self.bool_parameters)}
        self.component_constraint_store = Component_Constraint_Store(
            ctx, statistics, first_sat_var_id=len(self.bool_parameters) + 1)
        self.statistics = statistics
        self.sat_solver = pysat.solvers.Solver()
        self.contains_empty_clause = False
        self.phase_preferred_constraint_count = 0
        self.group_product_cache: Dict[FrozenSet[int], Tuple[Optional[MTBDD_NFA], Tuple[Component_Constraint, ...]]] = {}

    def close(self):
        self.sat_solver.delete()

    def add_lemma(self, lemma: Projection_Lemma):
        clause = [(-1 if value else 1) * self.sat_var_id_of_bool_parameter[var] for var, value in lemma.guard_literals]
        clause.extend(constraint.sat_var_id for constraint in lemma.component_constraints)
        if not clause:
            self.contains_empty_clause = True
            return
        self.sat_solver.add_clause(clause)
        self._prefer_constraints_false()

    def _encode_into_sat(self, node: ASTp_Node) -> int:
        """
        A SAT literal implying the negation-normal-form `node` over Bool parameters and atoms over Int
        parameters (one-directional Tseitin encoding; `node` is monotone in its atoms and auxiliary
        variables, so the implication direction suffices). `node` must not be a Bool literal.
        """
        match node:
            case Var():
                return self.sat_var_id_of_bool_parameter[node]
            case AST_Negation():
                return -self.sat_var_id_of_bool_parameter[node.child]
            case Relation() | Congruence():
                return self.component_constraint_store.get_parameter_atom_constraint(node).sat_var_id
            case AST_Connective():
                child_literals = [self._encode_into_sat(child) for child in node.children]
                auxiliary_literal = self.component_constraint_store.allocate_auxiliary_sat_var()
                if node.type == Connective_Type.AND:
                    for child_literal in child_literals:
                        self.sat_solver.add_clause([-auxiliary_literal, child_literal])
                else:
                    self.sat_solver.add_clause([-auxiliary_literal] + child_literals)
                return auxiliary_literal
        raise ValueError(f'Unexpected node in a ground instance: {node}')

    def add_ground_instance(self, ground_instance: ASTp_Node):
        """
        Assert a ground instance `C_i(P, X*)`, a formula over the parameters only, in negation normal
        form. Sound because every model of `exists P. forall X. psi` satisfies it.
        """
        if isinstance(ground_instance, BoolLiteral):
            if not ground_instance.value:
                self.contains_empty_clause = True
            return
        for conjunct in _split_into_conjuncts(ground_instance):
            if isinstance(conjunct, AST_Connective) and conjunct.type == Connective_Type.OR:
                self.sat_solver.add_clause([self._encode_into_sat(child) for child in conjunct.children])
            else:
                self.sat_solver.add_clause([self._encode_into_sat(conjunct)])
        self._prefer_constraints_false()

    def _prefer_constraints_false(self):
        constraints = self.component_constraint_store.constraints
        if len(constraints) > self.phase_preferred_constraint_count:
            self.phase_preferred_constraint_count = len(constraints)
            self.sat_solver.set_phases([-constraint.sat_var_id for constraint in constraints])

    def _intersect_group(self, group: Sequence[Component_Constraint],
                         ) -> Tuple[Optional[MTBDD_NFA], Tuple[Component_Constraint, ...]]:
        """ The product of the group's allowed-value automata, or None and the constraints of the empty prefix. """
        cache_key = frozenset(constraint.constraint_id for constraint in group)
        cached = self.group_product_cache.get(cache_key)
        if cached is not None:
            return cached

        # Small automata first: an empty product is then found with fewer and smaller intersections.
        ordered_group = sorted(group, key=lambda constraint: len(constraint.allowed_values_automaton.states))
        product: Optional[MTBDD_NFA] = None
        intersected: List[Component_Constraint] = []
        result: Tuple[Optional[MTBDD_NFA], Tuple[Component_Constraint, ...]]
        for constraint in ordered_group:
            intersected.append(constraint)
            if product is None:
                product = constraint.allowed_values_automaton
            else:
                product = make_minimal_deterministic_automaton(product.intersection(constraint.allowed_values_automaton))
                self.statistics.synthesis_product_sizes.append(len(product.states))
            if product.find_model() is None:
                result = (None, tuple(intersected))
                break
        else:
            result = (product, tuple(intersected))
        self.group_product_cache[cache_key] = result
        return result

    def synthesize_candidate(self) -> Optional[Candidate]:
        if self.contains_empty_clause:
            return None
        while True:
            self.statistics.sat_calls += 1
            if not self.sat_solver.solve():
                return None
            sat_model = self.sat_solver.get_model()
            value_of_sat_var = {abs(literal): literal > 0 for literal in sat_model}
            bool_values = {var: value_of_sat_var.get(sat_var_id, False)
                           for var, sat_var_id in self.sat_var_id_of_bool_parameter.items()}
            asserted_constraints = [constraint for constraint in self.component_constraint_store.constraints
                                    if value_of_sat_var.get(constraint.sat_var_id, False)]

            int_values: Dict[Var, int] = {var: 0 for var in self.int_parameters}
            conflict_clauses: List[List[int]] = []
            for group in group_constraints_by_shared_tracks(asserted_constraints):
                product, intersected = self._intersect_group(group)
                if product is None:
                    conflict_clauses.append([-constraint.sat_var_id for constraint in intersected])
                    continue
                binary_model = product.find_model()
                assert binary_model is not None
                group_tracks = tuple(sorted(set().union(*(constraint.tracks for constraint in group))))
                int_values.update(convert_binary_model_into_decadic(
                    binary_model, tuple(product.used_variables), group_tracks))

            if not conflict_clauses:
                return Candidate(bool_values=bool_values, int_values=int_values)

            for conflict_clause in conflict_clauses:
                self.statistics.synthesis_blocking_clauses += 1
                logger.debug('CEGQI: synthesis theory conflict among %d constraints.', len(conflict_clause))
                self.sat_solver.add_clause(conflict_clause)


# ---------------------------------------------------------------------------------------------
# Forward ground seeding
# ---------------------------------------------------------------------------------------------

def pair_pre_state_and_post_state_vars(universal_vars: Iterable[Var],
                                       var_table: Dict[Var, VarInfo],
                                       pre_state_prefix: str,
                                       post_state_prefix: str) -> Dict[str, Tuple[Var, Var]]:
    """
    State dimensions `d` with a pre-state variable named `<pre_state_prefix><d>` and a post-state
    variable named `<post_state_prefix><d>`, e.g. `x_0` / `y_0`. The naming convention is that of the
    `gen_its_hard` generator; it is the only information seeding takes from outside the formula.
    """
    var_by_name = {var_table[var].name: var for var in universal_vars}
    state_dimensions: Dict[str, Tuple[Var, Var]] = {}
    for name, var in var_by_name.items():
        if not name.startswith(pre_state_prefix):
            continue
        dimension = name[len(pre_state_prefix):]
        post_state_var = var_by_name.get(post_state_prefix + dimension)
        if post_state_var is not None:
            state_dimensions[dimension] = (var, post_state_var)
    return state_dimensions


def collect_parameter_free_conjuncts(negated_clause: Negated_Clause) -> List[Relation | Congruence]:
    """ The top-level conjuncts of `N_i` that are atoms over universal variables only (the transition). """
    universal_vars = set(negated_clause.universal_vars)
    return [conjunct for conjunct in _split_into_conjuncts(negated_clause.negated_clause_in_negation_normal_form)
            if isinstance(conjunct, (Relation, Congruence)) and set(conjunct.vars) <= universal_vars]


def make_ground_instance_of_clause(negated_clause: Negated_Clause, universal_values: Dict[Var, int]) -> ASTp_Node:
    """ `C_i(P, X*) = NOT N_i(P, X*)` in negation normal form, a formula over the parameters only. """
    clause_in_negation_normal_form = convert_to_negation_normal_form(
        negated_clause.negated_clause_in_negation_normal_form, is_negated=True)
    return substitute_parameter_values(clause_in_negation_normal_form, {}, universal_values)


class Forward_Ground_Seeder:
    """
    Ground instances of the clauses along forward executions.

    A state assigns a value to every state dimension (`pair_pre_state_and_post_state_vars`). Expanding
    a state `s` instantiates every clause with its pre-state variables set to `s`: values of the other
    universal variables satisfying the clause's transition (`collect_parameter_free_conjuncts`) are
    taken from the early-stopping construction; if the transition admits none, the clause is skipped
    at `s`. The ground instance at that point is recorded and the post-state values form a successor
    state (a dimension the clause leaves unconstrained keeps its value from `s`). `expand` explores
    breadth first for a given number of steps; states and instances are deduplicated across calls.

    Every ground instance is sound to assert whatever the chosen point is; the choice of points only
    affects which refutations the instances support. Transitions with several successors contribute
    one of them.
    """

    def __init__(self, shape: Exists_Forall_Shape, ctx: EvaluationContext,
                 pre_state_prefix: str = 'x_', post_state_prefix: str = 'y_'):
        self.shape = shape
        self.ctx = ctx
        self.state_dimensions = pair_pre_state_and_post_state_vars(shape.universal_vars, ctx.var_table,
                                                                   pre_state_prefix, post_state_prefix)
        if not self.state_dimensions:
            logger.warning('CEGQI seeding: no universal variable pairs named %s<d> / %s<d>; no seeds.',
                           pre_state_prefix, post_state_prefix)
        self.ordered_dimensions = tuple(sorted(self.state_dimensions))
        self.transition_atoms_of_clause = {clause.clause_index: collect_parameter_free_conjuncts(clause)
                                           for clause in shape.negated_clauses}
        self.expanded_states: Set[Tuple[int, ...]] = set()
        self.seen_instance_keys: Set[Tuple[int, Tuple[Tuple[int, int], ...]]] = set()

    def is_enabled(self) -> bool:
        return bool(self.state_dimensions)

    def initial_state(self) -> Tuple[int, ...]:
        return tuple(0 for _ in self.ordered_dimensions)

    def states_of_point(self, universal_values: Dict[Var, int]) -> List[Tuple[int, ...]]:
        """ The pre-state and the post-state of a point, where the point assigns all their dimensions. """
        states = []
        for state_var_index in (0, 1):
            state_vars = [self.state_dimensions[dimension][state_var_index] for dimension in self.ordered_dimensions]
            if all(var in universal_values for var in state_vars):
                states.append(tuple(universal_values[var] for var in state_vars))
        return states

    def make_ground_instance_if_new(self, negated_clause: Negated_Clause,
                                    universal_values: Dict[Var, int]) -> Optional[ASTp_Node]:
        instance_key = (negated_clause.clause_index,
                        tuple(sorted((var.id, value) for var, value in universal_values.items())))
        if instance_key in self.seen_instance_keys:
            return None
        self.seen_instance_keys.add(instance_key)
        return make_ground_instance_of_clause(negated_clause, universal_values)

    def _expand_state(self, state: Tuple[int, ...],
                      ground_instances: List[Tuple[int, Dict[Var, int], ASTp_Node]]) -> List[Tuple[int, ...]]:
        pre_state_values = {self.state_dimensions[dimension][0]: value
                            for dimension, value in zip(self.ordered_dimensions, state)}
        successor_states = []
        for clause in self.shape.negated_clauses:
            clause_universal_vars = set(clause.universal_vars)
            fixed_values = {var: value for var, value in pre_state_values.items() if var in clause_universal_vars}
            transition_atoms: Optional[List[ASTp_Node]] = []
            for atom in self.transition_atoms_of_clause[clause.clause_index]:
                substituted_atom = substitute_parameter_values(atom, {}, fixed_values)
                if substituted_atom == BoolLiteral(False):
                    transition_atoms = None
                    break
                if not isinstance(substituted_atom, BoolLiteral):
                    transition_atoms.append(substituted_atom)
            if transition_atoms is None:
                continue
            transition_model = decide_atom_conjunction_and_find_model(transition_atoms, self.ctx)
            if transition_model is None:
                continue

            universal_values = {var: 0 for var in clause.universal_vars}
            universal_values.update(fixed_values)
            universal_values.update(transition_model)
            ground_instance = self.make_ground_instance_if_new(clause, universal_values)
            if ground_instance is not None:
                ground_instances.append((clause.clause_index, universal_values, ground_instance))

            if not any(self.state_dimensions[dimension][1] in clause_universal_vars
                       for dimension in self.ordered_dimensions):
                continue  # The clause has no post-state, e.g. a safety clause
            successor_states.append(tuple(
                universal_values.get(self.state_dimensions[dimension][1], value)
                for dimension, value in zip(self.ordered_dimensions, state)))
        return successor_states

    def expand(self, start_states: Iterable[Tuple[int, ...]], depth: int,
               ) -> List[Tuple[int, Dict[Var, int], ASTp_Node]]:
        """ New ground instances (clause index, point, instance) from `depth` breadth-first steps. """
        ground_instances: List[Tuple[int, Dict[Var, int], ASTp_Node]] = []
        if not self.is_enabled():
            return ground_instances
        frontier = [state for state in dict.fromkeys(start_states) if state not in self.expanded_states]
        for _ in range(depth):
            next_frontier: List[Tuple[int, ...]] = []
            for state in frontier:
                if state in self.expanded_states:
                    continue
                self.expanded_states.add(state)
                next_frontier.extend(successor for successor in self._expand_state(state, ground_instances)
                                     if successor not in self.expanded_states)
            frontier = list(dict.fromkeys(next_frontier))
        return ground_instances


def compute_forward_ground_seeds(shape: Exists_Forall_Shape, seed_depth: int, ctx: EvaluationContext,
                                 ) -> List[Tuple[int, Dict[Var, int], ASTp_Node]]:
    """ Ground instances along forward executions of up to `seed_depth` steps from the all-zero state. """
    seeder = Forward_Ground_Seeder(shape, ctx)
    return seeder.expand([seeder.initial_state()], seed_depth)


# ---------------------------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------------------------

@dataclass
class Cegqi_Outcome:
    is_sat: bool
    bool_values: Dict[Var, bool] = field(default_factory=dict)
    int_values: Dict[Var, int] = field(default_factory=dict)
    statistics: Cegqi_Statistics = field(default_factory=Cegqi_Statistics)


GENERALIZATION_DISJUNCT = 'disjunct'
GENERALIZATION_POINT = 'point'
GENERALIZATION_DISJUNCT_AND_POINT = 'disjunct-and-point'
GENERALIZATIONS = (GENERALIZATION_DISJUNCT, GENERALIZATION_POINT, GENERALIZATION_DISJUNCT_AND_POINT)


def solve_exists_forall_shape_with_cegqi(shape: Exists_Forall_Shape,
                                         ctx: EvaluationContext,
                                         max_rounds: Optional[int] = None,
                                         seed_depth: int = 0,
                                         generalization: str = GENERALIZATION_DISJUNCT,
                                         counterexample_seed_depth: int = 0) -> Cegqi_Outcome:
    """
    The loop. Per counterexample, `generalization` selects what is learned:

    | `generalization` | Learned | Termination |
    | --- | --- | --- |
    | `disjunct` | projection lemma (`generalize_counterexample_into_lemma`) | guaranteed (finitely many keys) |
    | `point` | ground instance `C_i(P, X*)` | not guaranteed |
    | `disjunct-and-point` | both | guaranteed |

    With `counterexample_seed_depth > 0`, forward seeding (`Forward_Ground_Seeder`) continues from the
    pre- and post-state of every counterexample for that many steps.
    """
    statistics = Cegqi_Statistics()
    synthesizer = Candidate_Synthesizer(shape.bool_parameters, shape.int_parameters, ctx, statistics)
    clause_order = list(range(len(shape.negated_clauses)))
    known_lemma_keys: Set[Lemma_Key] = set()
    seeder = Forward_Ground_Seeder(shape, ctx) if (seed_depth > 0 or counterexample_seed_depth > 0) else None

    def add_seeded_instances(ground_instances: List[Tuple[int, Dict[Var, int], ASTp_Node]]):
        for clause_index, universal_values, ground_instance in ground_instances:
            logger.debug('CEGQI seeding: clause %d at %s: %s', clause_index,
                         {ctx.var_table[var].name: value for var, value in universal_values.items()},
                         ground_instance)
            synthesizer.add_ground_instance(ground_instance)
        statistics.seeded_ground_instances += len(ground_instances)

    try:
        if seeder is not None and seed_depth > 0:
            seeding_start_ns = time.perf_counter_ns()
            add_seeded_instances(seeder.expand([seeder.initial_state()], seed_depth))
            statistics.seeding_time_ns += time.perf_counter_ns() - seeding_start_ns
            logger.info('CEGQI seeding: depth %d, %d states expanded, %d ground instances, %.3fs.',
                        seed_depth, len(seeder.expanded_states), statistics.seeded_ground_instances,
                        statistics.seeding_time_ns / 1e9)

        while max_rounds is None or statistics.rounds < max_rounds:
            statistics.rounds += 1

            synthesis_start_ns = time.perf_counter_ns()
            candidate = synthesizer.synthesize_candidate()
            statistics.synthesis_time_ns += time.perf_counter_ns() - synthesis_start_ns
            if candidate is None:
                logger.info('CEGQI: no candidate left after %d rounds, UNSAT. %s',
                            statistics.rounds, statistics.format_report())
                return Cegqi_Outcome(is_sat=False, statistics=statistics)

            assert set(candidate.int_values) <= set(shape.int_parameters), \
                'A candidate assigns a variable that is not an Int parameter'
            logger.debug('CEGQI: round %d candidate: Bool %s, Int %s', statistics.rounds,
                         {var.id: value for var, value in candidate.bool_values.items() if value},
                         {var.id: value for var, value in candidate.int_values.items() if value != 0})

            verification_start_ns = time.perf_counter_ns()
            counterexample = find_counterexample_to_candidate(shape, clause_order, candidate.bool_values,
                                                              candidate.int_values, ctx, statistics)
            statistics.verification_time_ns += time.perf_counter_ns() - verification_start_ns
            logger.info('CEGQI: round %d: synthesis %.3fs total, verification %.3fs total, projection %.3fs total.',
                        statistics.rounds, statistics.synthesis_time_ns / 1e9,
                        statistics.verification_time_ns / 1e9, statistics.projection_time_ns / 1e9)
            if counterexample is None:
                logger.info('CEGQI: candidate verified after %d rounds, SAT. %s',
                            statistics.rounds, statistics.format_report())
                return Cegqi_Outcome(is_sat=True, bool_values=candidate.bool_values,
                                     int_values=candidate.int_values, statistics=statistics)

            if generalization in (GENERALIZATION_POINT, GENERALIZATION_DISJUNCT_AND_POINT):
                # The instance is false under the candidate, so the next candidate differs.
                point_values = dict(counterexample.universal_values)
                ground_instance = make_ground_instance_of_clause(counterexample.negated_clause, point_values)
                logger.debug('CEGQI point: clause %d at %s: %s', counterexample.negated_clause.clause_index,
                             {ctx.var_table[var].name: value for var, value in point_values.items()}, ground_instance)
                synthesizer.add_ground_instance(ground_instance)
                statistics.point_instances += 1

            if seeder is not None and counterexample_seed_depth > 0:
                seeding_start_ns = time.perf_counter_ns()
                add_seeded_instances(seeder.expand(seeder.states_of_point(counterexample.universal_values),
                                                   counterexample_seed_depth))
                statistics.seeding_time_ns += time.perf_counter_ns() - seeding_start_ns

            if generalization in (GENERALIZATION_DISJUNCT, GENERALIZATION_DISJUNCT_AND_POINT):
                lemma = generalize_counterexample_into_lemma(counterexample, candidate.bool_values,
                                                             candidate.int_values, lemma_id=statistics.rounds,
                                                             component_constraint_store=synthesizer.component_constraint_store,
                                                             statistics=statistics)
                # The progress argument of the design document: a key that already has a lemma cannot be
                # produced again, because that lemma excludes the candidate's Int values.
                assert lemma.key not in known_lemma_keys, f'Lemma key repeated: {lemma.key}'
                known_lemma_keys.add(lemma.key)
                synthesizer.add_lemma(lemma)
    finally:
        synthesizer.close()

    logger.warning('CEGQI: round limit %s reached. %s', max_rounds, statistics.format_report())
    raise TimeoutError(f'CEGQI: round limit {max_rounds} reached')


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def evaluate_prepared_formula_with_cegqi_automata(astp: ASTp_Node, ctx: EvaluationContext) -> Evaluation_Result:
    """
    Evaluation strategy for `parse.perform_whole_evaluation_on_source_text`. Formulae outside the
    exists-forall shape, and configurations other than MTBDD over the integers, fall through to
    `parse.evaluate_prepared_formula_with_automata`.
    """
    config = solver_config.cegqi_automata
    if solver_config.backend_type != BackendType.MTBDD or solver_config.solution_domain != SolutionDomain.INTEGERS:
        logger.warning('CEGQI: requires -m MTBDD over the integers; falling through to the default evaluation.')
        return parse.evaluate_prepared_formula_with_automata(astp, ctx)

    try:
        shape = detect_exists_forall_shape(astp, ctx.var_table)
    except Unsupported_Formula_Shape as reason:
        logger.info('CEGQI: formula outside the exists-forall shape (%s); falling through.', reason)
        return parse.evaluate_prepared_formula_with_automata(astp, ctx)

    logger.info('CEGQI: %d Bool and %d Int parameters, %d universal variables, %d negated clauses, '
                'max |P_i| / |P| = %.3f.',
                len(shape.bool_parameters), len(shape.int_parameters), len(shape.universal_vars),
                len(shape.negated_clauses), shape.compute_maximal_clause_parameter_ratio())

    if config.show_shape:
        print_exists_forall_shape(shape, ctx.var_table)
        import sys
        sys.exit(0)

    previous_early_stop_setting = solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction
    solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction = True
    try:
        outcome = solve_exists_forall_shape_with_cegqi(shape, ctx, max_rounds=config.max_rounds,
                                                       seed_depth=config.seed_depth,
                                                       generalization=config.generalization,
                                                       counterexample_seed_depth=config.counterexample_seed_depth)
    except TimeoutError as round_limit:
        # No verdict: the run-amaya result protocol has only sat/unsat, so report and stop here.
        print('unknown')
        logger.warning('%s', round_limit)
        import sys
        sys.exit(0)
    finally:
        solver_config.decide_atom_conjunctions_by_early_stopping_lazy_construction = previous_early_stop_setting

    if config.report:
        logger.warning('CEGQI report: %s', outcome.statistics.format_report())

    if not outcome.is_sat:
        return Evaluation_Result(run_stats=ctx.stats, solutions_nfa=None, model=None, var_table=ctx.var_table)

    model: Dict[Var, int] = {var: 0 for var, var_info in ctx.var_table.items() if var_info.is_formula_param}
    model.update(outcome.int_values)
    model.update({var: int(value) for var, value in outcome.bool_values.items()})
    return Evaluation_Result(run_stats=ctx.stats, solutions_nfa=None, model=model, var_table=ctx.var_table)


def print_exists_forall_shape(shape: Exists_Forall_Shape, var_table: Dict[Var, VarInfo]):
    def name(var: Var) -> str:
        return var_table[var].name if var in var_table else f'v{var.id}'

    print(f'bool_parameters: {len(shape.bool_parameters)}')
    print(f'int_parameters: {len(shape.int_parameters)}')
    print(f'universal_vars: {len(shape.universal_vars)} ({", ".join(name(var) for var in shape.universal_vars)})')
    print(f'negated_clauses: {len(shape.negated_clauses)}')
    print(f'max_clause_parameter_ratio: {shape.compute_maximal_clause_parameter_ratio():.3f}')
    for clause in shape.negated_clauses:
        print(f'  clause {clause.clause_index}: |B_i|={len(clause.bool_parameters)} '
              f'|K_i|={len(clause.int_parameters)} |X_i|={len(clause.universal_vars)}')
