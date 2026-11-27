#include "../include/base.hpp"
#include "../include/custom_leaf.hpp"
#include "../include/operations.hpp"
#include "../include/sylvan-extra.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

using sylvan::MTBDD;

/*
Reduction of an NFA by the maximal forward direct simulation `<=` (Bustan and Grumberg; the
pruning step is `P(id, <)` of Mayr and Clemente, "Advanced automata minimization", POPL 2013):

    p <= q  iff  (p final implies q final) and for every symbol a and every a-successor p' of p
                 there is an a-successor q' of q with p' <= q'.

Every state `q` with `p <= q` accepts a superset of the words `p` accepts. `reduce_nfa_by_forward_simulation`
applies, in this order, five language-preserving steps:

On a deterministic automaton (at most one initial state and one destination per state and symbol),
simulation coincides with language inclusion and minimization is the reduction to apply: such an
automaton is minimized if `minimize_if_deterministic` is set and returned unchanged otherwise.

| Step | Rewrite |
| --- | --- |
| useless states | remove states from which no final state is reachable (`remove_nonfinishing_states`) |
| bisimulation | merge every class of bisimilar states into its smallest state (`quotient_by_bisimulation`); cheap, and it shrinks the input of the quadratic simulation |
| quotient | merge every class of simulation-equivalent states (`p <= q` and `q <= p`) into its smallest state |
| pruning | drop a destination `s` of a transition (and an initial state `s`) if the same set contains a `t` with `s < t` (strictly) |
| unreachable states | remove states not reachable from an initial state |

The relation is a bit matrix over the states, refined downwards from `p final implies q final` until
it is a fixpoint (`compute_forward_simulation`). A pair is checked by
`Simulation_Context::is_transition_mtbdd_simulated`, a joint
traversal of the two transition MTBDDs that compares the destination sets at every pair of leaves,
so no MTBDD is built while the relation is computed. The matrix has |Q|^2 bits and every pass
checks every pair still in the relation; automata with more than `max_states` states (after the first step) skip the
relation and keep only the first step.

`determinize_nfa_pruned_by_simulation` keeps the relation of the reduction and passes the subset construction a
pruning of every macrostate: a member `p` is dropped if another member `q` has `p < q`, or `p` and `q` are
equivalent and `q` is the smaller state. The reduction preserves the language of every remaining state, so the
relation still implies language inclusion among them, and the union of the members' languages is unchanged.
*/

struct Simulation_Relation {
    u64 state_count;
    u64 words_per_row;
    std::vector<u64> bits;

    explicit Simulation_Relation(u64 state_count)
        : state_count(state_count), words_per_row((state_count + 63) / 64), bits(state_count * ((state_count + 63) / 64), 0) {}

    bool contains(u64 smaller, u64 larger) const {
        return (bits[smaller * words_per_row + larger / 64] >> (larger % 64)) & 1u;
    }

    void insert(u64 smaller, u64 larger) {
        bits[smaller * words_per_row + larger / 64] |= (1ull << (larger % 64));
    }

    void remove(u64 smaller, u64 larger) {
        bits[smaller * words_per_row + larger / 64] &= ~(1ull << (larger % 64));
    }
};

/*
Dense indices of states. State numbers are usually close to 0..|Q|-1, so the index is a vector lookup
over the span of state numbers when the span is at most `MAX_SPAN_PER_STATE` times |Q|, and a hash map
lookup otherwise. A state outside the automaton has index -1.
*/
struct State_Indexer {
    static constexpr u64 MAX_SPAN_PER_STATE = 16;
    State smallest_state = 0;
    std::vector<s64> index_of_state_offset;
    std::unordered_map<State, u64> index_of_state;

    explicit State_Indexer(const std::vector<State>& state_of_index) {
        if (state_of_index.empty()) return;
        smallest_state = state_of_index.front();
        u64 span = static_cast<u64>(state_of_index.back() - smallest_state) + 1;
        if (span <= MAX_SPAN_PER_STATE * state_of_index.size()) {
            index_of_state_offset.assign(span, -1);
            for (u64 index = 0; index < state_of_index.size(); index++) {
                index_of_state_offset[state_of_index[index] - smallest_state] = static_cast<s64>(index);
            }
        } else {
            for (u64 index = 0; index < state_of_index.size(); index++) index_of_state[state_of_index[index]] = index;
        }
    }

    s64 get_index(State state) const {
        if (!index_of_state_offset.empty()) {
            if (state < smallest_state || static_cast<u64>(state - smallest_state) >= index_of_state_offset.size()) return -1;
            return index_of_state_offset[state - smallest_state];
        }
        auto index_it = index_of_state.find(state);
        return index_it == index_of_state.end() ? -1 : static_cast<s64>(index_it->second);
    }
};

struct Simulation_Context {
    const State_Indexer& state_indexer;
    const Simulation_Relation& relation;

    bool is_state_simulated(State smaller, State larger) const {
        if (smaller == larger) return true;
        s64 smaller_index = state_indexer.get_index(smaller);
        s64 larger_index  = state_indexer.get_index(larger);
        if (smaller_index < 0 || larger_index < 0) return false;
        return relation.contains(smaller_index, larger_index);
    }

    bool is_destination_set_simulated(const Transition_Destination_Set& smaller, const Transition_Destination_Set& larger) const {
        for (State smaller_destination : smaller.destination_set) {
            if (std::binary_search(larger.destination_set.begin(), larger.destination_set.end(), smaller_destination)) {
                continue;
            }
            bool is_matched = false;
            for (State larger_destination : larger.destination_set) {
                if (is_state_simulated(smaller_destination, larger_destination)) {
                    is_matched = true;
                    break;
                }
            }
            if (!is_matched) return false;
        }
        return true;
    }

    /*
    Whether for every symbol, every destination of `smaller` is simulated by some destination of
    `larger`; `mtbdd_false` is the empty destination set. The traversal visits at most one pair of
    nodes per path, and transition MTBDDs have one level per track.
    */
    bool is_transition_mtbdd_simulated(MTBDD smaller, MTBDD larger) const {
        if (smaller == sylvan::mtbdd_false || smaller == larger) return true;

        bool is_smaller_leaf = sylvan::mtbdd_isleaf(smaller);
        bool is_larger_leaf  = larger == sylvan::mtbdd_false || sylvan::mtbdd_isleaf(larger);
        if (is_smaller_leaf && is_larger_leaf) {
            auto smaller_leaf = reinterpret_cast<Transition_Destination_Set*>(sylvan::mtbdd_getvalue(smaller));
            if (larger == sylvan::mtbdd_false) return smaller_leaf->destination_set.empty();
            auto larger_leaf = reinterpret_cast<Transition_Destination_Set*>(sylvan::mtbdd_getvalue(larger));
            return is_destination_set_simulated(*smaller_leaf, *larger_leaf);
        }

        u32 smaller_var = is_smaller_leaf ? UINT32_MAX : sylvan::mtbdd_getvar(smaller);
        u32 larger_var  = is_larger_leaf  ? UINT32_MAX : sylvan::mtbdd_getvar(larger);
        u32 top_var = std::min(smaller_var, larger_var);
        MTBDD smaller_low  = smaller_var == top_var ? sylvan::mtbdd_getlow(smaller)  : smaller;
        MTBDD smaller_high = smaller_var == top_var ? sylvan::mtbdd_gethigh(smaller) : smaller;
        MTBDD larger_low   = larger_var  == top_var ? sylvan::mtbdd_getlow(larger)   : larger;
        MTBDD larger_high  = larger_var  == top_var ? sylvan::mtbdd_gethigh(larger)  : larger;
        return is_transition_mtbdd_simulated(smaller_low, larger_low)
               && is_transition_mtbdd_simulated(smaller_high, larger_high);
    }
};

static MTBDD get_transitions_of_state(const NFA& nfa, State state) {
    auto transitions_it = nfa.transitions.find(state);
    return transitions_it == nfa.transitions.end() ? sylvan::mtbdd_false : transitions_it->second;
}

/*
Refines the relation in passes over the pairs it still contains, removing a pair as soon as it fails
the check against the current relation, until a pass removes nothing. Every removal is justified by
the relation at the time of the removal, which contains the maximal simulation, so the fixpoint is
the maximal simulation.
*/
static Simulation_Relation compute_forward_simulation(const NFA& nfa, const std::vector<State>& state_of_index,
                                                      const State_Indexer& state_indexer) {
    const u64 state_count = state_of_index.size();
    Simulation_Relation relation(state_count);

    std::vector<bool> is_final(state_count);
    std::vector<MTBDD> transitions_of_index(state_count);
    for (u64 index = 0; index < state_count; index++) {
        is_final[index] = nfa.final_states.contains(state_of_index[index]);
        transitions_of_index[index] = get_transitions_of_state(nfa, state_of_index[index]);
    }
    for (u64 smaller = 0; smaller < state_count; smaller++) {
        for (u64 larger = 0; larger < state_count; larger++) {
            if (!is_final[smaller] || is_final[larger]) relation.insert(smaller, larger);
        }
    }

    Simulation_Context ctx = {.state_indexer = state_indexer, .relation = relation};
    u64 checked_pair_count = 0;
    bool was_relation_refined = true;
    while (was_relation_refined) {
        was_relation_refined = false;
        for (u64 smaller = 0; smaller < state_count; smaller++) {
            for (u64 word_index = 0; word_index < relation.words_per_row; word_index++) {
                u64 word = relation.bits[smaller * relation.words_per_row + word_index];
                while (word) {
                    u64 larger = word_index * 64 + __builtin_ctzll(word);
                    word &= word - 1;
                    if (larger == smaller) continue;
                    if ((++checked_pair_count & 0xFFF) == 0) AMAYA_CHECK_INTERRUPT();
                    if (!ctx.is_transition_mtbdd_simulated(transitions_of_index[smaller], transitions_of_index[larger])) {
                        relation.remove(smaller, larger);
                        was_relation_refined = true;
                    }
                }
            }
        }
    }
    return relation;
}

struct Simulation_Quotient_Info {
    const std::unordered_map<State, State>& representative_of_state;
    const State_Indexer& state_indexer;
    const Simulation_Relation& relation;

    /* Representatives of `states`, without those strictly simulated by another representative among them. */
    std::vector<State> rename_and_prune(const std::vector<State>& states) const {
        std::vector<State> representatives;
        for (State state : states) representatives.push_back(representative_of_state.at(state));
        std::sort(representatives.begin(), representatives.end());
        representatives.erase(std::unique(representatives.begin(), representatives.end()), representatives.end());

        std::vector<State> maximal_representatives;
        for (State candidate : representatives) {
            s64 candidate_index = state_indexer.get_index(candidate);
            bool is_strictly_simulated = false;
            for (State other : representatives) {
                // Distinct representatives are not simulation-equivalent, so `candidate <= other` is strict.
                if (other != candidate && relation.contains(candidate_index, state_indexer.get_index(other))) {
                    is_strictly_simulated = true;
                    break;
                }
            }
            if (!is_strictly_simulated) maximal_representatives.push_back(candidate);
        }
        return maximal_representatives;
    }
};

TASK_DECL_2(MTBDD, rename_and_prune_destinations_by_simulation_op, MTBDD, size_t);
TASK_IMPL_2(MTBDD, rename_and_prune_destinations_by_simulation_op, MTBDD, dd, size_t, param) {
    if (dd == sylvan::mtbdd_false) return sylvan::mtbdd_false;
    if (!sylvan::mtbdd_isleaf(dd)) return sylvan::mtbdd_invalid;

    auto quotient_info = reinterpret_cast<const Simulation_Quotient_Info*>(param);
    auto old_leaf_contents = reinterpret_cast<Transition_Destination_Set*>(sylvan::mtbdd_getvalue(dd));
    Transition_Destination_Set new_leaf_contents(quotient_info->rename_and_prune(old_leaf_contents->destination_set));
    if (new_leaf_contents.destination_set.empty()) return sylvan::mtbdd_false;
    return make_set_leaf(&new_leaf_contents);
}

static void collect_largest_destination_set_size_rec(MTBDD node, std::unordered_set<MTBDD>& visited, u64& largest_size) {
    if (node == sylvan::mtbdd_false || !visited.insert(node).second) return;
    if (sylvan::mtbdd_isleaf(node)) {
        auto leaf_contents = reinterpret_cast<Transition_Destination_Set*>(sylvan::mtbdd_getvalue(node));
        largest_size = std::max(largest_size, static_cast<u64>(leaf_contents->destination_set.size()));
        return;
    }
    collect_largest_destination_set_size_rec(sylvan::mtbdd_getlow(node), visited, largest_size);
    collect_largest_destination_set_size_rec(sylvan::mtbdd_gethigh(node), visited, largest_size);
}

/* Whether `nfa` has at most one initial state and at most one destination per state and symbol. */
static bool is_nfa_deterministic(const NFA& nfa) {
    if (nfa.initial_states.size() > 1) return false;
    std::unordered_set<MTBDD> visited;
    u64 largest_destination_set_size = 0;
    for (auto& [state, transitions] : nfa.transitions) {
        collect_largest_destination_set_size_rec(transitions, visited, largest_destination_set_size);
        if (largest_destination_set_size > 1) return false;
    }
    return true;
}

/* Maps every destination of a leaf through `destination_map` (state -> state), removing duplicates. */
struct Destination_Map_Info {
    const std::unordered_map<State, State>* destination_map;
};

TASK_DECL_2(MTBDD, map_destinations_op, MTBDD, size_t);
TASK_IMPL_2(MTBDD, map_destinations_op, MTBDD, dd, size_t, param) {
    if (dd == sylvan::mtbdd_false) return sylvan::mtbdd_false;
    if (!sylvan::mtbdd_isleaf(dd)) return sylvan::mtbdd_invalid;
    auto info = reinterpret_cast<const Destination_Map_Info*>(param);
    auto leaf_contents = reinterpret_cast<Transition_Destination_Set*>(sylvan::mtbdd_getvalue(dd));
    std::vector<State> mapped_destinations;
    for (State destination : leaf_contents->destination_set) mapped_destinations.push_back(info->destination_map->at(destination));
    std::sort(mapped_destinations.begin(), mapped_destinations.end());
    mapped_destinations.erase(std::unique(mapped_destinations.begin(), mapped_destinations.end()), mapped_destinations.end());
    if (mapped_destinations.empty()) return sylvan::mtbdd_false;
    Transition_Destination_Set mapped_leaf_contents(std::move(mapped_destinations));
    return make_set_leaf(&mapped_leaf_contents);
}

struct Hash_Block_Signature {
    size_t operator()(const std::pair<State, MTBDD>& signature) const {
        return std::hash<u64>()(static_cast<u64>(signature.first) * 0x9E3779B97F4A7C15ull ^ signature.second);
    }
};

/*
The quotient of `nfa` by its maximal bisimulation, computed by signature refinement: starting from the
partition into final and nonfinal states, a state's signature is its block together with its transition
MTBDD with every destination replaced by the smallest state of the destination's block (MTBDDs are
canonical, so equal signatures are equal handles); blocks are refined until their number is stable. Every
block is merged into its smallest state. Bisimilar states are simulation-equivalent, so the quotient
keeps every simulation-equivalence class that matters to `reduce_nfa_by_forward_simulation`.
*/
NFA quotient_by_bisimulation(const NFA& nfa) {
    LACE_ME;
    std::vector<State> states(nfa.states.begin(), nfa.states.end());
    std::unordered_map<State, State> block_of_state;  // the smallest state of the block
    State smallest_final = -1, smallest_nonfinal = -1;
    for (State state : states) {
        State& smallest = nfa.final_states.contains(state) ? smallest_final : smallest_nonfinal;
        if (smallest < 0) smallest = state;
        block_of_state[state] = smallest;
    }
    u64 block_count = (smallest_final >= 0) + (smallest_nonfinal >= 0);

    while (true) {
        AMAYA_CHECK_INTERRUPT();
        Destination_Map_Info info = {.destination_map = &block_of_state};
        const u64 operation_id = get_next_operation_id();
        std::unordered_map<std::pair<State, MTBDD>, State, Hash_Block_Signature> block_of_signature;
        std::unordered_map<State, State> refined_block_of_state;
        u64 pushed_signature_count = 0;
        for (State state : states) {  // increasing, so the first state of a refined block is its smallest
            using namespace sylvan;  // mtbdd_uapplyp is a macro
            MTBDD signature = mtbdd_uapplyp(get_transitions_of_state(nfa, state), TASK(map_destinations_op), operation_id,
                                            reinterpret_cast<size_t>(&info));
            mtbdd_refs_push(signature);
            pushed_signature_count++;
            auto [block_entry, was_inserted] = block_of_signature.emplace(std::make_pair(block_of_state.at(state), signature), state);
            refined_block_of_state[state] = block_entry->second;
        }
        sylvan::mtbdd_refs_pop(pushed_signature_count);
        bool is_stable = block_of_signature.size() == block_count;
        block_count = block_of_signature.size();
        block_of_state = std::move(refined_block_of_state);
        if (is_stable) break;
    }

    NFA quotient(nfa.vars, nfa.var_count);
    for (State initial_state : nfa.initial_states) quotient.initial_states.insert(block_of_state.at(initial_state));
    Destination_Map_Info info = {.destination_map = &block_of_state};
    const u64 operation_id = get_next_operation_id();
    for (State state : states) {
        if (block_of_state.at(state) != state) continue;
        quotient.states.insert(state);
        if (nfa.final_states.contains(state)) quotient.final_states.insert(state);
        using namespace sylvan;
        MTBDD mapped_transitions = mtbdd_uapplyp(get_transitions_of_state(nfa, state), TASK(map_destinations_op), operation_id,
                                                 reinterpret_cast<size_t>(&info));
        if (mapped_transitions != mtbdd_false) {
            mtbdd_ref(mapped_transitions);
            quotient.transitions[state] = mapped_transitions;
        }
    }
    return quotient;
}

/*
The relation computed by a reduction, over the states of the automaton it was computed on (a superset of the
states of the reduced automaton, whose per-state languages it preserves).
*/
struct Computed_Simulation {
    std::vector<State> state_of_index;
    State_Indexer state_indexer;
    Simulation_Relation relation;
};

static NFA reduce_nfa_by_forward_simulation_keeping_relation(NFA& nfa, u64 max_states, bool minimize_if_deterministic,
                                                             std::optional<Computed_Simulation>* computed_simulation);

NFA reduce_nfa_by_forward_simulation(NFA& nfa, u64 max_states, bool minimize_if_deterministic) {
    return reduce_nfa_by_forward_simulation_keeping_relation(nfa, max_states, minimize_if_deterministic, nullptr);
}

NFA determinize_nfa_pruned_by_simulation(NFA& nfa, u64 max_states) {
    std::optional<Computed_Simulation> computed_simulation;
    NFA reduced = reduce_nfa_by_forward_simulation_keeping_relation(nfa, max_states, false, &computed_simulation);
    if (!computed_simulation.has_value()) return determinize_nfa(reduced);

    const State_Indexer& state_indexer = computed_simulation->state_indexer;
    const Simulation_Relation& relation = computed_simulation->relation;
    std::vector<s64> member_indices;
    std::function<void(Macrostate&)> prune_macrostate = [&](Macrostate& macrostate) {
        if (macrostate.size() < 2) return;
        member_indices.clear();
        for (State member : macrostate) member_indices.push_back(state_indexer.get_index(member));

        // Members are sorted, so of two equivalent members the one at the larger position is dropped. The kept
        // members include a maximal member above every dropped one, so the union of languages is unchanged.
        Macrostate kept_members;
        for (u64 position = 0; position < macrostate.size(); position++) {
            s64 index = member_indices[position];
            bool is_member_dropped = false;
            for (u64 other_position = 0; index >= 0 && other_position < macrostate.size() && !is_member_dropped; other_position++) {
                s64 other_index = member_indices[other_position];
                if (other_position == position || other_index < 0) continue;
                if (!relation.contains(index, other_index)) continue;
                is_member_dropped = !relation.contains(other_index, index) || other_position < position;
            }
            if (!is_member_dropped) kept_members.push_back(macrostate[position]);
        }
        macrostate.swap(kept_members);
    };
    return determinize_nfa_with_macrostate_pruning(reduced, &prune_macrostate);
}

static NFA reduce_nfa_by_forward_simulation_keeping_relation(NFA& nfa, u64 max_states, bool minimize_if_deterministic,
                                                             std::optional<Computed_Simulation>* computed_simulation) {
    LACE_ME;
    Interrupt_Guard interrupt_guard;

    // Simulation coincides with language inclusion on a deterministic automaton; minimization reduces it.
    if (is_nfa_deterministic(nfa)) {
        if (!minimize_if_deterministic) return NFA(nfa);
        // minimize_hopcroft expects a complete DFA; determinize_nfa completes it with a trap state.
        NFA complete_dfa = determinize_nfa(nfa);
        return minimize_hopcroft(complete_dfa);
    }

    NFA useful_nfa(nfa);
    remove_nonfinishing_states(useful_nfa);
    if (useful_nfa.initial_states.empty()) {
        // No initial state reaches a final state: the empty language, as one initial nonfinal state.
        return NFA(nfa.vars, nfa.var_count, {0}, {}, {0});
    }
    if (useful_nfa.final_states.empty() || (max_states > 0 && useful_nfa.states.size() > max_states)) {
        return useful_nfa;
    }

    // The relation compares pairs of states; merging bisimilar states first is cheap and shrinks it.
    useful_nfa = quotient_by_bisimulation(useful_nfa);

    std::vector<State> state_of_index(useful_nfa.states.begin(), useful_nfa.states.end());
    State_Indexer state_indexer(state_of_index);
    Simulation_Relation relation = compute_forward_simulation(useful_nfa, state_of_index, state_indexer);

    // States are visited in increasing order, so every class is represented by its smallest state.
    std::unordered_map<State, State> representative_of_state;
    for (u64 index = 0; index < state_of_index.size(); index++) {
        State state = state_of_index[index];
        if (representative_of_state.contains(state)) continue;
        representative_of_state[state] = state;
        for (u64 other_index = index + 1; other_index < state_of_index.size(); other_index++) {
            State other = state_of_index[other_index];
            if (!representative_of_state.contains(other)
                && relation.contains(index, other_index) && relation.contains(other_index, index)) {
                representative_of_state[other] = state;
            }
        }
    }

    Simulation_Quotient_Info quotient_info = {.representative_of_state = representative_of_state,
                                              .state_indexer = state_indexer, .relation = relation};
    NFA quotient(useful_nfa.vars, useful_nfa.var_count);
    std::vector<State> initial_states(useful_nfa.initial_states.begin(), useful_nfa.initial_states.end());
    for (State initial_state : quotient_info.rename_and_prune(initial_states)) quotient.initial_states.insert(initial_state);

    const u64 operation_id = get_next_operation_id();
    for (State state : useful_nfa.states) {
        if (representative_of_state.at(state) != state) continue;
        quotient.states.insert(state);
        if (useful_nfa.final_states.contains(state)) quotient.final_states.insert(state);

        using namespace sylvan;  // mtbdd_uapplyp is a macro
        MTBDD renamed_transitions = mtbdd_uapplyp(get_transitions_of_state(useful_nfa, state),
                                                  TASK(rename_and_prune_destinations_by_simulation_op),
                                                  operation_id, reinterpret_cast<size_t>(&quotient_info));
        if (renamed_transitions != mtbdd_false) {
            mtbdd_ref(renamed_transitions);
            quotient.transitions[state] = renamed_transitions;
        }
    }

    std::set<State> reachable_states = quotient.initial_states;
    std::vector<State> frontier(reachable_states.begin(), reachable_states.end());
    while (!frontier.empty()) {
        State state = frontier.back();
        frontier.pop_back();
        for (State successor : quotient.get_state_post(state)) {
            if (reachable_states.insert(successor).second) frontier.push_back(successor);
        }
    }
    std::set<State> unreachable_states;
    std::set_difference(quotient.states.begin(), quotient.states.end(), reachable_states.begin(), reachable_states.end(),
                        std::inserter(unreachable_states, unreachable_states.begin()));
    if (!unreachable_states.empty()) quotient.remove_states(unreachable_states);
    if (computed_simulation != nullptr) {
        computed_simulation->emplace(Computed_Simulation{.state_of_index = std::move(state_of_index),
                                                         .state_indexer = std::move(state_indexer),
                                                         .relation = std::move(relation)});
    }
    return quotient;
}
