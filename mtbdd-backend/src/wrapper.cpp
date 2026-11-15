#include "../include/wrapper.hpp"
#include "../include/base.hpp"
#include "../include/custom_leaf.hpp"
#include "../include/operations.hpp"
#include "../include/algorithms.hpp"

#include <algorithm>
#include <sylvan.h>
#include <sylvan_common.h>
#include <sylvan_mtbdd.h>
#include <unistd.h>
#include <assert.h>
#include <unordered_set>
#include <lace.h>
#include <utility>
#include <iostream>
#include <limits>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <cstdlib>
#include <cerrno>
#include <map>
#include <optional>

using namespace sylvan;

using std::set;
using std::vector;
using std::stringstream;
using std::map;

void init_machinery()
{
    int n_workers = 1;
#ifdef __EMSCRIPTEN__
    // WebAssembly: mmap allocates memory eagerly and the address space is at most 4 GB, so the Lace
    // task deque and sylvan's tables must be much smaller than on a native build.
    size_t dequeue_size = 1 << 18;
    size_t sylvan_memory_mb = 256;
#else
    size_t dequeue_size = 10000000;
    size_t sylvan_memory_mb = 4096;
#endif
    // Upper bound for sylvan's node table plus operation cache (both are allocated up front).
    // A value that is not a number of megabytes in [1, 2^24] is reported and ignored: strtoull would turn it
    // into 0, which sylvan rejects, or into a cap whose conversion to bytes overflows.
    if (const char* memory_mb = getenv("AMAYA_SYLVAN_MEMORY_MB")) {
        char* parse_end = nullptr;
        errno = 0;
        unsigned long long parsed_mb = strtoull(memory_mb, &parse_end, 10);
        bool is_number = errno == 0 && parse_end != memory_mb && *parse_end == '\0' && memory_mb[0] != '-';
        if (is_number && parsed_mb >= 1 && parsed_mb <= (1ull << 24)) {
            sylvan_memory_mb = parsed_mb;
        } else {
            std::cerr << "AMAYA_SYLVAN_MEMORY_MB=" << memory_mb << ": expected a number of megabytes in [1, "
                      << (1ull << 24) << "]; using " << sylvan_memory_mb << "\n";
        }
    }
    if (sizeof(size_t) == 4 && sylvan_memory_mb >= 4096) sylvan_memory_mb = 4095;  // 32-bit address space

    lace_init(n_workers, dequeue_size);
    //lace_startup(program_stack_size, TASK(_main), NULL);

    // THIS SEEMS TO BE THE SECRET
    // When the TASK parameter (the middle one) is set to be NULL, it does not spawn a new thread
    // and instead uses the current thread for all tasks (makes it possible to call from python)
    const size_t stack_size = 1LL << 20;
    lace_startup(0, NULL, NULL);

    sylvan_set_limits(sylvan_memory_mb << 20, 5, 5);

    // Testing aid: AMAYA_SYLVAN_TABLE_LOG2 (optionally with AMAYA_SYLVAN_TABLE_MAX_LOG2,
    // AMAYA_SYLVAN_CACHE_LOG2 and AMAYA_SYLVAN_CACHE_MAX_LOG2) shrinks sylvan's tables so that garbage
    // collection runs often, which exposes MTBDDs that are not protected across allocating calls.
    // Bounds: llmsset_create requires at least 2^9 table buckets, sylvan_set_sizes allows at most 2^42.
    if (getenv("AMAYA_SYLVAN_TABLE_LOG2")) {
        auto size_from_env = [](const char* variable_name, long minimum_log2, long maximum_log2, size_t fallback) -> size_t {
            const char* value = getenv(variable_name);
            if (!value) return fallback;

            char* parse_end = nullptr;
            errno = 0;
            long parsed_log2 = strtol(value, &parse_end, 10);

            bool strtol_reported_error     = (errno != 0);
            bool parsing_reached_end       = (parse_end != value && *parse_end == '\0');
            bool parsed_value_within_range = (minimum_log2 <= parsed_log2 && parsed_log2 <= maximum_log2);

            if (strtol_reported_error || !parsing_reached_end || !parsed_value_within_range) {
                std::cerr << variable_name << "=" << value << ": expected an integer in ["
                          << minimum_log2 << ", " << maximum_log2 << "]\n";
                exit(1);
            }
            return size_t{1} << parsed_log2;
        };
        size_t table_size = size_from_env("AMAYA_SYLVAN_TABLE_LOG2",     9, 42, 0);
        size_t table_max  = size_from_env("AMAYA_SYLVAN_TABLE_MAX_LOG2", 9, 42, table_size);
        size_t cache_size = size_from_env("AMAYA_SYLVAN_CACHE_LOG2",     0, 42, table_size);
        size_t cache_max  = size_from_env("AMAYA_SYLVAN_CACHE_MAX_LOG2", 0, 42, cache_size);
        sylvan_set_sizes(table_size, table_max, cache_size, cache_max);
    }
    //sylvan_set_sizes(1LL << 27, 1LL << 26, 1LL << 26, 1LL << 20);
    //sylvan_set_sizes(1LL << 24, 1LL << 28, 1LL << 24, 1LL << 28);
    sylvan_init_package();
    sylvan_init_mtbdd();

    // Initialize leaf type for leaves containing sets - represents outgoing transition from a state
    g_solver_context = new Solver_Context{};  // Value-initialize: config.flags must not be indeterminate
    Set_Leaf::init_set_leaf(&g_solver_context->leaf_id_store);
    Bit_Set_Leaf::init_bit_set_leaf(&g_solver_context->leaf_id_store);

    g_solver_context->bit_set_alloc = new Bit_Set::Block_Arena_Allocator();
}

void shutdown_machinery()
{
    LACE_ME;

    sylvan_gc();
    sylvan_quit();
    lace_exit();

    delete g_solver_context->bit_set_alloc;
    delete g_solver_context;
}

NFA construct_dfa_for_atom_conjunction(Serialized_Quantified_Atom_Conjunction* raw_formula, bool stop_after_first_accepting_macrostate) {
    Interrupt_Guard interrupt_guard;

    if (raw_formula->var_cnt >= 64) {
        // Symbols are enumerated as u64 bit vectors (Alphabet_Iterator), so >= 64 tracks cannot be represented.
        throw std::invalid_argument("construct_dfa_for_atom_conjunction: at most 63 variables are supported");
    }

    Formula_Description formula_desc;
    formula_desc.var_count = raw_formula->var_cnt;
    for (u64 atom_i = 0; atom_i < raw_formula->atom_cnt; atom_i++) {
        auto& atom = raw_formula->atoms[atom_i];
        switch (atom.type) {
            case (Presburger_Atom_Type::PR_ATOM_EQ):         formula_desc.equation_count   += 1; break;
            case (Presburger_Atom_Type::PR_ATOM_INEQ):       formula_desc.inequation_count += 1; break;
            case (Presburger_Atom_Type::PR_ATOM_CONGRUENCE): formula_desc.congruence_count += 1; break;
        }
    }

    Formula_Pool pool (formula_desc);
    auto formula_atoms = pool.allocator.allocate_formula(formula_desc);

    u64 next_free_congruence = 0;
    u64 next_free_equation   = 0;
    u64 next_free_inequation = 0;

    vector<s64> initial_state_data (raw_formula->atom_cnt);

    for (u64 atom_i = 0; atom_i < raw_formula->atom_cnt; atom_i++) {
        auto& atom = raw_formula->atoms[atom_i];
        switch (atom.type) {
            case (Presburger_Atom_Type::PR_ATOM_EQ): {
                Equation& eq = formula_atoms.equations.items[next_free_equation];
                std::memcpy(eq.coefs.items, atom.coefs, sizeof(s64) * formula_desc.var_count);

                initial_state_data[formula_desc.congruence_count + next_free_equation] = raw_formula->initial_state[atom_i];

                next_free_equation += 1;
                break;
            }
            case (Presburger_Atom_Type::PR_ATOM_INEQ): {
                Inequation& ineq = formula_atoms.inequations.items[next_free_inequation];
                std::memcpy(ineq.coefs.items, atom.coefs, sizeof(s64) * formula_desc.var_count);

                initial_state_data[formula_desc.congruence_count + formula_desc.equation_count + next_free_inequation] = raw_formula->initial_state[atom_i];

                next_free_inequation += 1;
                break;
            }
            case (Presburger_Atom_Type::PR_ATOM_CONGRUENCE): {
                Congruence& congruence = formula_atoms.congruences.items[next_free_congruence];
                std::memcpy(congruence.coefs.items, atom.coefs, sizeof(s64) * formula_desc.var_count);

                auto decomposed_modulus = decompose_modulus(atom.modulus);
                congruence.modulus_2pow = decomposed_modulus.modulus_2pow;
                congruence.modulus_odd  = decomposed_modulus.modulus_odd;

                initial_state_data[next_free_congruence] = raw_formula->initial_state[atom_i];

                next_free_congruence += 1;
                break;
            }
        }
    }

    vector<u64> quantified_vars;
    quantified_vars.resize(raw_formula->quantified_var_cnt);
    std::memcpy(quantified_vars.data(), raw_formula->quantified_vars, sizeof(u64) * raw_formula->quantified_var_cnt);

    sylvan::BDDSET var_set = sylvan::mtbdd_set_empty();
    for (u64 var_idx = 0u; var_idx < raw_formula->var_cnt; var_idx++) {
        var_set = sylvan::mtbdd_set_add(var_set, raw_formula->vars[var_idx]);
    }

    Quantified_Atom_Conjunction formula (formula_atoms.congruences, formula_atoms.equations,
                                         formula_atoms.inequations, quantified_vars, formula_desc.var_count);

    auto stored_formula_ptr = pool.store_formula(formula);

    Conjunction_State initial_state(initial_state_data);
    auto created_nfa = build_nfa_with_formula_entailement(stored_formula_ptr, initial_state, var_set, pool,
                                                          stop_after_first_accepting_macrostate);

    return created_nfa;
}

NFA perform_pad_closure_using_bit_sets(NFA& nfa) {
    return do_pad_closure_using_bit_sets(&nfa, g_solver_context->bit_set_alloc);
}

/*
 * Symbolic construction of the transition MTBDD of a single state of an atom automaton.
 *
 * Does not iterate over the entire alphabet directly. Instead, branch on one
 * variable at a time and share every subtree reached with the same partial
 * dot product.
 *
 * `leaf_from_dot_product_constructor(dot)` returns the leaf with the destinations for symbols whose
 * dot product with `coefs` is `dot`, or mtbdd_false if there is no such transition. It is a parameter
 * because the destinations depend on the atom kind (inequation, equation, congruence) and the state.
 */
template <typename Leaf_From_Dot_Product_Constructor>
static MTBDD build_transitions_by_dot_product(const BDDVAR* vars, const s64* coefs, u64 var_count,
                                              Leaf_From_Dot_Product_Constructor& leaf_from_dot_product_constructor) {
    vector<unordered_map<s64, MTBDD>> memo(var_count + 1);
    vector<MTBDD> referenced_nodes;

    std::function<MTBDD(u64, s64)> build = [&](u64 var_idx, s64 partial_dot) -> MTBDD {
        auto& level_memo = memo[var_idx];
        auto cached = level_memo.find(partial_dot);
        if (cached != level_memo.end()) return cached->second;

        MTBDD result;
        if (var_idx == var_count) {
            result = leaf_from_dot_product_constructor(partial_dot);
        } else {
            MTBDD low  = build(var_idx + 1, partial_dot);
            MTBDD high = build(var_idx + 1, partial_dot + coefs[var_idx]);
            result = mtbdd_makenode(vars[var_idx], low, high);
        }

        // Keep every intermediate node alive until the whole MTBDD is built - a garbage collection
        // triggered by a later mtbdd_makenode would otherwise be free to reclaim it.
        mtbdd_ref(result);
        referenced_nodes.push_back(result);
        level_memo.emplace(partial_dot, result);
        return result;
    };

    MTBDD root = build(0, 0);
    mtbdd_ref(root);
    for (MTBDD node : referenced_nodes) mtbdd_deref(node);
    return root;
}

// Install the transition MTBDD built by `build_transitions_by_dot_product` for `state`, taking over its
// reference. Every state is explored exactly once, so there are no transitions to merge with.
static void set_state_transitions(NFA& nfa, State state, MTBDD state_transitions) {
    if (state_transitions == mtbdd_false) return;  // No outgoing transitions - same as never calling add_transition
    auto [position, was_inserted] = nfa.transitions.emplace(state, state_transitions);
    assert(was_inserted);
}

static MTBDD make_destination_leaf(State destination, bool also_final, State final_state) {
    Transition_Destination_Set leaf_contents;
    leaf_contents.insert(destination);
    if (also_final) leaf_contents.insert(final_state);
    leaf_contents.sort();
    return make_set_leaf(&leaf_contents);
}

static vector<BDDVAR> bddset_to_vector(BDDSET vars, u64 var_count) {
    assert(mtbdd_set_count(vars) == var_count);
    vector<BDDVAR> var_arr(var_count);
    mtbdd_set_to_array(vars, var_arr.data());
    return var_arr;
}

struct Congruence_State {
    s64 modulus_odd;
    s64 modulus_2pow;
    s64 value;

    bool operator==(const Congruence_State& other) const {
        return modulus_odd == other.modulus_odd &&
               modulus_2pow == other.modulus_2pow &&
               value == other.value;
    }
};

template <>
struct std::hash<Congruence_State> {
    std::size_t operator() (const Congruence_State& state) const {
        return (1 << state.modulus_2pow) + state.modulus_odd*state.value;
    }
};

// Explores the congruence state graph reachable from whatever is already sitting in `worklist`,
// growing `nfa` with the states and transitions it discovers. `discovered_states` (state -> handle)
// is owned by the caller so that repeated calls reuse everything found by the previous ones - that
// reuse is what makes the bounded-variable construction below cheap.
//
// State handles are handed out as `discovered_states.size()`, which keeps them a contiguous range
// starting at 0 as long as the caller pre-registers the final state and every initial state.
static void explore_congruence_states(NFA& nfa,
                                      const s64* coefs,
                                      u64 var_count,
                                      s64 final_state_handle,
                                      unordered_map<Congruence_State, s64>& discovered_states,
                                      vector<pair<Congruence_State, s64>>& worklist) {
    assert(var_count > 0);
    vector<BDDVAR> var_arr = bddset_to_vector(nfa.vars, var_count);

    while (!worklist.empty()) {
        AMAYA_CHECK_INTERRUPT();

        Congruence_State state = worklist.back().first;
        s64 handle = worklist.back().second;
        worklist.pop_back();

        s64 modulus = combine_moduli(state.modulus_2pow, state.modulus_odd);
        nfa.states.insert(handle);

        auto leaf_from_dot_product_constructor = [&](s64 dot) -> MTBDD {
            s64 post = state.value - dot;
            s64 fin_post = state.value + dot;

            Congruence_State dest_state;
            dest_state.modulus_odd = state.modulus_odd;
            dest_state.modulus_2pow = 0;

            if (state.modulus_2pow > 0) {
                if ((post % 2) != 0) return mtbdd_false;

                post /= 2;
                s64 new_modulus = combine_moduli(state.modulus_2pow - 1, state.modulus_odd);
                post %= new_modulus;
                post += (post < 0) * new_modulus;

                dest_state.modulus_2pow = state.modulus_2pow-1;
                dest_state.value = post;
            } else {
                post += state.modulus_odd * ((post % 2) != 0);
                post /= 2;
                post = post % state.modulus_odd;
                post += state.modulus_odd * (post < 0);

                dest_state.value = post;
            }

            auto [insert_position, did_insert_happen] = discovered_states.emplace(dest_state, discovered_states.size());
            s64 dest_handle = insert_position->second;
            if (did_insert_happen) {
                worklist.push_back({dest_state, dest_handle});
            }

            bool is_also_final = (fin_post % modulus) == 0;
            return make_destination_leaf(dest_handle, is_also_final, final_state_handle);
        };

        MTBDD state_transitions = build_transitions_by_dot_product(var_arr.data(), coefs, var_count, leaf_from_dot_product_constructor);
        set_state_transitions(nfa, handle, state_transitions);
    }
}

// A state value that `explore_congruence_states` can never produce, used as the stand-in key for the
// automaton's single accepting state so that it can take part in the shared handle numbering.
static const Congruence_State FINAL_STATE_KEY = {1, 299993, -1};

NFA construct_nfa_from_congruence(Serialized_Atom* congruence, s64 init_val, BDDSET vars, u64 var_count) {
    Interrupt_Guard interrupt_guard;

    auto moduli = decompose_modulus(congruence->modulus);
    Congruence_State initial_state = {
        .modulus_odd = moduli.modulus_odd,
        .modulus_2pow = moduli.modulus_2pow,
        .value = init_val
    };

    u64 final_state_handle = 0;
    u64 init_state_handle  = 1;

    unordered_map<Congruence_State, s64> discovered_states {
        {FINAL_STATE_KEY, final_state_handle},
        {initial_state, init_state_handle}
    };
    vector<pair<Congruence_State, s64>> worklist = {{initial_state, init_state_handle}};

    NFA constructed_nfa(vars);
    constructed_nfa.var_count = var_count;
    constructed_nfa.add_state_final(final_state_handle);
    constructed_nfa.initial_states.insert(init_state_handle);

    assert(var_count > 0);
    explore_congruence_states(constructed_nfa, congruence->coefs, var_count, final_state_handle,
                              discovered_states, worklist);

    return constructed_nfa;
}

NFA construct_nfa_from_congruence_with_bounded_var(Serialized_Atom* congruence, s64 rhs, u64 bound_var_idx,
                                                  s64 lower_bound, s64 upper_bound,
                                                  BDDSET vars, u64 var_count) {
    Interrupt_Guard interrupt_guard;

    assert(var_count >= 2);              // Something has to be left once the bound variable is projected away
    assert(bound_var_idx < var_count);
    assert(congruence->modulus > 0);

    // Drop the bound variable's track and its coefficient. The i-th coefficient corresponds to the
    // i-th variable of `vars` in ascending order (that is the order `mtbdd_cube` reads a symbol in),
    // so removing the same index from both keeps the two aligned.
    vector<BDDVAR> var_arr(var_count);
    mtbdd_set_to_array(vars, var_arr.data());

    BDDSET reduced_vars = mtbdd_set_empty();
    vector<s64> reduced_coefs;
    reduced_coefs.reserve(var_count - 1);
    for (u64 i = 0; i < var_count; i++) {
        if (i == bound_var_idx) continue;
        reduced_vars = mtbdd_set_add(reduced_vars, var_arr[i]);
        reduced_coefs.push_back(congruence->coefs[i]);
    }

    u64 final_state_handle = 0;
    unordered_map<Congruence_State, s64> discovered_states {{FINAL_STATE_KEY, final_state_handle}};
    vector<pair<Congruence_State, s64>> worklist;

    NFA constructed_nfa(reduced_vars);
    constructed_nfa.var_count = var_count - 1;
    constructed_nfa.add_state_final(final_state_handle);

    if (lower_bound > upper_bound) {
        // Empty bound range - no instantiation of the bound variable exists, so the union is empty.
        // Leaving `initial_states` empty makes the automaton accept nothing.
        return constructed_nfa;
    }

    auto moduli = decompose_modulus(congruence->modulus);
    s64 modulus = combine_moduli(moduli.modulus_2pow, moduli.modulus_odd);

    // Substituting the bound variable with `n` turns the congruence's right-hand side into
    // `rhs - coef_of_bound_var * n  (mod modulus)`; everything else about the atom is unchanged.
    // Work with residues throughout - only `n mod modulus` influences the result, and staying inside
    // [0, modulus) keeps the stepping below free of overflow.
    s64 bound_var_coef = congruence->coefs[bound_var_idx] % modulus;
    bound_var_coef += (bound_var_coef < 0) * modulus;

    s64 rhs_residue = rhs % modulus;
    rhs_residue += (rhs_residue < 0) * modulus;

    s64 lower_bound_residue = lower_bound % modulus;
    lower_bound_residue += (lower_bound_residue < 0) * modulus;

    __int128 shifted_rhs = static_cast<__int128>(rhs_residue) -
                           static_cast<__int128>(bound_var_coef) * static_cast<__int128>(lower_bound_residue);
    s64 instantiated_rhs = static_cast<s64>(shifted_rhs % modulus);
    instantiated_rhs += (instantiated_rhs < 0) * modulus;

    // `upper_bound - lower_bound` can overflow a s64 for extreme bounds; unsigned arithmetic gives
    // the correct span even then, and we never need the value for anything but the loop guard.
    u64 last_offset = static_cast<u64>(upper_bound) - static_cast<u64>(lower_bound);

    // Register one initial state per *distinct* right-hand side. The right-hand sides walk an
    // arithmetic progression of step `-bound_var_coef` in Z_modulus, so the moment one repeats,
    // every remaining one repeats too and the loop can stop - which is what keeps this cheap for
    // bound ranges far wider than the modulus.
    std::unordered_set<s64> seen_rhs;
    for (u64 offset = 0; ; offset++) {
        AMAYA_CHECK_INTERRUPT();

        if (!seen_rhs.insert(instantiated_rhs).second) break;

        Congruence_State seed = {
            .modulus_odd = moduli.modulus_odd,
            .modulus_2pow = moduli.modulus_2pow,
            .value = instantiated_rhs
        };

        auto [insert_position, did_insert_happen] = discovered_states.emplace(seed, discovered_states.size());
        s64 seed_handle = insert_position->second;
        constructed_nfa.states.insert(seed_handle);
        constructed_nfa.initial_states.insert(seed_handle);
        if (did_insert_happen) {
            worklist.push_back({seed, seed_handle});
        }

        if (offset == last_offset) break;

        instantiated_rhs -= bound_var_coef;
        instantiated_rhs += (instantiated_rhs < 0) * modulus;
    }

    // A single sweep over the shared graph: states reachable from several initial states are
    // explored (and their transitions built) exactly once.
    explore_congruence_states(constructed_nfa, reduced_coefs.data(), constructed_nfa.var_count,
                              final_state_handle, discovered_states, worklist);

    return constructed_nfa;
}


NFA construct_nfa_from_ineq(Serialized_Atom* ineq, s64 init_state, BDDSET vars, u64 var_count) {
    Interrupt_Guard interrupt_guard;

    s64 final_state_handle = 0;
    s64 init_state_handle  = 1;
    s64 final_state = std::numeric_limits<s64>::max();

    unordered_map<s64, s64> discovered_states {
        {final_state, final_state_handle},
        {init_state, init_state_handle}
    };
    vector<pair<s64, s64>> worklist = {{init_state, init_state_handle}};

    assert(var_count > 0);
    vector<BDDVAR> var_arr = bddset_to_vector(vars, var_count);

    NFA constructed_nfa(vars);
    constructed_nfa.var_count = var_count;
    constructed_nfa.initial_states.insert(init_state_handle);
    constructed_nfa.add_state_final(final_state_handle);

    while (!worklist.empty()) {
        AMAYA_CHECK_INTERRUPT();

        s64 state  = worklist.back().first;
        s64 handle = worklist.back().second;
        worklist.pop_back();

        constructed_nfa.states.insert(handle);

        auto leaf_from_dot_product_constructor = [&](s64 dot) -> MTBDD {
            s64 post     = state - dot;
            s64 fin_post = state + dot;

            s64 post_div_2 = post / 2;
            s64 post_mod_2 = post % 2;
            post_div_2 -= (post_mod_2 != 0) * (post < 0);  // floor((state - dot) / 2)
            post = post_div_2;

            s64 post_handle = discovered_states.size();
            auto [insert_pos, insert_happend] = discovered_states.emplace(post, post_handle);
            if (insert_happend) {
                worklist.push_back({post, post_handle});
            } else {
                post_handle = insert_pos->second;
            }

            return make_destination_leaf(post_handle, fin_post >= 0, final_state_handle);
        };

        MTBDD state_transitions = build_transitions_by_dot_product(var_arr.data(), ineq->coefs, var_count, leaf_from_dot_product_constructor);
        set_state_transitions(constructed_nfa, handle, state_transitions);
    }

    return constructed_nfa;
}

NFA construct_nfa_from_eq(Serialized_Atom* eq, s64 init_state, BDDSET vars, u64 var_count) {
    Interrupt_Guard interrupt_guard;

    s64 final_state_handle = 0;
    s64 init_state_handle  = 1;
    s64 final_state = std::numeric_limits<s64>::max();

    unordered_map<s64, s64> discovered_states {
        {final_state, final_state_handle},
        {init_state, init_state_handle}
    };
    vector<pair<s64, s64>> worklist = {{init_state, init_state_handle}};

    assert(var_count > 0);
    vector<BDDVAR> var_arr = bddset_to_vector(vars, var_count);

    NFA constructed_nfa(vars);
    constructed_nfa.var_count = var_count;
    constructed_nfa.initial_states.insert(init_state_handle);
    constructed_nfa.add_state_final(final_state_handle);

    while (!worklist.empty()) {
        AMAYA_CHECK_INTERRUPT();

        s64 state  = worklist.back().first;
        s64 handle = worklist.back().second;
        worklist.pop_back();

        constructed_nfa.states.insert(handle);

        auto leaf_from_dot_product_constructor = [&](s64 dot) -> MTBDD {
            s64 post     = state - dot;
            s64 fin_post = state + dot;

            if ((post % 2) != 0) {
                return mtbdd_false;  // (state - dot) odd => fin_post odd as well, so no transition at all
            }
            post /= 2;

            s64 post_handle = discovered_states.size();
            auto [insert_pos, insert_happend] = discovered_states.emplace(post, post_handle);
            if (insert_happend) {
                worklist.push_back({post, post_handle});
            } else {
                post_handle = insert_pos->second;
            }

            return make_destination_leaf(post_handle, fin_post == 0, final_state_handle);
        };

        MTBDD state_transitions = build_transitions_by_dot_product(var_arr.data(), eq->coefs, var_count, leaf_from_dot_product_constructor);
        set_state_transitions(constructed_nfa, handle, state_transitions);
    }

    return constructed_nfa;
}

/*
 * Construction of the automaton for a quantifier-free conjunction of atoms intersected with an NFA.
 *
 * A state of the result is a pair (atom-state vector, NFA state). The atom-state vector holds one state per
 * atom, with the transition semantics of the single-atom constructions above (construct_nfa_from_ineq,
 * construct_nfa_from_eq, construct_nfa_from_congruence). Each of those automata has one final state without
 * outgoing transitions, entered on a symbol that the atom accepts as the last (sign) symbol. In the product
 * of all of them with the NFA, a state whose atom components are all that final state is final exactly when
 * its NFA component is final, and has no outgoing transitions; all such final states are merged into one
 * accepting sink. Product states that are not final in this sense have no final counterpart and are never
 * final themselves.
 */

struct Product_Atom {
    Presburger_Atom_Type type;
    vector<s64> coefs;   // Dense, one coefficient per variable of the conjunction, in ascending variable order
    s64 modulus_odd;     // Congruences only
};

/*
 * Entry 2*i is the state of atom i. Entry 2*i + 1 is, for a congruence, the exponent of the power-of-two part
 * of its current modulus (see construct_nfa_from_congruence), and 0 for the other atoms.
 */
typedef vector<s64> Atom_State_Vector;

struct Atom_State_Vector_Hash {
    std::size_t operator()(const Atom_State_Vector& state_vector) const {
        std::size_t hash = state_vector.size();
        for (s64 entry : state_vector) {
            hash ^= std::hash<s64>{}(entry) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Atom-state handle standing for "every atom accepts the symbol as the last one".
static const State ALL_ATOMS_ACCEPT_LAST_SYMBOL = 0;

struct Atom_Conjunction_Product_Info {
    const NFA* nfa;
    std::map<std::pair<State, State>, State> product_handles;   // (atom-state handle, NFA state) -> product state
    vector<Intersection_Discovery> work_queue;                     // .left = atom-state handle, .right = NFA state
    State accepting_sink_handle;
    bool  was_accepting_sink_reached;
};

TASK_DECL_3(MTBDD, atom_conjunction_product_op, MTBDD*, MTBDD*, size_t);

TASK_IMPL_3(MTBDD, atom_conjunction_product_op, MTBDD*, p_atom_transitions, MTBDD*, p_nfa_transitions, size_t, param) {
    MTBDD atom_transitions = *p_atom_transitions, nfa_transitions = *p_nfa_transitions;

    if (atom_transitions == mtbdd_false || nfa_transitions == mtbdd_false) return mtbdd_false;
    if (!mtbdd_isleaf(atom_transitions) || !mtbdd_isleaf(nfa_transitions)) return mtbdd_invalid;

    auto info = reinterpret_cast<Atom_Conjunction_Product_Info*>(param);
    auto atom_destinations = reinterpret_cast<Transition_Destination_Set*>(mtbdd_getvalue(atom_transitions));
    auto nfa_destinations  = reinterpret_cast<Transition_Destination_Set*>(mtbdd_getvalue(nfa_transitions));

    Transition_Destination_Set product_destinations;
    for (State atom_destination : atom_destinations->destination_set) {
        if (atom_destination == ALL_ATOMS_ACCEPT_LAST_SYMBOL) {
            bool does_nfa_accept = false;
            for (State nfa_destination : nfa_destinations->destination_set) {
                if (info->nfa->final_states.contains(nfa_destination)) {
                    does_nfa_accept = true;
                    break;
                }
            }
            if (does_nfa_accept) {
                product_destinations.insert(info->accepting_sink_handle);
                info->was_accepting_sink_reached = true;
            }
            continue;
        }

        for (State nfa_destination : nfa_destinations->destination_set) {
            auto product_key = std::make_pair(atom_destination, nfa_destination);
            State product_handle = static_cast<State>(info->product_handles.size()) + 1;  // 0 is the accepting sink
            auto [entry_position, was_inserted] = info->product_handles.emplace(product_key, product_handle);
            if (was_inserted) {
                info->work_queue.push_back({.left = atom_destination, .right = nfa_destination, .handle = product_handle});
            } else {
                product_handle = entry_position->second;
            }
            product_destinations.insert(product_handle);
        }
    }

    if (product_destinations.destination_set.empty()) return mtbdd_false;

    product_destinations.sort();
    return make_set_leaf(&product_destinations);
}

/*
 * Transition MTBDD of one atom-state vector, built symbolically: like build_transitions_by_dot_product, but
 * with one partial dot product per atom. `leaf_from_dot_products(dots)` returns the leaf for the symbols whose
 * dot product with the coefficients of atom i is dots[i], or mtbdd_false.
 */
static MTBDD build_transitions_by_dot_products(const vector<BDDVAR>& vars, const vector<Product_Atom>& atoms,
                                               const std::function<MTBDD(const vector<s64>&)>& leaf_from_dot_products) {
    const u64 var_count = vars.size();
    vector<unordered_map<vector<s64>, MTBDD, Atom_State_Vector_Hash>> memo(var_count + 1);
    vector<MTBDD> referenced_nodes;

    std::function<MTBDD(u64, const vector<s64>&)> build = [&](u64 var_idx, const vector<s64>& partial_dots) -> MTBDD {
        auto& level_memo = memo[var_idx];
        auto cached = level_memo.find(partial_dots);
        if (cached != level_memo.end()) return cached->second;

        MTBDD result;
        if (var_idx == var_count) {
            result = leaf_from_dot_products(partial_dots);
        } else {
            vector<s64> partial_dots_with_bit_set (partial_dots);
            for (u64 atom_idx = 0; atom_idx < atoms.size(); atom_idx++) {
                partial_dots_with_bit_set[atom_idx] += atoms[atom_idx].coefs[var_idx];
            }
            MTBDD low  = build(var_idx + 1, partial_dots);
            MTBDD high = build(var_idx + 1, partial_dots_with_bit_set);
            result = mtbdd_makenode(vars[var_idx], low, high);
        }

        // Keep every intermediate node alive until the whole MTBDD is built (see build_transitions_by_dot_product).
        mtbdd_ref(result);
        referenced_nodes.push_back(result);
        level_memo.emplace(partial_dots, result);
        return result;
    };

    MTBDD root = build(0, vector<s64>(atoms.size(), 0));
    mtbdd_ref(root);
    for (MTBDD node : referenced_nodes) mtbdd_deref(node);
    return root;
}

/*
 * The successor of `state_vector` for symbols with the given per-atom dot products, or nullopt if some atom
 * has no transition on them (an equation, or a congruence with an even modulus, with an odd difference).
 * `do_all_atoms_accept` is set to whether every atom accepts such a symbol as the last one.
 */
static std::optional<Atom_State_Vector> compute_atom_state_vector_post(const vector<Product_Atom>& atoms,
                                                                       const Atom_State_Vector& state_vector,
                                                                       const vector<s64>& dots,
                                                                       bool& do_all_atoms_accept) {
    Atom_State_Vector post_vector(state_vector.size(), 0);
    do_all_atoms_accept = true;

    for (u64 atom_idx = 0; atom_idx < atoms.size(); atom_idx++) {
        const Product_Atom& atom = atoms[atom_idx];
        s64 state    = state_vector[2*atom_idx];
        s64 dot      = dots[atom_idx];
        s64 post     = state - dot;
        s64 fin_post = state + dot;

        switch (atom.type) {
            case Presburger_Atom_Type::PR_ATOM_INEQ: {  // construct_nfa_from_ineq
                s64 post_div_2 = post / 2;
                post_div_2 -= ((post % 2) != 0) * (post < 0);  // floor((state - dot) / 2)
                post_vector[2*atom_idx] = post_div_2;
                do_all_atoms_accept &= (fin_post >= 0);
                break;
            }
            case Presburger_Atom_Type::PR_ATOM_EQ: {  // construct_nfa_from_eq
                if ((post % 2) != 0) return std::nullopt;
                post_vector[2*atom_idx] = post / 2;
                do_all_atoms_accept &= (fin_post == 0);
                break;
            }
            case Presburger_Atom_Type::PR_ATOM_CONGRUENCE: {  // construct_nfa_from_congruence
                s64 modulus_2pow = state_vector[2*atom_idx + 1];
                s64 modulus = combine_moduli(modulus_2pow, atom.modulus_odd);
                do_all_atoms_accept &= ((fin_post % modulus) == 0);

                if (modulus_2pow > 0) {
                    if ((post % 2) != 0) return std::nullopt;
                    post /= 2;
                    s64 new_modulus = combine_moduli(modulus_2pow - 1, atom.modulus_odd);
                    post %= new_modulus;
                    post += (post < 0) * new_modulus;
                    post_vector[2*atom_idx]     = post;
                    post_vector[2*atom_idx + 1] = modulus_2pow - 1;
                } else {
                    post += atom.modulus_odd * ((post % 2) != 0);
                    post /= 2;
                    post = post % atom.modulus_odd;
                    post += atom.modulus_odd * (post < 0);
                    post_vector[2*atom_idx] = post;
                }
                break;
            }
            default:
                throw std::invalid_argument("construct_atom_conjunction_intersected_with_nfa: unknown atom type");
        }
    }

    return post_vector;
}

NFA construct_atom_conjunction_intersected_with_nfa(Serialized_Quantified_Atom_Conjunction* raw_conjunction,
                                                    NFA& nfa,
                                                    bool stop_after_first_accepting_state) {
    LACE_ME;
    Interrupt_Guard interrupt_guard;

    if (raw_conjunction->quantified_var_cnt != 0) {
        throw std::invalid_argument("construct_atom_conjunction_intersected_with_nfa: the conjunction must be quantifier-free");
    }

    const u64 var_count = raw_conjunction->var_cnt;
    vector<BDDVAR> conjunction_vars(var_count);
    for (u64 var_idx = 0; var_idx < var_count; var_idx++) {
        conjunction_vars[var_idx] = static_cast<BDDVAR>(raw_conjunction->vars[var_idx]);
        if (var_idx > 0 && conjunction_vars[var_idx - 1] >= conjunction_vars[var_idx]) {
            // The MTBDD is built by branching on the variables in the given order, which must be the variable order.
            throw std::invalid_argument("construct_atom_conjunction_intersected_with_nfa: variables must be strictly ascending");
        }
    }

    vector<Product_Atom> atoms(raw_conjunction->atom_cnt);
    Atom_State_Vector initial_state_vector(2 * raw_conjunction->atom_cnt, 0);
    for (u64 atom_idx = 0; atom_idx < raw_conjunction->atom_cnt; atom_idx++) {
        const Serialized_Atom& raw_atom = raw_conjunction->atoms[atom_idx];
        if (raw_atom.coef_cnt != var_count) {
            throw std::invalid_argument("construct_atom_conjunction_intersected_with_nfa: every atom needs one coefficient per variable");
        }

        Product_Atom& atom = atoms[atom_idx];
        atom.type = static_cast<Presburger_Atom_Type>(raw_atom.type);
        atom.coefs.assign(raw_atom.coefs, raw_atom.coefs + var_count);
        atom.modulus_odd = 0;
        initial_state_vector[2*atom_idx] = raw_conjunction->initial_state[atom_idx];

        if (atom.type == Presburger_Atom_Type::PR_ATOM_CONGRUENCE) {
            auto decomposed_modulus = decompose_modulus(raw_atom.modulus);
            atom.modulus_odd = decomposed_modulus.modulus_odd;
            initial_state_vector[2*atom_idx + 1] = decomposed_modulus.modulus_2pow;
        }
    }

    BDDSET result_vars = nfa.vars;
    for (BDDVAR var : conjunction_vars) result_vars = mtbdd_set_add(result_vars, var);
    NFA result(result_vars, mtbdd_set_count(result_vars));

    // Atom-state vectors are interned; handle 0 is ALL_ATOMS_ACCEPT_LAST_SYMBOL.
    unordered_map<Atom_State_Vector, State, Atom_State_Vector_Hash> atom_state_handles;
    vector<Atom_State_Vector> atom_states_by_handle = {Atom_State_Vector{}};
    auto intern_atom_state_vector = [&](const Atom_State_Vector& state_vector) -> State {
        auto [entry_position, was_inserted] = atom_state_handles.emplace(state_vector, static_cast<State>(atom_states_by_handle.size()));
        if (was_inserted) atom_states_by_handle.push_back(state_vector);
        return entry_position->second;
    };

    // The transitions of an atom-state vector do not depend on the NFA state it is paired with.
    unordered_map<State, MTBDD> atom_transitions_by_handle;
    auto get_atom_transitions = [&](State atom_state_handle) -> MTBDD {
        auto cached = atom_transitions_by_handle.find(atom_state_handle);
        if (cached != atom_transitions_by_handle.end()) return cached->second;

        // Copied: interning a successor may reallocate atom_states_by_handle.
        const Atom_State_Vector state_vector = atom_states_by_handle[atom_state_handle];
        auto leaf_from_dot_products = [&](const vector<s64>& dots) -> MTBDD {
            bool do_all_atoms_accept = false;
            auto post_vector = compute_atom_state_vector_post(atoms, state_vector, dots, do_all_atoms_accept);
            if (!post_vector.has_value()) return mtbdd_false;

            Transition_Destination_Set leaf_contents;
            if (do_all_atoms_accept) leaf_contents.insert(ALL_ATOMS_ACCEPT_LAST_SYMBOL);
            leaf_contents.insert(intern_atom_state_vector(post_vector.value()));
            leaf_contents.sort();
            return make_set_leaf(&leaf_contents);
        };

        MTBDD transitions = build_transitions_by_dot_products(conjunction_vars, atoms, leaf_from_dot_products);
        atom_transitions_by_handle.emplace(atom_state_handle, transitions);  // Holds the reference taken by the builder
        return transitions;
    };

    Atom_Conjunction_Product_Info info;
    info.nfa = &nfa;
    info.accepting_sink_handle = 0;
    info.was_accepting_sink_reached = false;

    State initial_atom_state_handle = intern_atom_state_vector(initial_state_vector);
    for (State nfa_initial_state : nfa.initial_states) {
        State product_handle = static_cast<State>(info.product_handles.size()) + 1;
        auto [entry_position, was_inserted] = info.product_handles.emplace(std::make_pair(initial_atom_state_handle, nfa_initial_state), product_handle);
        if (!was_inserted) continue;
        result.initial_states.insert(product_handle);
        info.work_queue.push_back({.left = initial_atom_state_handle, .right = nfa_initial_state, .handle = product_handle});
    }

    const u64 product_op_id = get_next_operation_id();

    while (!info.work_queue.empty()) {
        AMAYA_CHECK_INTERRUPT();

        Intersection_Discovery explored_product = info.work_queue.back();
        info.work_queue.pop_back();

        result.states.insert(explored_product.handle);

        MTBDD atom_transitions = get_atom_transitions(explored_product.left);
        auto nfa_transitions_position = nfa.transitions.find(explored_product.right);
        MTBDD nfa_transitions = (nfa_transitions_position == nfa.transitions.end()) ? mtbdd_false : nfa_transitions_position->second;

        MTBDD product_transitions = mtbdd_applyp(atom_transitions, nfa_transitions, reinterpret_cast<u64>(&info),
                                                 TASK(atom_conjunction_product_op), product_op_id);
        if (product_transitions != mtbdd_false) {
            mtbdd_ref(product_transitions);
            result.transitions[explored_product.handle] = product_transitions;
        }

        if (stop_after_first_accepting_state && info.was_accepting_sink_reached) {
            // Transitions lead into the queued product states; they are kept as states without outgoing transitions.
            for (auto& queued_product : info.work_queue) result.states.insert(queued_product.handle);
            break;
        }
    }

    if (info.was_accepting_sink_reached) {
        result.add_state_final(info.accepting_sink_handle);
    }

    for (auto& [atom_state_handle, transitions] : atom_transitions_by_handle) mtbdd_deref(transitions);

    return result;
}

void amaya_enable_bit_sets() {
    g_solver_context->config.enable_feature(SOLVER_CONFIG_USE_BIT_SETS);
}
