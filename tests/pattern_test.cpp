// pattern_test -- pin MusicGen's delayed codebook pattern to audiocraft.
//
// The interleaving is not recoverable from the weights and it is not checked by anything at
// run time: get a delay wrong by one and the model still produces audio, just misaligned
// across codebooks. So the closed form in mg/pattern.h is checked here against fixtures
// produced by the real `DelayedPatternProvider`, driven through the same
// `build_pattern_sequence` / `revert_pattern_sequence` calls `LMModel.generate` makes.
//
// Three shapes: the shipped 4-codebook model, a degenerate single codebook, and a case
// whose audio is shorter than the deepest delay so both ends of the sequence run past it.
//
// Regenerate the fixtures with tests/gen_pattern_fixture.py (needs gary's venv).
#include "mg/pattern.h"

#include <cstdio>
#include <numeric>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

// Compare a whole [rows][cols] block, naming the first cell that differs.
void check_block(const char* what, const std::vector<int32_t>& got, const int32_t* want,
                 int rows, int cols) {
    check((int)got.size() == rows * cols, what, (long long)got.size(), (long long)rows * cols);
    if ((int)got.size() != rows * cols) return;
    for (int q = 0; q < rows; ++q) {
        for (int s = 0; s < cols; ++s) {
            const size_t i = (size_t)q * cols + s;
            if (got[i] == want[i]) continue;
            std::fprintf(stderr, "FAIL %s at [q=%d, s=%d]: got %d, want %d\n",
                         what, q, s, got[i], want[i]);
            ++failures;
            return;   // one report per block is enough to diagnose
        }
    }
}

ac::DelayPattern shipped(int n_q, int timesteps) {
    std::vector<int> delays((size_t)n_q);
    std::iota(delays.begin(), delays.end(), 0);
    return ac::DelayPattern(std::move(delays), timesteps);
}

// One fixture's worth of checks: build, revert, mask and the loop's start offsets.
void check_case(const char* tag, int n_q, int timesteps, int steps, int32_t special,
                const int32_t* codes, const int32_t* sequence, const int32_t* mask_want,
                const int32_t* reverted, const int32_t* first_step) {
    const ac::DelayPattern p = shipped(n_q, timesteps);
    char what[96];

    std::snprintf(what, sizeof(what), "%s sequence length", tag);
    check(p.steps() == steps, what, p.steps(), steps);
    if (p.steps() != steps) return;

    std::snprintf(what, sizeof(what), "%s build", tag);
    const std::vector<int32_t> built = ac::build_pattern_sequence(p, codes, special);
    check_block(what, built, sequence, n_q, steps);

    std::snprintf(what, sizeof(what), "%s revert", tag);
    const std::vector<int32_t> back = ac::revert_pattern_sequence(p, sequence, special);
    check_block(what, back, reverted, n_q, timesteps);

    // Round trip: everything the pattern carries comes back, whatever the codes were.
    std::snprintf(what, sizeof(what), "%s round trip", tag);
    const std::vector<int32_t> round = ac::revert_pattern_sequence(p, built.data(), special);
    check_block(what, round, codes, n_q, timesteps);

    std::snprintf(what, sizeof(what), "%s mask", tag);
    const std::vector<uint8_t> mask = ac::pattern_mask(p);
    std::vector<int32_t> mask_i32(mask.begin(), mask.end());
    check_block(what, mask_i32, mask_want, n_q, steps);

    for (int t = 0; t < timesteps; ++t) {
        std::snprintf(what, sizeof(what), "%s first step with timestep %d", tag, t);
        check(p.first_step_with_timestep(t) == first_step[t], what,
              p.first_step_with_timestep(t), first_step[t]);
    }
}

} // namespace

int main() {
    // MG: n_q=4, timesteps=6, special=2048
    constexpr int MG_NQ = 4, MG_T = 6, MG_S = 10, MG_SPECIAL = 2048;
    static const int32_t MG_codes[] = {
        0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205, 300,
        301, 302, 303, 304, 305
    };
    static const int32_t MG_sequence[] = {
        2048, 0, 1, 2, 3, 4, 5, 2048, 2048, 2048, 2048, 2048, 100, 101, 102, 103, 104, 105,
        2048, 2048, 2048, 2048, 2048, 200, 201, 202, 203, 204, 205, 2048, 2048, 2048, 2048,
        2048, 300, 301, 302, 303, 304, 305
    };
    static const int32_t MG_mask[] = {
        0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1
    };
    static const int32_t MG_reverted[] = {
        0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205, 300,
        301, 302, 303, 304, 305
    };
    static const int32_t MG_first_step[] = {
        1, 2, 3, 4, 5, 6
    };
    // ONE: n_q=1, timesteps=5, special=2048
    constexpr int ONE_NQ = 1, ONE_T = 5, ONE_S = 6, ONE_SPECIAL = 2048;
    static const int32_t ONE_codes[] = {
        0, 1, 2, 3, 4
    };
    static const int32_t ONE_sequence[] = {
        2048, 0, 1, 2, 3, 4
    };
    static const int32_t ONE_mask[] = {
        0, 1, 1, 1, 1, 1
    };
    static const int32_t ONE_reverted[] = {
        0, 1, 2, 3, 4
    };
    static const int32_t ONE_first_step[] = {
        1, 2, 3, 4, 5
    };
    // THREE: n_q=3, timesteps=2, special=2048
    constexpr int THREE_NQ = 3, THREE_T = 2, THREE_S = 5, THREE_SPECIAL = 2048;
    static const int32_t THREE_codes[] = {
        0, 1, 100, 101, 200, 201
    };
    static const int32_t THREE_sequence[] = {
        2048, 0, 1, 2048, 2048, 2048, 2048, 100, 101, 2048, 2048, 2048, 2048, 200, 201
    };
    static const int32_t THREE_mask[] = {
        0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1
    };
    static const int32_t THREE_reverted[] = {
        0, 1, 100, 101, 200, 201
    };
    static const int32_t THREE_first_step[] = {
        1, 2
    };

    check_case("MG", MG_NQ, MG_T, MG_S, MG_SPECIAL, MG_codes, MG_sequence, MG_mask,
               MG_reverted, MG_first_step);
    check_case("ONE", ONE_NQ, ONE_T, ONE_S, ONE_SPECIAL, ONE_codes, ONE_sequence, ONE_mask,
               ONE_reverted, ONE_first_step);
    check_case("THREE", THREE_NQ, THREE_T, THREE_S, THREE_SPECIAL, THREE_codes,
               THREE_sequence, THREE_mask, THREE_reverted, THREE_first_step);

    // --- the properties the generation loop actually leans on ---------------------------
    {
        // At the real generation shape: 30 s at 50 Hz, a 6 s prompt.
        const ac::DelayPattern p = shipped(4, 1500);
        check(p.steps() == 1504, "1500 timesteps give 1504 sequence steps", p.steps(), 1504);
        check(p.first_step_with_timestep(300) == 301, "a 6 s prompt starts at step 301",
              p.first_step_with_timestep(300), 301);
        // The loop runs [first_step, steps): one prefill of the whole prefix, then one step
        // per remaining position.
        check(p.steps() - p.first_step_with_timestep(300) == 1203,
              "1203 decode steps after a 6 s prompt",
              p.steps() - p.first_step_with_timestep(300), 1203);

        // Every codebook covers every timestep exactly once, and every sequence position
        // either carries a timestep or is special. That is the invariant `generate`'s
        // final assertions check, stated once here instead.
        const std::vector<uint8_t> mask = ac::pattern_mask(p);
        for (int q = 0; q < p.codebooks(); ++q) {
            std::vector<int> seen((size_t)p.timesteps, 0);
            int live = 0;
            for (int s = 0; s < p.steps(); ++s) {
                const int t = p.timestep_at(q, s);
                const bool m = mask[(size_t)q * p.steps() + s] != 0;
                if ((t >= 0) != m) {
                    std::fprintf(stderr, "FAIL mask disagrees with timestep_at at [%d, %d]\n",
                                 q, s);
                    ++failures;
                }
                if (t >= 0) { ++seen[(size_t)t]; ++live; }
            }
            char what[64];
            std::snprintf(what, sizeof(what), "codebook %d covers every timestep", q);
            check(live == p.timesteps, what, live, p.timesteps);
            for (int t = 0; t < p.timesteps; ++t) {
                if (seen[(size_t)t] == 1) continue;
                std::fprintf(stderr, "FAIL codebook %d sees timestep %d %d time(s)\n",
                             q, t, seen[(size_t)t]);
                ++failures;
                break;
            }
        }

        // Causality: within one sequence step, deeper codebooks carry older timesteps. This
        // is the whole point of the delay -- codebook q at step s is predicted from a
        // context that already contains codebook q-1 for the same timestep.
        for (int s = 1; s < p.steps(); ++s) {
            for (int q = 1; q < p.codebooks(); ++q) {
                const int a = p.timestep_at(q - 1, s), b = p.timestep_at(q, s);
                if (a < 0 || b < 0) continue;
                if (b < a) continue;
                std::fprintf(stderr, "FAIL codebook %d at step %d is not older than %d\n",
                             q, s, q - 1);
                ++failures;
                break;
            }
        }
    }

    // Rejections: audiocraft asserts sorted delays, and a pattern with none is meaningless.
    {
        bool threw = false;
        try { ac::DelayPattern({0, 2, 1}, 4); } catch (const std::exception&) { threw = true; }
        check(threw, "unsorted delays are rejected", threw, 1);
        threw = false;
        try { ac::DelayPattern({}, 4); } catch (const std::exception&) { threw = true; }
        check(threw, "an empty pattern is rejected", threw, 1);
        threw = false;
        try { shipped(4, 4).first_step_with_timestep(4); }
        catch (const std::exception&) { threw = true; }
        check(threw, "a timestep past the end is rejected", threw, 1);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("pattern_test: ok\n");
    return 0;
}
