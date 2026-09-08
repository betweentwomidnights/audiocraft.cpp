// sampling_test -- pin how a step's logits become a token.
//
// gary sends `use_sampling=True`, `top_p=0.0`, `top_k=250`, `temperature=1.0`, so top-k is
// the whole sampling story. Everything about it except the draw is deterministic -- the
// softmax, the cutoff, which tokens survive, the renormalization -- and those are checked
// here against `audiocraft.utils.utils.sample_top_k`'s own arithmetic.
//
// The draw is not comparable: `src/rng.h` is reproducible within audiocraft.cpp and shares
// nothing with torch's stream, which is why the parity check in docs/MUSICGEN_LM.md uses
// greedy decoding. What the draw *can* be held to is checked directly: it stays inside the
// surviving set, it is reproducible for a seed, and over many draws it tracks the
// distribution it was given.
//
// Regenerate the fixtures with tests/gen_sampling_fixture.py (needs gary's venv).
#include "mg/sampling.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

void check_close(float got, float want, float tol, const char* what, int index) {
    if (std::fabs(got - want) <= tol) return;
    std::fprintf(stderr, "FAIL %s[%d]: got %.9g, want %.9g\n", what, index, got, want);
    ++failures;
}

// One fixture: softmax, then the top-k filter, then the argmax it implies.
void check_case(const char* tag, int n, int k, int survivors, float temperature,
                const float* logits, const float* softmax_want, const float* filtered_want,
                int argmax_want) {
    char what[96];

    std::vector<float> probs;
    ac::softmax_into(logits, n, temperature, probs);
    std::snprintf(what, sizeof(what), "%s softmax", tag);
    for (int i = 0; i < n; ++i) check_close(probs[(size_t)i], softmax_want[i], 1e-6f, what, i);

    ac::top_k_filter(probs, k);
    std::snprintf(what, sizeof(what), "%s top-k", tag);
    for (int i = 0; i < n; ++i) check_close(probs[(size_t)i], filtered_want[i], 1e-6f, what, i);

    int live = 0;
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        if (probs[(size_t)i] > 0.0f) ++live;
        total += (double)probs[(size_t)i];
    }
    std::snprintf(what, sizeof(what), "%s survivor count", tag);
    check(live == survivors, what, live, survivors);
    std::snprintf(what, sizeof(what), "%s renormalized", tag);
    check_close((float)total, 1.0f, 1e-5f, what, 0);

    std::snprintf(what, sizeof(what), "%s argmax", tag);
    check(ac::argmax_token(logits, n) == argmax_want, what, ac::argmax_token(logits, n),
          argmax_want);

    // Temperature must not move the argmax: it scales the logits, and scaling by a positive
    // number is monotone. Worth stating, because a temperature applied in the wrong place
    // (to the probabilities rather than the logits) would break it.
    std::vector<float> hot;
    ac::softmax_into(logits, n, temperature, hot);
    int hottest = 0;
    for (int i = 1; i < n; ++i) if (hot[(size_t)i] > hot[(size_t)hottest]) hottest = i;
    std::snprintf(what, sizeof(what), "%s temperature preserves the argmax", tag);
    check(hottest == argmax_want, what, hottest, argmax_want);
}

} // namespace

int main() {
    // PLAIN: n=24, k=6, temperature=1.0, 6 survivor(s)
    constexpr int PLAIN_N = 24, PLAIN_K = 6, PLAIN_SURVIVORS = 6;
    constexpr float PLAIN_TEMPERATURE = 1.0f;
    static const float PLAIN_logits[] = {
        -3.13857436f, 3.69156361f, 5.59863663f, -1.03059995f, 0.2026456f, 1.34440184f,
        -1.79576969f, 0.754388928f, 0.0204596892f, -5.21306753f, 3.88128138f, -0.498500586f,
        0.294594258f, -0.671605408f, -0.555282235f, -1.28915954f, 1.97460616f, 0.650619805f,
        -0.908822417f, -1.77072883f, -1.38622177f, -5.4939661f, 0.200034186f, 5.3374896f
    };
    static const float PLAIN_softmax[] = {
        7.34777204e-05f, 0.0679903105f, 0.457802653f, 0.000604835688f, 0.00207600929f,
        0.0065026083f, 0.000281402841f, 0.00360452663f, 0.00173024193f, 9.23027437e-06f,
        0.0821940526f, 0.00102973427f, 0.00227594608f, 0.000866057468f, 0.000972893555f,
        0.000467031641f, 0.0122118602f, 0.0032492415f, 0.000683163584f, 0.000288538286f,
        0.000423831079f, 6.96982943e-06f, 0.00207059435f, 0.352584809f
    };
    static const float PLAIN_filtered[] = {
        0.0f, 0.069428429f, 0.467486024f, 0.0f, 0.0f, 0.00664015021f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0839326084f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0124701634f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.360042632f
    };
    constexpr int PLAIN_ARGMAX = 2;
    // HOT: n=24, k=6, temperature=2.5, 6 survivor(s)
    constexpr int HOT_N = 24, HOT_K = 6, HOT_SURVIVORS = 6;
    constexpr float HOT_TEMPERATURE = 2.5f;
    static const float HOT_logits[] = {
        -3.13857436f, 3.69156361f, 5.59863663f, -1.03059995f, 0.2026456f, 1.34440184f,
        -1.79576969f, 0.754388928f, 0.0204596892f, -5.21306753f, 3.88128138f, -0.498500586f,
        0.294594258f, -0.671605408f, -0.555282235f, -1.28915954f, 1.97460616f, 0.650619805f,
        -0.908822417f, -1.77072883f, -1.38622177f, -5.4939661f, 0.200034186f, 5.3374896f
    };
    static const float HOT_softmax[] = {
        0.00644310517f, 0.0989946499f, 0.212277651f, 0.014972277f, 0.0245202333f,
        0.0387139954f, 0.0110246353f, 0.0305754039f, 0.0227968935f, 0.002810081f,
        0.106799461f, 0.0185234994f, 0.0254388619f, 0.0172842927f, 0.0181075223f,
        0.0135011729f, 0.0498133376f, 0.0293322708f, 0.0157196485f, 0.0111356182f,
        0.0129870363f, 0.00251143379f, 0.0244946349f, 0.191222236f
    };
    static const float HOT_filtered[] = {
        0.0f, 0.141862467f, 0.30420059f, 0.0f, 0.0f, 0.0554783791f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.153046995f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0713840872f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.274027497f
    };
    constexpr int HOT_ARGMAX = 2;
    // COLD: n=24, k=6, temperature=0.4, 6 survivor(s)
    constexpr int COLD_N = 24, COLD_K = 6, COLD_SURVIVORS = 6;
    constexpr float COLD_TEMPERATURE = 0.4f;
    static const float COLD_logits[] = {
        -3.13857436f, 3.69156361f, 5.59863663f, -1.03059995f, 0.2026456f, 1.34440184f,
        -1.79576969f, 0.754388928f, 0.0204596892f, -5.21306753f, 3.88128138f, -0.498500586f,
        0.294594258f, -0.671605408f, -0.555282235f, -1.28915954f, 1.97460616f, 0.650619805f,
        -0.908822417f, -1.77072883f, -1.38622177f, -5.4939661f, 0.200034186f, 5.3374896f
    };
    static const float COLD_softmax[] = {
        2.11526963e-10f, 0.00550926197f, 0.648144543f, 4.11215666e-08f, 8.97529958e-07f,
        1.55845846e-05f, 6.07151973e-09f, 3.56530427e-06f, 5.69171675e-07f, 1.18307729e-12f,
        0.00885272492f, 1.55521136e-07f, 1.12948635e-06f, 1.00888855e-07f, 1.34939597e-07f,
        2.15447944e-08f, 7.53235799e-05f, 2.75062234e-06f, 5.57555069e-08f, 6.46375264e-09f,
        1.69027707e-08f, 5.86180952e-13f, 8.91690036e-07f, 0.3373923f
    };
    static const float COLD_filtered[] = {
        0.0f, 0.00550931832f, 0.648151159f, 0.0f, 0.0f, 1.55847447e-05f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.00885281526f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.53243512e-05f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.337395757f
    };
    constexpr int COLD_ARGMAX = 2;
    // WIDE: n=24, k=24, temperature=1.0, 24 survivor(s)
    constexpr int WIDE_N = 24, WIDE_K = 24, WIDE_SURVIVORS = 24;
    constexpr float WIDE_TEMPERATURE = 1.0f;
    static const float WIDE_logits[] = {
        -3.13857436f, 3.69156361f, 5.59863663f, -1.03059995f, 0.2026456f, 1.34440184f,
        -1.79576969f, 0.754388928f, 0.0204596892f, -5.21306753f, 3.88128138f, -0.498500586f,
        0.294594258f, -0.671605408f, -0.555282235f, -1.28915954f, 1.97460616f, 0.650619805f,
        -0.908822417f, -1.77072883f, -1.38622177f, -5.4939661f, 0.200034186f, 5.3374896f
    };
    static const float WIDE_softmax[] = {
        7.34777204e-05f, 0.0679903105f, 0.457802653f, 0.000604835688f, 0.00207600929f,
        0.0065026083f, 0.000281402841f, 0.00360452663f, 0.00173024193f, 9.23027437e-06f,
        0.0821940526f, 0.00102973427f, 0.00227594608f, 0.000866057468f, 0.000972893555f,
        0.000467031641f, 0.0122118602f, 0.0032492415f, 0.000683163584f, 0.000288538286f,
        0.000423831079f, 6.96982943e-06f, 0.00207059435f, 0.352584809f
    };
    static const float WIDE_filtered[] = {
        7.34777204e-05f, 0.0679903105f, 0.457802653f, 0.000604835688f, 0.00207600929f,
        0.0065026083f, 0.000281402841f, 0.00360452663f, 0.00173024193f, 9.23027437e-06f,
        0.0821940526f, 0.00102973427f, 0.00227594608f, 0.000866057468f, 0.000972893555f,
        0.000467031641f, 0.0122118602f, 0.0032492415f, 0.000683163584f, 0.000288538286f,
        0.000423831079f, 6.96982943e-06f, 0.00207059435f, 0.352584809f
    };
    constexpr int WIDE_ARGMAX = 2;
    // TIED: n=24, k=6, temperature=1.0, 7 survivor(s)
    constexpr int TIED_N = 24, TIED_K = 6, TIED_SURVIVORS = 7;
    constexpr float TIED_TEMPERATURE = 1.0f;
    static const float TIED_logits[] = {
        -3.13857436f, 3.69156361f, 5.59863663f, -1.03059995f, 0.2026456f, 1.34440184f,
        -1.79576969f, 1.34440184f, 0.0204596892f, -5.21306753f, 3.88128138f, -0.498500586f,
        0.294594258f, -0.671605408f, -0.555282235f, -1.28915954f, 1.97460616f, 0.650619805f,
        -0.908822417f, -1.77072883f, -1.38622177f, -5.4939661f, 0.200034186f, 5.3374896f
    };
    static const float TIED_softmax[] = {
        7.32653934e-05f, 0.0677938387f, 0.456479758f, 0.000603087887f, 0.00207001017f,
        0.00648381794f, 0.00028058968f, 0.00648381794f, 0.00172524212f, 9.20360253e-06f,
        0.081956543f, 0.00102675869f, 0.00226936932f, 0.000863554829f, 0.000970082241f,
        0.000465682097f, 0.0121765714f, 0.00323985238f, 0.000681189471f, 0.000287704519f,
        0.000422606361f, 6.94968912e-06f, 0.00206461106f, 0.351565957f
    };
    static const float TIED_filtered[] = {
        0.0f, 0.0689704493f, 0.464402318f, 0.0f, 0.0f, 0.00659634965f, 0.0f, 0.00659634965f,
        0.0f, 0.0f, 0.0833789632f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0123879053f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.357667655f
    };
    constexpr int TIED_ARGMAX = 2;

    check_case("PLAIN", PLAIN_N, PLAIN_K, PLAIN_SURVIVORS, PLAIN_TEMPERATURE, PLAIN_logits,
               PLAIN_softmax, PLAIN_filtered, PLAIN_ARGMAX);
    check_case("HOT", HOT_N, HOT_K, HOT_SURVIVORS, HOT_TEMPERATURE, HOT_logits,
               HOT_softmax, HOT_filtered, HOT_ARGMAX);
    check_case("COLD", COLD_N, COLD_K, COLD_SURVIVORS, COLD_TEMPERATURE, COLD_logits,
               COLD_softmax, COLD_filtered, COLD_ARGMAX);
    check_case("WIDE", WIDE_N, WIDE_K, WIDE_SURVIVORS, WIDE_TEMPERATURE, WIDE_logits,
               WIDE_softmax, WIDE_filtered, WIDE_ARGMAX);
    // k = 6 with a tie at the sixth value: `>=` keeps both, so seven survive.
    check_case("TIED", TIED_N, TIED_K, TIED_SURVIVORS, TIED_TEMPERATURE, TIED_logits,
               TIED_softmax, TIED_filtered, TIED_ARGMAX);
    check(TIED_SURVIVORS > TIED_K, "a tie at the cutoff keeps more than k candidates",
          TIED_SURVIVORS, TIED_K + 1);

    // --- what the draw itself can be held to -------------------------------------------
    {
        const int n = PLAIN_N, k = PLAIN_K;
        std::vector<float> reference(PLAIN_filtered, PLAIN_filtered + n);

        // It never leaves the surviving set. A sampler that ignored the filter would look
        // fine on average and occasionally emit a token the model gave no weight to.
        ac::Rng rng(7);
        std::vector<float> scratch;
        std::vector<int> counts((size_t)n, 0);
        const int draws = 200000;
        for (int i = 0; i < draws; ++i) {
            const int t = ac::sample_top_k(PLAIN_logits, n, k, PLAIN_TEMPERATURE, rng, scratch);
            if (t < 0 || t >= n) {
                std::fprintf(stderr, "FAIL draw %d is outside [0, n)\n", t);
                ++failures;
                break;
            }
            if (reference[(size_t)t] <= 0.0f) {
                std::fprintf(stderr, "FAIL draw %d was filtered out but still sampled\n", t);
                ++failures;
                break;
            }
            ++counts[(size_t)t];
        }

        // And it tracks the distribution. 200k draws over ~6 live tokens puts the standard
        // error near 1e-3, so 0.01 is loose enough never to flake and tight enough to catch
        // an off-by-one in the cumulative walk.
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            const double seen = (double)counts[(size_t)i] / draws;
            worst = std::max(worst, std::fabs(seen - (double)reference[(size_t)i]));
        }
        std::printf("  empirical vs requested distribution: max abs err %.4f over %d draws\n",
                    worst, draws);
        if (worst > 0.01) {
            std::fprintf(stderr, "FAIL sampled distribution is off by %.4f\n", worst);
            ++failures;
        }

        // Reproducible for a seed, which is what `--seed` promises within audiocraft.cpp.
        ac::Rng a(99), b(99);
        std::vector<float> sa, sb;
        for (int i = 0; i < 64; ++i) {
            const int x = ac::sample_top_k(PLAIN_logits, n, k, PLAIN_TEMPERATURE, a, sa);
            const int y = ac::sample_top_k(PLAIN_logits, n, k, PLAIN_TEMPERATURE, b, sb);
            if (x == y) continue;
            check(false, "the same seed draws the same tokens", x, y);
            break;
        }

        // Temperature 0 degenerates to greedy, which is the branch `use_sampling=False` and
        // `temp=0` share in audiocraft.
        ac::Rng z(1);
        check(ac::sample_top_k(PLAIN_logits, n, k, 0.0f, z, scratch) == PLAIN_ARGMAX,
              "temperature 0 is greedy",
              ac::sample_top_k(PLAIN_logits, n, k, 0.0f, z, scratch), PLAIN_ARGMAX);
    }

    // A softmax over huge logits must not overflow: the max-subtracted form is the only
    // reason this is finite.
    {
        std::vector<float> big{1000.0f, 1001.0f, 999.0f, -1000.0f};
        std::vector<float> probs;
        ac::softmax_into(big.data(), (int)big.size(), 1.0f, probs);
        double total = 0.0;
        bool finite = true;
        for (float p : probs) { total += p; finite = finite && std::isfinite(p); }
        check(finite, "a softmax over huge logits stays finite", finite, 1);
        check_close((float)total, 1.0f, 1e-5f, "huge-logit softmax sums to 1", 0);
        check(ac::argmax_token(big.data(), (int)big.size()) == 1, "huge-logit argmax",
              ac::argmax_token(big.data(), (int)big.size()), 1);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("sampling_test: ok\n");
    return 0;
}
