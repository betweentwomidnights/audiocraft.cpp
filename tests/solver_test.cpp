// solver_test -- pin the flow schedule and the KL regularization to audiocraft.
//
// Both are pure host arithmetic, so this needs no checkpoint and no backend -- which is
// the point: the ODE solver's correctness is separable from the DiT's, and this is the
// half that can be pinned exactly.
//
// The schedules are terry's real ones: inversion 1.0 -> 0.12 and generation 0.12 -> 1.0,
// 25 euler steps with sway coefficient -0.8. The regularization fixture deliberately uses
// a frame count that is NOT a multiple of 4, so the tail that `unfold` drops -- and which
// therefore receives no gradient -- is covered.
//
// Regenerate the fixtures with tests/gen_solver_fixture.py (needs terry's venv).
#include "mf/solver.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check_close(float got, float want, float tol, const char* what, int index) {
    if (std::fabs(got - want) <= tol) return;
    std::fprintf(stderr, "FAIL %s[%d]: got %.9g, want %.9g\n", what, index, got, want);
    ++failures;
}

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

} // namespace

int main() {
// --- schedules: terry's inversion (1.0 -> 0.12) and generation (0.12 -> 1.0) ---
    static const float inversion_schedule[] = {
        1.0f, 0.948755443f, 0.897685468f, 0.846963584f, 0.796762288f, 0.747252047f,
        0.698600352f, 0.650971413f, 0.604525447f, 0.559417963f, 0.515799224f, 0.473813504f,
        0.433598816f, 0.395286143f, 0.358998656f, 0.32485202f, 0.292953163f, 0.263400108f,
        0.236281753f, 0.211677343f, 0.189656198f, 0.170277447f, 0.153589785f, 0.139631227f,
        0.128429174f, 0.119999997f
    };
    static const float generation_schedule[] = {
        0.119999997f, 0.128429174f, 0.139631227f, 0.153589785f, 0.170277447f, 0.189656198f,
        0.211677343f, 0.236281753f, 0.263400108f, 0.292953134f, 0.32485202f, 0.358998656f,
        0.395286083f, 0.433598816f, 0.473813504f, 0.515799165f, 0.559417963f, 0.604525447f,
        0.650971413f, 0.698600352f, 0.747252047f, 0.796762288f, 0.846963584f, 0.897685468f,
        0.948755443f, 1.0f
    };
// --- a uniform schedule, sway 0, to pin the mapping without the warp ---
    static const float uniform_schedule[] = {
        0.0f, 0.25f, 0.5f, 0.75f, 1.0f
    };
    constexpr int C = 8, T = 10;
    static const float reg_velocity[] = {
        -0.51079756f, 1.02827179f, -0.35315159f, 0.122991316f, -0.181554466f, -1.49722815f,
        0.142104536f, -0.524282455f, -0.248737112f, -0.52523762f, 2.89221358f, -0.59471339f,
        1.31183314f, 0.352179378f, -1.31512427f, -0.00799220521f, 0.247880027f, 1.57270348f,
        -1.6394645f, -1.59253228f, -0.154625118f, -1.09643877f, 1.36659765f, 0.68928194f,
        -0.393504918f, 0.61710012f, 0.752840221f, 0.602252185f, 2.01754236f, -1.16862941f,
        -1.32419419f, 1.12674356f, -0.225540623f, 0.521760404f, -2.0597744f, 0.138337851f,
        0.496174872f, -0.605279505f, -0.800693989f, 0.158651799f, 0.712770581f,
        -0.743821919f, 0.280077606f, 0.373490721f, 2.99505401f, -0.256943256f,
        -0.683797359f, 1.06211865f, -0.0205178354f, -0.630507588f, 0.548507869f,
        -1.58850551f, 0.528063953f, 0.896397829f, 0.997478426f, -0.215572059f, 1.32659531f,
        -0.108640872f, -1.02650404f, 0.0435771458f, -0.972385883f, 0.280290842f,
        0.569985271f, 1.48411644f, -1.45556903f, 0.558226109f, -0.50624758f, 0.465491265f,
        -0.860426724f, -0.852817237f, 2.01634765f, 0.169276997f, -0.902977705f,
        -1.71020591f, -0.136228889f, 1.45302069f, 0.561969042f, 1.15905154f, -1.72097874f,
        -0.0589746684f
    };
    static const float reg_reference[] = {
        0.845495045f, -0.130360767f, -0.323162973f, -0.711867034f, -0.22731714f,
        -2.53219557f, 0.341463149f, 1.10483027f, 0.0957392976f, -0.0299304835f,
        0.855258822f, -0.0248897914f, 2.26964855f, 0.860169888f, -1.64989245f,
        -0.043485757f, 0.44980073f, -0.550595939f, -0.82031548f, 1.12641799f, -1.23751569f,
        -1.78754508f, 1.01143587f, -0.998236418f, 2.15350509f, 0.463138402f, -1.6792134f,
        1.48996258f, -0.938731432f, 2.02926254f, 0.727409422f, 0.403503209f, 0.764191628f,
        -0.840285361f, 0.346381307f, -2.85904169f, -1.75177789f, -2.95186782f,
        0.0745307282f, 0.281849951f, 0.388998747f, -2.26788282f, -0.695019305f,
        0.854308605f, 0.175839677f, 0.664169252f, -1.17173827f, 1.11464834f, -0.955421567f,
        -1.00547123f, 1.23866868f, 1.90568626f, -0.37328732f, 0.628815293f, 1.3660841f,
        -0.747591078f, 0.631863236f, 0.889452279f, 0.969295204f, 1.33736241f, 0.0667163953f,
        -1.19847059f, -0.0629564524f, -0.352851152f, 0.330958009f, -0.574968755f,
        0.412627429f, -0.378686368f, 1.25368857f, -0.133342117f, -1.35129178f, -2.31663632f,
        0.482513547f, 0.711906433f, 1.28393352f, 1.33429873f, -0.390855908f, -0.843680024f,
        0.255460501f, 0.245594412f
    };
    static const float reg_expected[] = {
        -0.526523948f, 1.00644064f, -0.36950326f, 0.104750998f, -0.200413719f, -1.61602068f,
        0.147829115f, -0.569173932f, -0.248737112f, -0.52523762f, 2.86298895f,
        -0.610106885f, 1.28887713f, 0.333029985f, -1.42008495f, -0.013668376f, 0.26163888f,
        1.68709052f, -1.6394645f, -1.59253228f, -0.17176424f, -1.10984218f, 1.34342444f,
        0.668795407f, -0.428463072f, 0.65890342f, 0.804953754f, 0.642927647f, 2.01754236f,
        -1.16862941f, -1.33669412f, 1.10452175f, -0.242398456f, 0.501938343f, -2.22129583f,
        0.143776327f, 0.528793156f, -0.656323254f, -0.800693989f, 0.158651799f, 0.70725733f,
        -0.780825377f, 0.265209973f, 0.360642582f, 2.84224629f, -0.247562245f,
        -0.653127611f, 1.00571346f, -0.0205178354f, -0.630507588f, 0.539443433f,
        -1.64377034f, 0.518557549f, 0.894854426f, 0.944297016f, -0.208254397f, 1.25699961f,
        -0.106656246f, -1.02650404f, 0.0435771458f, -1.01433063f, 0.265427768f,
        0.561385155f, 1.49527907f, -1.38640833f, 0.526951671f, -0.484432876f, 0.438841909f,
        -0.860426724f, -0.852817237f, 2.03901672f, 0.152013943f, -0.943422019f,
        -1.76810169f, -0.13286835f, 1.37711978f, 0.530507982f, 1.09781194f, -1.72097874f,
        -0.0589746684f
    };

    // --- schedules ---------------------------------------------------------------------
    {
        ac::FlowParams p;
        p.steps = 25;
        p.source_flowstep = 1.0f;
        p.target_flowstep = 0.12f;
        const std::vector<float> got = ac::flow_schedule(p);
        check((int)got.size() == 26, "inversion schedule length", (long long)got.size(), 26);
        for (size_t i = 0; i < got.size(); ++i)
            check_close(got[i], inversion_schedule[i], 1e-6f, "inversion", (int)i);
        check(p.inverting(), "1.0 -> 0.12 is an inversion", p.inverting(), 1);
        check(p.effective_cfg_coef() == 0.0f, "inversion forces guidance off",
              (long long)(p.effective_cfg_coef() * 1000), 0);
    }
    {
        ac::FlowParams p;
        p.steps = 25;
        p.source_flowstep = 0.12f;
        p.target_flowstep = 1.0f;
        p.cfg_coef = 4.0f;
        const std::vector<float> got = ac::flow_schedule(p);
        check((int)got.size() == 26, "generation schedule length", (long long)got.size(), 26);
        for (size_t i = 0; i < got.size(); ++i)
            check_close(got[i], generation_schedule[i], 1e-6f, "generation", (int)i);
        check(!p.inverting(), "0.12 -> 1.0 is a generation", p.inverting(), 0);
        check(p.effective_cfg_coef() == 4.0f, "generation keeps guidance",
              (long long)(p.effective_cfg_coef() * 1000), 4000);
    }
    {
        // Without the sway warp the schedule is just a linear ramp, which pins the
        // source/target mapping independently of the warp.
        ac::FlowParams p;
        p.steps = 4;
        p.sway_coefficient = 0.0f;
        const std::vector<float> got = ac::flow_schedule(p);
        for (size_t i = 0; i < got.size(); ++i)
            check_close(got[i], uniform_schedule[i], 1e-6f, "uniform", (int)i);
    }
    {
        // Whichever direction, the schedule starts at the source and ends at the target.
        for (int steps : {2, 8, 25, 64}) {
            for (int flip = 0; flip < 2; ++flip) {
                ac::FlowParams p;
                p.steps = steps;
                p.source_flowstep = flip ? 1.0f : 0.0f;
                p.target_flowstep = flip ? 0.37f : 1.0f;
                const std::vector<float> s = ac::flow_schedule(p);
                check_close(s.front(), p.source_flowstep, 1e-6f, "starts at source", steps);
                check_close(s.back(), p.target_flowstep, 1e-6f, "ends at target", steps);
                for (size_t i = 1; i < s.size(); ++i) {
                    const bool monotone = flip ? (s[i] <= s[i - 1]) : (s[i] >= s[i - 1]);
                    if (!monotone) {
                        std::fprintf(stderr, "FAIL schedule not monotone at %zu (steps %d)\n",
                                     i, steps);
                        ++failures;
                    }
                }
            }
        }
    }

    // --- forward accounting -------------------------------------------------------------
    {
        // terry's inversion: 25 steps, guidance off, regularize_iters 2 with the KL on the
        // last one -- one forward for the plain iteration, two for the regularized one.
        ac::FlowParams inv;
        inv.steps = 25;
        inv.source_flowstep = 1.0f;
        inv.target_flowstep = 0.12f;
        inv.regularize = true;
        inv.regularize_iters = 2;
        inv.keep_last_k_iters = 1;
        check(ac::flow_forward_count(inv) == 75, "inversion forwards",
              ac::flow_forward_count(inv), 75);

        // terry's generation: 25 steps, guidance on, no regularization -- two forwards each.
        ac::FlowParams gen;
        gen.steps = 25;
        gen.source_flowstep = 0.12f;
        gen.target_flowstep = 1.0f;
        check(ac::flow_forward_count(gen) == 50, "generation forwards",
              ac::flow_forward_count(gen), 50);
    }

    // --- KL regularization ---------------------------------------------------------------
    {
        std::vector<float> velocity(reg_velocity, reg_velocity + C * T);
        const std::vector<float> reference(reg_reference, reg_reference + C * T);
        const std::vector<float> got =
            ac::noise_regularization(velocity, reference, C, T, 0.2f, 4);
        check((int)got.size() == C * T, "regularized size", (long long)got.size(), C * T);
        double worst = 0.0;
        for (int i = 0; i < C * T; ++i) {
            const double err = std::fabs((double)got[i] - (double)reg_expected[i]);
            if (err > worst) worst = err;
            check_close(got[i], reg_expected[i], 2e-5f, "regularized", i);
        }
        std::printf("  kl regularization max abs err %.3e\n", worst);

        // The last T % 4 frames fall outside every patch, so `unfold` never sees them and
        // they must come back untouched. That is faithful, not an oversight.
        for (int c = 0; c < C; ++c) {
            for (int t = (T / 4) * 4; t < T; ++t) {
                const int i = c * T + t;
                check_close(got[i], reg_velocity[i], 0.0f, "untouched tail", i);
            }
        }
    }
    {
        // lambda_kl == 0 is the identity, which is the path a non-regularized solve takes.
        std::vector<float> velocity(reg_velocity, reg_velocity + C * T);
        const std::vector<float> reference(reg_reference, reg_reference + C * T);
        const std::vector<float> got =
            ac::noise_regularization(velocity, reference, C, T, 0.0f, 4);
        for (int i = 0; i < C * T; ++i)
            check_close(got[i], reg_velocity[i], 0.0f, "lambda_kl=0 identity", i);
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("solver_test: ok\n");
    return 0;
}
