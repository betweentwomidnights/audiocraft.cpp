// conditioner_test -- pin what counts as an empty description, and what that costs.
//
// This one is here because the parity suite missed it for an entire phase. Every reference
// comparison used a real prompt, and with a real prompt we are token-exact against torch. Run
// the same comparison with no description and we diverged at the *first* generated token --
// which is gary's default for twelve of the fourteen `thepatch` models, since only the two
// `gary_orchestra` checkpoints carry a description of their own.
//
// The rule, from audiocraft/modules/conditioners.py::T5Conditioner:
//
//     entries   = [xi if xi is not None else "" for xi in x]
//     empty_idx = [i for i, xi in enumerate(entries) if xi == ""]
//     mask[empty_idx, :] = 0
//     embeds = output_proj(t5(...)) * mask.unsqueeze(-1)
//
// So an empty description is not "T5's encoding of an empty string". It is an all-zero
// context, identical to the null branch -- which makes classifier-free guidance a no-op,
// because `uncond + (cond - uncond) * coef` with `cond == uncond` is just `uncond`.
//
// Two ways to get this wrong, both silent and both plausible:
//   * trim the prompt before the test, so " " is treated as empty and loses its guidance;
//   * guide anyway, and steer every unprompted generation toward the EOS token's embedding.
#include "ac/conditioner.h"
#include "mg/pipeline.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool got, bool want, const char* what) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: got %s, want %s\n", what,
                 got ? "true" : "false", want ? "true" : "false");
    ++failures;
}

void empty_is_exactly_the_empty_string() {
    // audiocraft compares against "" and nothing else; `normalize_text` is false in both
    // checkpoints and `word_dropout` only fires while training, so nothing rewrites a
    // description on the way in.
    check(ac::description_is_empty(""), true, "\"\" is empty");
    check(ac::description_is_empty(" "), false, "a single space is a description");
    check(ac::description_is_empty("\t"), false, "a tab is a description");
    check(ac::description_is_empty("  \n "), false, "whitespace is a description");
    check(ac::description_is_empty("drum and bass"), false, "a real prompt is not empty");
    check(ac::description_is_empty("a"), false, "one character is not empty");
}

void an_empty_description_turns_guidance_off() {
    // The whole point: with no description there is nothing to guide towards, and running
    // the two streams anyway does not just waste half the compute, it disagrees with torch.
    check(ac::mg_use_guidance(3.0f, /*description_empty=*/true), false,
          "cfg 3.0 with an empty description does not guide");
    check(ac::mg_use_guidance(3.0f, /*description_empty=*/false), true,
          "cfg 3.0 with a description guides");

    // cfg 0 is the explicit off switch and stays off either way.
    check(ac::mg_use_guidance(0.0f, /*description_empty=*/false), false,
          "cfg 0 with a description does not guide");
    check(ac::mg_use_guidance(0.0f, /*description_empty=*/true), false,
          "cfg 0 with an empty description does not guide");
}

} // namespace

int main() {
    empty_is_exactly_the_empty_string();
    an_empty_description_turns_guidance_off();
    if (failures) {
        std::fprintf(stderr, "conditioner_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("conditioner_test: ok\n");
    return 0;
}
