#include "ac/train_controls.h"
#include <cmath>

static void require(bool value) { if(!value) throw std::runtime_error("optimizer invariant failed"); }

int main() {
    ac::TrainControls c;
    c.warmup = 4;
    c.validate();
    require(c.rate(1, 1) == .25f && c.rate(1, 4) == 1 && c.rate(1, 5) == 1);
    std::vector<ac::MfTrainParam> p(2);
    p[0].name = "weight.lora_A";
    p[1].name = "weight.magnitude";
    for (auto &x : p) {
        x.value = {2};
        x.m = {0};
        x.v = {0};
    }
    ac::TrainAccumulation a;
    a.count = 2;
    a.gradients = {{6}, {8}};
    c.beta1 = 0;
    c.beta2 = 0;
    c.clip = 1;
    c.decay = .1f;
    a.update(p, c, 1, .1f);
    // Mean gradients (3,4), clipped globally to (.6,.8).
    require(std::fabs(p[0].m[0] - .6f) < 1e-6f);
    require(std::fabs(p[1].m[0] - .8f) < 1e-6f);
    require(std::fabs(p[0].value[0] - 1.88f) < 1e-6f);
    require(std::fabs(p[1].value[0] - 1.9f) < 1e-6f);
    require(a.count == 0 && a.gradients.empty());
    return 0;
}
