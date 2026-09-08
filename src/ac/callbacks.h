// ac/callbacks.h -- the two callback shapes both families use.
//
// Deliberately free of ggml and of everything else: `mf/solver.h` is pure host arithmetic
// and testable without a backend, which is worth keeping, and `mg/pipeline.h` needs the
// same progress shape without inheriting MelodyFlow's solver.
#pragma once

#include <cstddef>
#include <functional>

namespace ac {

// (elapsed, total). Both services report progress this way, and the counts are the ones the
// Python side sends so a client's progress bar does not change behaviour.
using ProgressFn = std::function<void(int done, int total)>;

// Fills `n` floats with standard normal noise. Injecting it rather than owning it is what
// makes exact parity testable: torch's stream cannot be reproduced here, so both sides read
// the same dump instead.
using NoiseFn = std::function<void(float* dst, size_t n)>;

} // namespace ac
