// t5_bucket_test -- pin ac::t5_relative_position_bucket to transformers' reference values.
//
// T5's learned relative-attention bias is the one place where a silent off-by-one produces
// plausible-but-wrong conditioning, so the expected table below was generated directly from
// transformers.T5Attention._relative_position_bucket with bidirectional=True,
// num_buckets=32, max_distance=128 -- the exact settings in t5-base's config.
#include "ac/t5.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, long long got, long long want) {
    if (ok) return;
    fprintf(stderr, "FAIL %s: got %lld, want %lld\n", what, got, want);
    ++failures;
}

} // namespace

int main() {
    struct Case { int relative_position; int bucket; };
    // relative_position = key_position - query_position.
    static const Case cases[] = {
        {-256, 15}, {-129, 15}, {-128, 15}, {-127, 15}, {-64, 14}, {-33, 12}, {-32, 12},
        {-17, 10},  {-16, 10},  {-15, 9},   {-8, 8},    {-7, 7},   {-1, 1},   {0, 0},
        {1, 17},    {7, 23},    {8, 24},    {15, 25},   {16, 26},  {17, 26},  {32, 28},
        {33, 28},   {64, 30},   {127, 31},  {128, 31},  {129, 31}, {256, 31},
    };
    for (const Case& c : cases) {
        const int got = ac::t5_relative_position_bucket(c.relative_position, 32, 128);
        char what[64];
        snprintf(what, sizeof(what), "bucket(%d)", c.relative_position);
        check(got == c.bucket, what, got, c.bucket);
    }

    // Every bucket must be in range, and the sign split must land on the two halves.
    for (int rp = -300; rp <= 300; ++rp) {
        const int b = ac::t5_relative_position_bucket(rp, 32, 128);
        if (b < 0 || b >= 32) {
            fprintf(stderr, "FAIL bucket(%d) = %d out of [0, 32)\n", rp, b);
            ++failures;
        }
        const bool upper_half = b >= 16;
        if (upper_half != (rp > 0)) {
            fprintf(stderr, "FAIL bucket(%d) = %d on the wrong half\n", rp, b);
            ++failures;
        }
    }

    // The flattened matrix is indexed [query][key] and holds bucket(key - query), which is
    // what ggml_get_rows + reshape_3d(heads, seq, seq) expects before the permute to
    // [key, query, heads].
    const int seq = 9;
    const std::vector<int32_t> flat = ac::t5_relative_position_buckets(seq, 32, 128);
    check((int)flat.size() == seq * seq, "matrix size", (long long)flat.size(),
          (long long)seq * seq);
    for (int q = 0; q < seq; ++q) {
        for (int k = 0; k < seq; ++k) {
            const int want = ac::t5_relative_position_bucket(k - q, 32, 128);
            const int got = flat[(size_t)q * seq + k];
            if (got != want) {
                fprintf(stderr, "FAIL matrix[q=%d][k=%d]: got %d, want %d\n", q, k, got, want);
                ++failures;
            }
        }
    }
    // The diagonal is bucket(0) = 0 for every position.
    for (int q = 0; q < seq; ++q)
        check(flat[(size_t)q * seq + q] == 0, "diagonal", flat[(size_t)q * seq + q], 0);

    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("t5_bucket_test: ok\n");
    return 0;
}
