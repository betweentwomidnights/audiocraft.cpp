// serve_test -- the plumbing under the two HTTP services.
//
// None of this is clever, and all of it is the kind of thing that fails quietly: a base64
// encoder that mishandles the last two bytes truncates every request's audio, a JSON escaper
// that misses a control character produces a body the client cannot parse, and a session
// registry that never forgets holds a few megabytes of WAV per request forever.
//
// The audio path is checked as a round trip through the same helpers the servers use, so a
// change to either side has to keep them consistent.
#include "serve/http.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

void check_eq(const std::string& got, const std::string& want, const char* what) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: got %s, want %s\n", what, got.c_str(), want.c_str());
    ++failures;
}

} // namespace

int main() {
    using namespace ac::serve;

    // --- base64 -------------------------------------------------------------------------
    // The RFC 4648 vectors, which cover all three padding cases.
    check_eq(b64_encode(""), "", "b64 empty");
    check_eq(b64_encode("f"), "Zg==", "b64 one byte");
    check_eq(b64_encode("fo"), "Zm8=", "b64 two bytes");
    check_eq(b64_encode("foo"), "Zm9v", "b64 three bytes");
    check_eq(b64_encode("foob"), "Zm9vYg==", "b64 four bytes");
    check_eq(b64_encode("fooba"), "Zm9vYmE=", "b64 five bytes");
    check_eq(b64_encode("foobar"), "Zm9vYmFy", "b64 six bytes");
    for (const char* s : {"", "f", "fo", "foo", "foob", "fooba", "foobar"})
        check_eq(b64_decode(b64_encode(s)), s, "b64 round trip");

    // Every byte value, since audio is binary and 0x00 / 0xff are not special.
    {
        std::string all;
        for (int i = 0; i < 256; ++i) all += (char)i;
        check_eq(b64_decode(b64_encode(all)), all, "b64 covers every byte value");
    }

    // What a browser or a careless client actually sends.
    check_eq(b64_decode("Zm9v\nYmFy"), "foobar", "b64 ignores newlines");
    check_eq(b64_decode("data:audio/wav;base64,Zm9vYmFy"), "foobar",
             "b64 strips a data: prefix");
    check_eq(b64_decode("Zm9vYmFy"), "foobar", "b64 without padding");
    {
        bool threw = false;
        try { b64_decode("not base64!"); } catch (const std::exception&) { threw = true; }
        check(threw, "b64 rejects a bad character");
    }

    // --- JSON ---------------------------------------------------------------------------
    check_eq(json_escape("a\"b\\c"), "a\\\"b\\\\c", "json escapes quotes and backslashes");
    check_eq(json_escape("a\nb\tc"), "a\\nb\\tc", "json escapes newline and tab");
    check_eq(json_escape(std::string("a\x01" "b")), "a\\u0001b", "json escapes control bytes");
    check_eq(json_number(0.12), "0.12", "json renders a double readably");
    check_eq(json_number(std::nan("")), "null", "json renders NaN as null");
    check_eq(JsonObject().boolean("success", true).num("progress", 42)
                         .str("status", "ok").str(),
             "{\"success\":true,\"progress\":42,\"status\":\"ok\"}", "json object");
    // An error message with a quote in it must not break the body it lands in.
    check_eq(json_error("bad \"model\""),
             "{\"success\":false,\"error\":\"bad \\\"model\\\"\"}", "json error escaping");

    // --- audio round trip ---------------------------------------------------------------
    {
        const int n = 1000, ch = 2, rate = 32000;
        std::vector<float> planar((size_t)n * ch);
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < n; ++i)
                planar[(size_t)c * n + i] = (c ? -1.0f : 1.0f) * std::sin(i * 0.05f) * 0.8f;

        const std::string b64 = encode_audio(planar, n, ch, rate);
        int got_n = 0, got_ch = 0, got_rate = 0;
        const std::vector<float> back = decode_audio(b64, got_n, got_ch, got_rate);
        check(got_n == n && got_ch == ch && got_rate == rate, "audio round trip shape");
        double worst = 0.0;
        for (size_t i = 0; i < planar.size() && i < back.size(); ++i)
            worst = std::max(worst, std::fabs((double)planar[i] - back[i]));
        // 16-bit PCM, which is what both Python services write, so a step is 1/32768.
        check(worst < 1.0 / 16000.0, "audio round trip is within 16-bit quantization");
        std::printf("  audio round trip max abs err %.3e (16-bit step %.3e)\n",
                    worst, 1.0 / 32768.0);

        bool threw = false;
        try { int a = 0, b = 0, c = 0; decode_audio("AAAA", a, b, c); }
        catch (const std::exception&) { threw = true; }
        check(threw, "a payload too short to be a WAV is rejected");
    }

    // --- sessions -------------------------------------------------------------------------
    {
        Sessions sessions(4);
        Session initial;
        initial.seed = 7;
        initial.original_b64 = "abc";
        const std::string id = sessions.create(initial);
        check(id.size() == 32, "session ids are 32 hex characters");

        Session s;
        check(sessions.get(id, s) && s.seed == 7 && s.status == "queued",
              "session round trip");
        check(!sessions.get("nope", s), "an unknown session is absent");

        sessions.set_progress(id, 3, 4);
        sessions.get(id, s);
        check(s.progress == 75 && s.status == "processing",
              "progress becomes a percentage");
        // Progress is a percentage of the caller's own total, so it must clamp rather than
        // report 137% when one miscounts.
        sessions.set_progress(id, 9, 4);
        sessions.get(id, s);
        check(s.progress == 100, "progress clamps at 100");
        sessions.set_progress(id, 1, 0);
        sessions.get(id, s);
        check(s.progress == 0, "a zero total is 0%, not a division by zero");

        sessions.fail(id, "boom");
        sessions.get(id, s);
        check(s.status == "failed" && s.error == "boom", "failure is recorded");

        // The oldest sessions are dropped: each finished one holds a base64 WAV, which for a
        // 30 s stereo result is about 8 MB of string.
        std::string newest;
        for (int i = 0; i < 6; ++i) newest = sessions.create({});
        check(!sessions.get(id, s), "the oldest session is swept once the registry is full");
        check(sessions.get(newest, s), "the newest session survives");
    }

    // --- seeds ---------------------------------------------------------------------------
    check(resolve_seed(42) == 42, "an explicit seed is kept");
    for (int i = 0; i < 200; ++i) {
        if (resolve_seed(-1) < 100000) continue;
        check(false, "a requested seed of -1 stays in the readable range");
        break;
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("serve_test: ok\n");
    return 0;
}
