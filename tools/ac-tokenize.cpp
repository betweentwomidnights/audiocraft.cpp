// ac-tokenize -- print the T5 token ids audiocraft's T5Conditioner would produce.
//
// audiocraft calls `T5Tokenizer.from_pretrained("t5-base")(text, padding=True)`, which
// SentencePiece-encodes the text and appends </s>. With a single prompt there is nothing
// to pad to, so the id sequence printed here is exactly what the reference produces.
#include "ac/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage() {
    fprintf(stderr,
            "usage: ac-tokenize --t5 <t5.gguf> --prompt <text> [--max-length N]\n"
            "\n"
            "  --max-length N   truncate to N ids (0 = no limit, the audiocraft default)\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string t5_path;
    std::vector<std::string> prompts;
    int max_length = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--t5") t5_path = next("--t5");
        else if (a == "--prompt") prompts.push_back(next("--prompt"));
        else if (a == "--max-length") max_length = std::atoi(next("--max-length").c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (t5_path.empty()) { fprintf(stderr, "error: --t5 is required\n"); usage(); return 2; }
    if (prompts.empty()) { fprintf(stderr, "error: at least one --prompt is required\n"); usage(); return 2; }

    try {
        const ac::UnigramTokenizer tokenizer = ac::UnigramTokenizer::load(t5_path.c_str());
        for (const std::string& prompt : prompts) {
            const std::vector<int32_t> ids = tokenizer.encode(prompt, max_length);
            printf("%zu:", ids.size());
            for (int32_t id : ids) printf(" %d", id);
            printf("\n");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
