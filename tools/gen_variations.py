#!/usr/bin/env python3
"""Generate src/mf/variations.h from terry's variations.py.

  python tools/gen_variations.py --src .../services/melodyflow/variations.py \
      > src/mf/variations.h

Transcribing 34 presets by hand is how a prompt quietly loses a word, so they are generated.
The table is the service's, verbatim.
"""
import argparse
import runpy
import sys


def cstr(s):
    out = []
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\n")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to terry's variations.py")
    args = ap.parse_args()
    variations = runpy.run_path(args.src)["VARIATIONS"]

    print("// mf/variations.h -- terry's transform presets.")
    print("//")
    print("// Generated from the service's variations.py by tools/gen_variations.py; do not")
    print("// edit by hand. Each preset is a target prompt plus the flow step to invert to,")
    print("// which is the only per-preset parameter -- everything else is the same euler/25")
    print("// setting for all of them.")
    print("#pragma once")
    print()
    print("#include <cstddef>")
    print()
    print("namespace ac {")
    print()
    print("struct Variation {")
    print("    const char* name;")
    print("    const char* prompt;")
    # double, not float: /variations echoes this straight back to the client, and 0.12f
    # renders as 0.119999997 through %.9g.
    print("    double flowstep;")
    print("    int steps;")
    print("};")
    print()
    print("inline const Variation* variations(size_t& count) {")
    print("    static const Variation table[] = {")
    for name, v in variations.items():
        flow = float(v["default_flowstep"])
        steps = int(v["steps"])
        print(f"        {{{cstr(name)},")
        print(f"         {cstr(v['prompt'])},")
        print(f"         {flow:g}, {steps}}},")
    print("    };")
    print("    count = sizeof(table) / sizeof(table[0]);")
    print("    return table;")
    print("}")
    print()
    print("// The preset by name, or null. Callers may still override the flow step and the")
    print("// prompt: terry's /transform takes custom_flowstep and custom_prompt alongside a")
    print("// preset name, and a custom prompt replaces the preset's rather than adding to it.")
    print("inline const Variation* find_variation(const char* name) {")
    print("    size_t count = 0;")
    print("    const Variation* table = variations(count);")
    print("    for (size_t i = 0; i < count; ++i) {")
    print("        const char* a = table[i].name;")
    print("        const char* b = name;")
    print("        while (*a && *a == *b) { ++a; ++b; }")
    print("        if (!*a && !*b) return &table[i];")
    print("    }")
    print("    return nullptr;")
    print("}")
    print()
    print("} // namespace ac")
    return 0


if __name__ == "__main__":
    sys.exit(main())
