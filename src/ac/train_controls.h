#pragma once
#include "ac/train_utils.h"
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>

namespace ac {
// Shared host optimizer: average microbatches BEFORE global clipping and AdamW.
struct TrainControls {
    int accumulation = 1, warmup = 0;
    float beta1 = .9f, beta2 = .999f, eps = 1e-8f, decay = 0, clip = 1;
    std::string schedule = "constant", resume;
    bool feedforward = false, random_crop = false, eval_null = false;
    template <class Next> bool parse(const std::string &a, Next next) {
        if (a == "--accumulation-steps")
            accumulation = std::stoi(next());
        else if (a == "--warmup-steps")
            warmup = std::stoi(next());
        else if (a == "--lr-schedule")
            schedule = next();
        else if (a == "--beta1")
            beta1 = std::stof(next());
        else if (a == "--beta2")
            beta2 = std::stof(next());
        else if (a == "--adam-eps")
            eps = std::stof(next());
        else if (a == "--weight-decay")
            decay = std::stof(next());
        else if (a == "--grad-clip")
            clip = std::stof(next());
        else if (a == "--train-ffn")
            feedforward = true;
        else if (a == "--random-crop")
            random_crop = true;
        else if (a == "--eval-unconditional")
            eval_null = true;
        else if (a == "--resume")
            resume = next();
        else
            return false;
        return true;
    }
    void validate() const {
        if (accumulation <= 0 || warmup < 0 || !std::isfinite(beta1) || beta1 < 0 || beta1 >= 1 ||
            !std::isfinite(beta2) || beta2 < 0 || beta2 >= 1 || !std::isfinite(eps) || eps <= 0 ||
            !std::isfinite(decay) || decay < 0 || !std::isfinite(clip) || clip <= 0 ||
            (schedule != "constant" && schedule != "inverse"))
            throw std::runtime_error("invalid optimizer controls");
    }
    float rate(float lr, int step) const {
        float factor = warmup ? std::min(1.f, float(step) / warmup) : 1.f;
        if (schedule == "inverse")
            factor *= std::pow(1.f + float(step - 1) / 1000000.f, -.5f);
        return lr * factor;
    }
    std::string signature(float lr) const {
        std::ostringstream o;
        o << std::setprecision(9) << accumulation << ' ' << warmup << ' ' << beta1 << ' ' << beta2 << ' '
          << eps << ' ' << decay << ' ' << clip << ' ' << schedule << ' ' << feedforward << ' ' << random_crop
          << ' ' << lr;
        return o.str();
    }
    static const char *help() {
        return "  --accumulation-steps N --warmup-steps N --lr-schedule constant|inverse\n"
               "  --beta1 F --beta2 F --adam-eps F --weight-decay F --grad-clip F\n"
               "  --train-ffn (also adapt feed-forward projections)\n"
               "  --random-crop (fresh training crops; validation center crops)\n"
               "  --eval-unconditional (null-text probes) --resume FILE.state\n";
    }
};

struct TrainAccumulation {
    std::vector<std::vector<float>> gradients;
    int count = 0;
    void add(const std::vector<MfTrainParam> &params) {
        if (gradients.empty())
            for (const auto &p : params)
                gradients.emplace_back(p.value.size(), 0.f);
        for (size_t j = 0; j < params.size(); ++j) {
            auto g = train_read_tensor(params[j].grad);
            for (size_t i = 0; i < g.size(); ++i)
                gradients[j][i] += g[i];
        }
        ++count;
    }
    void update(std::vector<MfTrainParam> &params, const TrainControls &c, int step, float lr) {
        if (count <= 0)
            throw std::runtime_error("empty gradient accumulation");
        double norm = 0;
        for (auto &g : gradients)
            for (auto &x : g) {
                x /= count;
                norm += double(x) * x;
            }
        const float scale = float(std::min(1., double(c.clip) / (std::sqrt(norm) + 1e-12)));
        float bc1 = 1 - std::pow(c.beta1, float(step)), bc2 = 1 - std::pow(c.beta2, float(step));
        for (size_t j = 0; j < params.size(); ++j) {
            auto &p = params[j];
            // Magnitudes describe base row norms: never decay them toward zero.
            const float wd =
                p.name.find(".lora_A") != std::string::npos || p.name.find(".lora_B") != std::string::npos
                    ? c.decay
                    : 0;
            for (size_t i = 0; i < p.value.size(); ++i) {
                float g = scale * gradients[j][i];
                p.m[i] = c.beta1 * p.m[i] + (1 - c.beta1) * g;
                p.v[i] = c.beta2 * p.v[i] + (1 - c.beta2) * g * g;
                p.value[i] -= lr * ((p.m[i] / bc1) / (std::sqrt(p.v[i] / bc2) + c.eps) + wd * p.value[i]);
                if (!std::isfinite(p.value[i]))
                    throw std::runtime_error("nonfinite optimizer update");
            }
        }
        gradients.clear();
        count = 0;
    }
};

// Portable, versioned state written only at completed optimizer-update boundaries.
inline void train_state(const std::string &path, bool load, const std::string &signature,
                        std::vector<MfTrainParam> &params, int &step, size_t &seen, std::mt19937 &rng,
                        std::vector<size_t> &order) {
    std::fstream f;
    if (!load && std::filesystem::exists(path))
        throw std::runtime_error("trainer state already exists");
    const std::string temporary = path + ".tmp";
    f.open(load ? path : temporary, load ? std::ios::in : std::ios::out);
    if (!f)
        throw std::runtime_error("cannot open trainer state: " + path);
    f << std::setprecision(std::numeric_limits<float>::max_digits10);
    auto scalar = [&](auto &v) {
        if (load)
            f >> v;
        else
            f << v << '\n';
    };
    std::string tag = "ac-trainer-state-v1", sig = signature;
    if (load) {
        f >> std::quoted(tag) >> std::quoted(sig);
        if (tag != "ac-trainer-state-v1" || sig != signature)
            throw std::runtime_error("trainer state configuration/data mismatch");
    } else
        f << std::quoted(tag) << '\n' << std::quoted(sig) << '\n';
    scalar(step);
    scalar(seen);
    scalar(rng);
    size_t n = order.size();
    scalar(n);
    if (n != order.size())
        throw std::runtime_error("trainer state order mismatch");
    for (auto &v : order)
        scalar(v);
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size(); ++i)
        if (sorted[i] != i)
            throw std::runtime_error("invalid trainer sample order");
    n = params.size();
    scalar(n);
    if (n != params.size())
        throw std::runtime_error("trainer state parameter count mismatch");
    for (auto &p : params) {
        std::string name = p.name;
        if (load) {
            f >> std::quoted(name);
            if (name != p.name)
                throw std::runtime_error("trainer state parameter mismatch");
        } else
            f << std::quoted(name) << '\n';
        n = p.value.size();
        scalar(n);
        if (n != p.value.size())
            throw std::runtime_error("trainer state shape mismatch");
        for (auto *vec : {&p.value, &p.m, &p.v})
            for (auto &x : *vec) {
                scalar(x);
                if (!std::isfinite(x))
                    throw std::runtime_error("nonfinite trainer state");
            }
    }
    if (!f)
        throw std::runtime_error("truncated/unwritable trainer state");
    if (load) {
        f >> std::ws;
        if (!f.eof())
            throw std::runtime_error("trailing trainer state data");
    } else {
        f.flush();
        if (!f)
            throw std::runtime_error("trainer state flush failed");
        f.close();
        std::filesystem::rename(temporary, path);
    }
}

template <class... T> inline std::string train_key(const T &...values) {
    std::ostringstream o;
    o << std::setprecision(std::numeric_limits<float>::max_digits10);
    ((o << values << '|'), ...);
    return o.str();
}

inline std::string train_file_identity(const std::string &file) {
    std::ifstream f(file, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot fingerprint " + file);
    uint64_t h = 14695981039346656037ull;
    char buf[65536];
    while (f) {
        f.read(buf, sizeof(buf));
        for (std::streamsize i = 0; i < f.gcount(); ++i) {
            h ^= (unsigned char)buf[i];
            h *= 1099511628211ull;
        }
    }
    if (!f.eof())
        throw std::runtime_error("fingerprint read failed");
    return std::to_string(h);
}
} // namespace ac
