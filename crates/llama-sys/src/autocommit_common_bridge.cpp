#include "autocommit_common_bridge.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#include "arg.h"
#include "common.h"
#include "sampling.h"

// common.h defines a global `build_info` string derived from these variables.
// In this integration path libcommon.a is linked directly from the build tree
// (where build_info object may not be installed), so provide safe fallbacks.
int LLAMA_BUILD_NUMBER = 0;
const char * LLAMA_COMMIT = "unknown";
const char * LLAMA_COMPILER = "unknown";
const char * LLAMA_BUILD_TARGET = "unknown";

struct autocommit_common_config {
    common_params params;
};

struct autocommit_common_sampler {
    common_sampler * sampler;
};

namespace {

void write_error(char * err, const size_t err_len, const std::string & msg) {
    if (err == nullptr || err_len == 0) {
        return;
    }

    const size_t write_len = std::min(err_len - 1, msg.size());
    std::memcpy(err, msg.data(), write_len);
    err[write_len] = '\0';
}

bool parse_bool_or_throw(const std::string & value) {
    if (common_arg_utils::is_truthy(value)) {
        return true;
    }
    if (common_arg_utils::is_falsey(value)) {
        return false;
    }
    throw std::invalid_argument("invalid boolean value");
}

void finalize_env_only_params(common_params & params) {
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.speculative.cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.speculative.cpuparams_batch, &params.cpuparams_batch);

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }
    if (!params.tensor_buft_overrides.empty() &&
        params.tensor_buft_overrides.back().pattern != nullptr) {
        params.tensor_buft_overrides.push_back(
            llama_model_tensor_buft_override {
                /* pattern = */ nullptr,
                /* buft    = */ nullptr,
            });
    }
}

} // namespace

extern "C" {

autocommit_common_config * autocommit_common_config_new(void) {
    auto * cfg = new (std::nothrow) autocommit_common_config();
    if (cfg == nullptr) {
        return nullptr;
    }

    // Startup-focused defaults for autocommit:
    // - avoid full-train context allocations by default
    // - keep batch memory moderate for short analysis prompts
    // - leave room for UI responsiveness on desktop systems
    cfg->params.n_gpu_layers = -1;
    cfg->params.n_ctx        = 8192;
    cfg->params.n_parallel   = 1;
    cfg->params.n_batch      = 1024;
    cfg->params.n_ubatch     = 256;
    cfg->params.sampling.top_p = 0.90f;
    cfg->params.sampling.temp = 0.20f;
    cfg->params.sampling.min_p = 0.0f;

    const unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 0) {
        int n_threads = static_cast<int>(hw_threads);
        if (n_threads > 4) {
            n_threads -= 2;
        }
        n_threads = std::max(1, n_threads);
        cfg->params.cpuparams.n_threads = n_threads;
        cfg->params.cpuparams_batch.n_threads = n_threads;
    }

    return cfg;
}

void autocommit_common_config_free(autocommit_common_config * cfg) {
    delete cfg;
}

void autocommit_common_config_set_model_path(autocommit_common_config * cfg, const char * path) {
    if (cfg == nullptr) {
        return;
    }
    cfg->params.model.path = path != nullptr ? path : "";
}

void autocommit_common_config_set_n_parallel(autocommit_common_config * cfg, const int32_t n_parallel) {
    if (cfg == nullptr || n_parallel <= 0) {
        return;
    }
    cfg->params.n_parallel = n_parallel;
}

int autocommit_common_config_apply_env(
        autocommit_common_config * cfg,
        char * err,
        const size_t err_len) {
    if (cfg == nullptr) {
        write_error(err, err_len, "common config is null");
        return 0;
    }

    try {
        const int32_t n_parallel_seed = cfg->params.n_parallel > 0 ? cfg->params.n_parallel : 1;
        auto ctx = common_params_parser_init(cfg->params, LLAMA_EXAMPLE_SERVER, nullptr);

        // Apply only environment-bound options to avoid CLI/remote-preset side effects.
        for (auto & opt : ctx.options) {
            std::string value;
            if (!opt.get_value_from_env(value)) {
                continue;
            }

            if (opt.handler_void && common_arg_utils::is_truthy(value)) {
                opt.handler_void(cfg->params);
            }
            if (opt.handler_int) {
                opt.handler_int(cfg->params, std::stoi(value));
            }
            if (opt.handler_bool) {
                opt.handler_bool(cfg->params, parse_bool_or_throw(value));
            }
            if (opt.handler_string) {
                opt.handler_string(cfg->params, value);
            }
        }

        // LLAMA_EXAMPLE_SERVER sets n_parallel to -1 (auto) by default.
        // For this runtime we keep explicit positive values only.
        if (cfg->params.n_parallel <= 0) {
            cfg->params.n_parallel = n_parallel_seed;
        }

        finalize_env_only_params(cfg->params);
        return 1;
    } catch (const std::exception & ex) {
        write_error(err, err_len, ex.what());
        return 0;
    }
}

int autocommit_common_config_export_llama_params(
        autocommit_common_config * cfg,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        char * err,
        const size_t err_len) {
    if (cfg == nullptr || mparams == nullptr || cparams == nullptr) {
        write_error(err, err_len, "invalid null pointer passed to export params");
        return 0;
    }

    try {
        *mparams = common_model_params_to_llama(cfg->params);
        *cparams = common_context_params_to_llama(cfg->params);
        return 1;
    } catch (const std::exception & ex) {
        write_error(err, err_len, ex.what());
        return 0;
    }
}

int autocommit_common_config_fill_fit_buffers(
        autocommit_common_config * cfg,
        float * tensor_split,
        const size_t tensor_split_len,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        const size_t tensor_buft_overrides_len,
        size_t * margins,
        const size_t margins_len,
        char * err,
        const size_t err_len) {
    if (cfg == nullptr || tensor_split == nullptr || tensor_buft_overrides == nullptr || margins == nullptr) {
        write_error(err, err_len, "invalid null pointer passed to fit buffer export");
        return 0;
    }

    const size_t max_devices = llama_max_devices();
    const size_t max_overrides = llama_max_tensor_buft_overrides();
    if (tensor_split_len < max_devices || margins_len < max_devices) {
        write_error(err, err_len, "fit buffers are too small for llama_max_devices");
        return 0;
    }
    if (tensor_buft_overrides_len < max_overrides) {
        write_error(err, err_len, "override buffer is too small for llama_max_tensor_buft_overrides");
        return 0;
    }

    const size_t tensor_split_capacity = sizeof(cfg->params.tensor_split) / sizeof(cfg->params.tensor_split[0]);
    for (size_t i = 0; i < max_devices; ++i) {
        tensor_split[i] = i < tensor_split_capacity ? cfg->params.tensor_split[i] : 0.0f;
    }

    const size_t default_margin = 1024ull * 1024ull * 1024ull;
    for (size_t i = 0; i < max_devices; ++i) {
        margins[i] = i < cfg->params.fit_params_target.size() ? cfg->params.fit_params_target[i] : default_margin;
    }

    for (size_t i = 0; i < tensor_buft_overrides_len; ++i) {
        tensor_buft_overrides[i] = llama_model_tensor_buft_override {
            /* pattern = */ nullptr,
            /* buft    = */ nullptr,
        };
    }

    const size_t copy_count = std::min(tensor_buft_overrides_len, cfg->params.tensor_buft_overrides.size());
    for (size_t i = 0; i < copy_count; ++i) {
        tensor_buft_overrides[i] = cfg->params.tensor_buft_overrides[i];
    }

    // Ensure terminator when buffer is filled.
    if (tensor_buft_overrides_len > 0 && copy_count == tensor_buft_overrides_len) {
        tensor_buft_overrides[tensor_buft_overrides_len - 1] = llama_model_tensor_buft_override {
            /* pattern = */ nullptr,
            /* buft    = */ nullptr,
        };
    }

    return 1;
}

int autocommit_common_config_ctx_shift_enabled(const autocommit_common_config * cfg) {
    if (cfg == nullptr) {
        return 0;
    }
    return cfg->params.ctx_shift ? 1 : 0;
}

int32_t autocommit_common_config_n_keep(const autocommit_common_config * cfg) {
    if (cfg == nullptr) {
        return 0;
    }
    return cfg->params.n_keep;
}

autocommit_common_sampler * autocommit_common_sampler_new(
        const autocommit_common_config * cfg,
        struct llama_model * model,
        const char * grammar,
        int grammar_lazy) {
    if (cfg == nullptr || model == nullptr) {
        return nullptr;
    }

    common_params_sampling params = cfg->params.sampling;
    if (grammar != nullptr) {
        params.grammar = grammar;
        params.grammar_lazy = grammar_lazy != 0;
    } else {
        params.grammar.clear();
        params.grammar_lazy = false;
    }

    auto * sampler = common_sampler_init(model, params);
    if (sampler == nullptr) {
        return nullptr;
    }

    auto * wrapper = new (std::nothrow) autocommit_common_sampler();
    if (wrapper == nullptr) {
        common_sampler_free(sampler);
        return nullptr;
    }
    wrapper->sampler = sampler;
    return wrapper;
}

autocommit_common_sampler * autocommit_common_sampler_clone(
        autocommit_common_sampler * sampler) {
    if (sampler == nullptr || sampler->sampler == nullptr) {
        return nullptr;
    }

    auto * cloned = common_sampler_clone(sampler->sampler);
    if (cloned == nullptr) {
        return nullptr;
    }

    auto * wrapper = new (std::nothrow) autocommit_common_sampler();
    if (wrapper == nullptr) {
        common_sampler_free(cloned);
        return nullptr;
    }
    wrapper->sampler = cloned;
    return wrapper;
}

void autocommit_common_sampler_free(autocommit_common_sampler * sampler) {
    if (sampler == nullptr) {
        return;
    }
    if (sampler->sampler != nullptr) {
        common_sampler_free(sampler->sampler);
        sampler->sampler = nullptr;
    }
    delete sampler;
}

llama_token autocommit_common_sampler_sample(
        autocommit_common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int grammar_first) {
    if (sampler == nullptr || sampler->sampler == nullptr || ctx == nullptr) {
        return LLAMA_TOKEN_NULL;
    }
    return common_sampler_sample(sampler->sampler, ctx, idx, grammar_first != 0);
}

void autocommit_common_sampler_accept(
        autocommit_common_sampler * sampler,
        llama_token token,
        int accept_grammar) {
    if (sampler == nullptr || sampler->sampler == nullptr) {
        return;
    }
    common_sampler_accept(sampler->sampler, token, accept_grammar != 0);
}

void autocommit_common_sampler_reset(autocommit_common_sampler * sampler) {
    if (sampler == nullptr || sampler->sampler == nullptr) {
        return;
    }
    common_sampler_reset(sampler->sampler);
}

} // extern "C"
