#include "autocommit_common_bridge.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "arg.h"
#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "log.h"
#include "sampling.h"

// build-info symbols (llama_build_number(), llama_commit(), the LLAMA_BUILD_*
// globals, ...) are provided by libllama-common at link time: from the prebuilt
// libllama-common.so for binary releases, and from the static llama-common-base
// archive for source builds (linked explicitly in build.rs). Defining fallbacks
// here would collide with those real definitions, so we don't.

struct autocommit_common_config {
    common_params params;
    std::string cache_dir_override;
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

void write_out(char * out, const size_t out_len, const std::string & msg) {
    if (out == nullptr || out_len == 0) {
        return;
    }

    const size_t write_len = std::min(out_len - 1, msg.size());
    std::memcpy(out, msg.data(), write_len);
    out[write_len] = '\0';
}

} // namespace

extern "C" {

autocommit_common_config * autocommit_common_config_new(void) {
    auto * cfg = new (std::nothrow) autocommit_common_config();
    if (cfg == nullptr) {
        return nullptr;
    }

    cfg->params.n_gpu_layers = -1;
    cfg->params.n_ctx        = 8192;
    cfg->params.n_parallel   = 1;
    cfg->params.n_batch      = 1024;
    cfg->params.n_ubatch     = 256;
    cfg->params.cache_type_k = GGML_TYPE_Q8_0;
    cfg->params.cache_type_v = GGML_TYPE_Q8_0;
    cfg->params.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
    cfg->params.speculative.draft.cache_type_v = GGML_TYPE_Q8_0;
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
    if (!cfg->params.model.path.empty()) {
        cfg->params.model.hf_repo.clear();
        cfg->params.model.hf_file.clear();
        cfg->params.model.url.clear();
    }
}

void autocommit_common_config_set_hf_repo(autocommit_common_config * cfg, const char * repo) {
    if (cfg == nullptr) {
        return;
    }
    cfg->params.model.hf_repo = repo != nullptr ? repo : "";
    if (!cfg->params.model.hf_repo.empty()) {
        cfg->params.model.path.clear();
        cfg->params.model.url.clear();
        cfg->params.model.hf_file.clear();
    }
}

void autocommit_common_config_set_cache_dir(autocommit_common_config * cfg, const char * dir) {
    if (cfg == nullptr) {
        return;
    }
    cfg->cache_dir_override = dir != nullptr ? dir : "";
}

void autocommit_common_config_set_n_parallel(autocommit_common_config * cfg, const int32_t n_parallel) {
    if (cfg == nullptr || n_parallel <= 0) {
        return;
    }
    cfg->params.n_parallel = n_parallel;
}

static bool is_hf_repo(const std::string & s) {
    return s.find('/') != std::string::npos;
}

static std::string resolve_hf_model(autocommit_common_config & cfg) {
    auto & model = cfg.params.model;

    common_download_opts opts;
    opts.bearer_token = cfg.params.hf_token;
    opts.offline  = cfg.params.offline;

    auto plan = common_download_get_hf_plan(model, opts);
    if (plan.primary.path.empty()) {
        throw std::runtime_error("failed to auto-detect GGUF file from Hugging Face repo");
    }

    if (model.hf_file.empty()) {
        model.hf_file = plan.primary.path;
    }

    common_download_task task(plan.primary, opts);
    common_download_run_tasks({ task });

    // finalize_file creates the HF-cache snapshot symlink
    // (snapshots/<rev>/<file> -> ../../blobs/<sha>) and returns the path to use:
    // the snapshot symlink, or the blob path itself if symlink creation fails.
    // Without it the blob downloads but the resolved final_path dangles. Mirrors
    // how llama.cpp's own common/arg.cpp finalizes downloaded models.
    model.path = hf_cache::finalize_file(plan.primary);
    return model.path;
}

static std::string resolve_model_path(autocommit_common_config & cfg) {
    auto & model = cfg.params.model;

    if (!model.path.empty()) {
        return model.path;
    }

    if (!model.docker_repo.empty()) {
        model.path = common_docker_resolve_model(model.docker_repo);
        return model.path;
    }

    if (!model.hf_repo.empty()) {
        return resolve_hf_model(cfg);
    }

    if (!model.url.empty()) {
        if (model.path.empty()) {
            auto pos = model.url.find_last_of("/\\");
            model.path = pos != std::string::npos ? model.url.substr(pos + 1) : model.url;
        }

        common_download_opts opts;
        opts.bearer_token = cfg.params.hf_token;
        opts.offline  = cfg.params.offline;

        if (common_download_file_single(model.url, model.path, opts) < 0) {
            throw std::runtime_error("failed to download model from " + model.url);
        }
    }

    if (model.path.empty()) {
        throw std::runtime_error("model path is not configured");
    }

    return model.path;
}

int autocommit_common_config_resolve_model_path(
        autocommit_common_config * cfg,
        char * out_path,
        const size_t out_path_len,
        char * err,
        const size_t err_len) {
    if (cfg == nullptr) {
        write_error(err, err_len, "common config is null");
        return 0;
    }

    try {
        const std::string path = resolve_model_path(*cfg);
        write_out(out_path, out_path_len, path);
        return 1;
    } catch (const std::exception & ex) {
        write_error(err, err_len, ex.what());
        return 0;
    }
}

int autocommit_common_config_list_cached_models(
        autocommit_common_config * cfg,
        char * out_models,
        const size_t out_models_len,
        char * out_cache_dir,
        const size_t out_cache_dir_len,
        char * err,
        const size_t err_len) {
    if (cfg == nullptr) {
        write_error(err, err_len, "common config is null");
        return 0;
    }

    try {
        std::vector<common_cached_model_info> models = common_list_cached_models();
        std::sort(models.begin(), models.end(), [](const auto & a, const auto & b) {
            return a.to_string() < b.to_string();
        });

        std::string joined;
        for (size_t i = 0; i < models.size(); ++i) {
            if (i > 0) {
                joined.push_back('\n');
            }
            joined += models[i].to_string();
        }

        std::string cache_dir = cfg->cache_dir_override.empty()
            ? fs_get_cache_directory()
            : cfg->cache_dir_override;

        write_out(out_models, out_models_len, joined);
        write_out(out_cache_dir, out_cache_dir_len, cache_dir);
        return 1;
    } catch (const std::exception & ex) {
        write_error(err, err_len, ex.what());
        return 0;
    }
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
                opt.handler_bool(cfg->params, common_arg_utils::is_truthy(value));
            }
            if (opt.handler_string) {
                opt.handler_string(cfg->params, value);
            }
        }

        if (cfg->params.n_parallel <= 0) {
            cfg->params.n_parallel = n_parallel_seed;
        }

        postprocess_cpu_params(cfg->params.cpuparams, nullptr);
        postprocess_cpu_params(cfg->params.cpuparams_batch, &cfg->params.cpuparams);
        postprocess_cpu_params(cfg->params.speculative.draft.cpuparams, &cfg->params.cpuparams);
        postprocess_cpu_params(cfg->params.speculative.draft.cpuparams_batch, &cfg->params.cpuparams_batch);

        if (!cfg->params.kv_overrides.empty()) {
            cfg->params.kv_overrides.emplace_back();
            cfg->params.kv_overrides.back().key[0] = 0;
        }
        if (!cfg->params.tensor_buft_overrides.empty() &&
            cfg->params.tensor_buft_overrides.back().pattern != nullptr) {
            cfg->params.tensor_buft_overrides.push_back(
                llama_model_tensor_buft_override {
                    /* pattern = */ nullptr,
                    /* buft    = */ nullptr,
                });
        }
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
        params.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
        params.grammar_lazy = grammar_lazy != 0;
    } else {
        params.grammar = common_grammar();
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

void autocommit_common_log_set_verbosity(int verbosity) {
    common_log_set_verbosity_thold(verbosity);
}

struct autocommit_init_result {
    llama_model * model;
    llama_context_params cparams;
};

static int fit_llama_params(
        const std::string & model_path,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        const common_params & params) {
    const size_t nd = std::max<size_t>(llama_max_devices(), 1);
    const size_t no = std::max<size_t>(llama_max_tensor_buft_overrides(), 1);

    std::vector<float> tensor_split(nd, 0.0f);
    std::vector<size_t> margins(nd, 1024ull * 1024 * 1024);
    std::vector<llama_model_tensor_buft_override> overrides(
        no, llama_model_tensor_buft_override{nullptr, nullptr});

    const size_t ts_cap = sizeof(params.tensor_split) / sizeof(params.tensor_split[0]);
    for (size_t i = 0; i < nd && i < ts_cap; ++i) {
        tensor_split[i] = params.tensor_split[i];
    }
    for (size_t i = 0; i < nd && i < params.fit_params_target.size(); ++i) {
        margins[i] = params.fit_params_target[i];
    }
    const size_t copy_count = std::min(no, params.tensor_buft_overrides.size());
    for (size_t i = 0; i < copy_count; ++i) {
        overrides[i] = params.tensor_buft_overrides[i];
    }

    mparams->tensor_split = tensor_split.data();
    mparams->tensor_buft_overrides = overrides.data();

    return autocommit_llama_params_fit(
        model_path.c_str(),
        mparams, cparams,
        tensor_split.data(),
        overrides.data(),
        margins.data(),
        4096,
        4); // GGML_LOG_LEVEL_ERROR
}

autocommit_init_result * autocommit_init(
        autocommit_common_config * cfg,
        int embedding,
        int cpu_only,
        char * err,
        size_t err_len) {
    if (cfg == nullptr) {
        write_error(err, err_len, "common config is null");
        return nullptr;
    }

    try {
        auto & params = cfg->params;

        // Override for model type and CPU-only
        if (embedding) {
            params.embedding = true;
        }
        if (cpu_only) {
            params.n_gpu_layers = 0;
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
            params.main_gpu = -1;
            params.no_kv_offload = true;
            params.no_op_offload = true;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }

        // Resolve model path (download if needed)
        std::string model_path = resolve_model_path(*cfg);

        // Create llama C API params from common_params
        llama_model_params mparams = common_model_params_to_llama(params);
        llama_context_params cparams = common_context_params_to_llama(params);

        // Apply embedding override directly on C params
        // Note: llama_model_params no longer has an embedding field in b9837;
        // embedding is controlled via context_params.embeddings and llama_set_embeddings.
        if (embedding) {
            cparams.embeddings = true;
            cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        }

        // Fit params to available device memory
        fit_llama_params(model_path, &mparams, &cparams, params);

        // Load model
        llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model == nullptr) {
            write_error(err, err_len, "failed to load model");
            return nullptr;
        }

        auto * result = new (std::nothrow) autocommit_init_result{model, cparams};
        if (result == nullptr) {
            llama_model_free(model);
            write_error(err, err_len, "allocation failed");
            return nullptr;
        }
        return result;
    } catch (const std::exception & ex) {
        write_error(err, err_len, ex.what());
        return nullptr;
    }
}

struct llama_model * autocommit_init_get_model(autocommit_init_result * result) {
    return result ? result->model : nullptr;
}

void autocommit_init_get_context_params(
        autocommit_init_result * result,
        struct llama_context_params * out) {
    if (result && out) {
        *out = result->cparams;
    }
}

void autocommit_init_free(autocommit_init_result * result) {
    delete result;
}

int autocommit_llama_params_fit(
        const char * path_model,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        float * tensor_split,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        int log_level) {
    // llama_params_fit is not exposed by the pinned llama.cpp release (b9837)
    // in either build mode: the binary release omits it, and in the from-source
    // build the fitting logic lives in libcommon's internal common/fit.cpp with
    // no matching public `llama_params_fit` symbol. Skip fitting and return
    // success — the caller loads the model with whatever params were configured.
    // This matches the behavior of the prebuilt backends that already ship.
    (void)path_model;
    (void)mparams;
    (void)cparams;
    (void)tensor_split;
    (void)tensor_buft_overrides;
    (void)margins;
    (void)n_ctx_min;
    (void)log_level;
    return 0;
}

} // extern "C"

// --- Hardware-accelerated cosine similarity ---

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include <cmath>

extern "C" float autocommit_cosine_similarity(const float * a, const float * b, int n) {
    if (n <= 0 || a == nullptr || b == nullptr) {
        return 0.0f;
    }

    float dot    = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

#ifdef __APPLE__
    const vDSP_Length len = static_cast<vDSP_Length>(n);
    vDSP_dotpr(a, 1, b, 1, &dot, len);
    vDSP_dotpr(a, 1, a, 1, &norm_a, len);
    vDSP_dotpr(b, 1, b, 1, &norm_b, len);
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
#else
    for (int i = 0; i < n; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
#endif

    if (norm_a <= 1e-7f || norm_b <= 1e-7f) {
        return 0.0f;
    }
    return dot / (norm_a * norm_b);
}
