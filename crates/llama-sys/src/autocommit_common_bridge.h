#ifndef AUTOCOMMIT_COMMON_BRIDGE_H
#define AUTOCOMMIT_COMMON_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct autocommit_common_config autocommit_common_config;
typedef struct autocommit_common_sampler autocommit_common_sampler;

autocommit_common_config * autocommit_common_config_new(void);
void autocommit_common_config_free(autocommit_common_config * cfg);

void autocommit_common_config_set_model_path(autocommit_common_config * cfg, const char * path);
void autocommit_common_config_set_hf_repo(autocommit_common_config * cfg, const char * repo);
void autocommit_common_config_set_cache_dir(autocommit_common_config * cfg, const char * dir);
void autocommit_common_config_set_n_parallel(autocommit_common_config * cfg, int32_t n_parallel);

int autocommit_common_config_resolve_model_path(
        autocommit_common_config * cfg,
        char * out_path,
        size_t out_path_len,
        char * err,
        size_t err_len);

int autocommit_common_config_list_cached_models(
        autocommit_common_config * cfg,
        char * out_models,
        size_t out_models_len,
        char * out_cache_dir,
        size_t out_cache_dir_len,
        char * err,
        size_t err_len);

int autocommit_common_config_apply_env(
        autocommit_common_config * cfg,
        char * err,
        size_t err_len);

int autocommit_common_config_export_llama_params(
        autocommit_common_config * cfg,
        struct llama_model_params * mparams,
        struct llama_context_params * cparams,
        char * err,
        size_t err_len);

int autocommit_common_config_fill_fit_buffers(
        autocommit_common_config * cfg,
        float * tensor_split,
        size_t tensor_split_len,
        struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t tensor_buft_overrides_len,
        size_t * margins,
        size_t margins_len,
        char * err,
        size_t err_len);

int autocommit_common_config_ctx_shift_enabled(const autocommit_common_config * cfg);
int32_t autocommit_common_config_n_keep(const autocommit_common_config * cfg);

autocommit_common_sampler * autocommit_common_sampler_new(
        const autocommit_common_config * cfg,
        struct llama_model * model,
        const char * grammar,
        int grammar_lazy);
autocommit_common_sampler * autocommit_common_sampler_clone(
        autocommit_common_sampler * sampler);
void autocommit_common_sampler_free(autocommit_common_sampler * sampler);
llama_token autocommit_common_sampler_sample(
        autocommit_common_sampler * sampler,
        struct llama_context * ctx,
        int idx,
        int grammar_first);
void autocommit_common_sampler_accept(
        autocommit_common_sampler * sampler,
        llama_token token,
        int accept_grammar);
void autocommit_common_sampler_reset(autocommit_common_sampler * sampler);

/// Compute cosine similarity between two float vectors of length n.
/// Uses BLAS (Accelerate on macOS) when available, falls back to plain C loop.
/// Returns 0.0 if either vector has zero norm or n <= 0.
float autocommit_cosine_similarity(const float * a, const float * b, int n);

/// Set the common-library log verbosity threshold.
/// 0 = errors only, 1 = +warnings, 2 = +info, 3 = +debug.
/// Set to 0 to suppress the "using cached file" INFO messages from download.cpp.
void autocommit_common_log_set_verbosity(int verbosity);

#ifdef __cplusplus
}
#endif

#endif // AUTOCOMMIT_COMMON_BRIDGE_H
