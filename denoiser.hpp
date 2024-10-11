#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include "ggml_extend.hpp"
#include "gits_noise.inl"

/*================================================= CompVisDenoiser ==================================================*/

// Ref: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/external.py

#define TIMESTEPS 1000
#define FLUX_TIMESTEPS 1000

struct SigmaSchedule {
    int version = 0;
    typedef std::function<float(float)> t_to_sigma_t;

    virtual std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) = 0;
};

struct DiscreteSchedule : SigmaSchedule {
    std::vector<float> get_sigmas(uint32_t n, float sigma_min, float sigma_max, t_to_sigma_t t_to_sigma) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct Denoiser {
    std::shared_ptr<SigmaSchedule> schedule                                                  = std::make_shared<DiscreteSchedule>();
    virtual float sigma_min()                                                                = 0;
    virtual float sigma_max()                                                                = 0;
    virtual float sigma_to_t(float sigma)                                                    = 0;
    virtual float t_to_sigma(float t)                                                        = 0;
    virtual std::vector<float> get_scalings(float sigma)                                     = 0;
    virtual ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) = 0;
    virtual ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent)             = 0;

    virtual std::vector<float> get_sigmas(uint32_t n) {
        auto bound_t_to_sigma = std::bind(&Denoiser::t_to_sigma, this, std::placeholders::_1);
        return schedule->get_sigmas(n, sigma_min(), sigma_max(), bound_t_to_sigma);
    }
};

struct CompVisDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float sigma_data = 1.0f;

    float sigma_min() {
        return sigmas[0];
    }

    float sigma_max() {
        return sigmas[TIMESTEPS - 1];
    }

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }

    std::vector<float> get_scalings(float sigma) {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) {
        ggml_tensor_scale(noise, sigma);
        ggml_tensor_add(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) {
        return latent;
    }
};

typedef std::function<ggml_tensor*(ggml_tensor*, float, int)> denoise_cb_t;

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static void sample_k_diffusion(sample_method_t method,
                               denoise_cb_t model,
                               ggml_context* work_ctx,
                               ggml_tensor* x,
                               std::vector<float> sigmas,
                               std::shared_ptr<RNG> rng) {
    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
        case EULER_A: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int i = 0; i < ggml_nelements(d); i++) {
                        vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
                    }
                }

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

                // Euler method
                float dt = sigma_down - sigmas[i];
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int i = 0; i < ggml_nelements(x); i++) {
                        vec_x[i] = vec_x[i] + vec_d[i] * dt;
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                    ggml_tensor_set_f32_randn(noise, rng);
                    // noise = load_tensor_from_file(work_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case EULER:  // Implemented without any sigma churn
        {
            struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                    }
                }

                float dt = sigmas[i + 1] - sigma;
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;
        case HEUN: {
            struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], -(i + 1));

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }
                }

                float dt = sigmas[i + 1] - sigmas[i];
                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // Heun step
                    float* vec_d  = (float*)d->data;
                    float* vec_d2 = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt;
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);
                    float* vec_denoised   = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigmas[i + 1];
                        vec_d[j] = (vec_d[j] + d2) / 2;
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;
        case DPM2: {
            struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }
                }

                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float dt     = sigmas[i + 1] - sigmas[i];
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // DPM-Solver-2
                    float sigma_mid = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                    float dt_1      = sigma_mid - sigmas[i];
                    float dt_2      = sigmas[i + 1] - sigmas[i];

                    float* vec_d  = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt_1;
                    }

                    ggml_tensor* denoised = model(x2, sigma_mid, i + 1);
                    float* vec_denoised   = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigma_mid;
                        vec_x[j] = vec_x[j] + d2 * dt_2;
                    }
                }
            }

        } break;
        case DPMPP2S_A: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2    = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                auto t_fn        = [](float sigma) -> float { return -log(sigma); };
                auto sigma_fn    = [](float t) -> float { return exp(-t); };

                if (sigma_down == 0) {
                    // Euler step
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }

                    // TODO: If sigma_down == 0, isn't this wrong?
                    // But
                    // https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py#L525
                    // has this exactly the same way.
                    float dt = sigma_down - sigmas[i];
                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // DPM-Solver++(2S)
                    float t      = t_fn(sigmas[i]);
                    float t_next = t_fn(sigma_down);
                    float h      = t_next - t;
                    float s      = t + 0.5f * h;

                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_x2       = (float*)x2->data;
                    float* vec_denoised = (float*)denoised->data;

                    // First half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = (sigma_fn(s) / sigma_fn(t)) * vec_x[j] - (exp(-h * 0.5f) - 1) * vec_denoised[j];
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);

                    // Second half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = (sigma_fn(t_next) / sigma_fn(t)) * vec_x[j] - (exp(-h) - 1) * vec_denoised[j];
                    }
                }

                // Noise addition
                if (sigmas[i + 1] > 0) {
                    ggml_tensor_set_f32_randn(noise, rng);
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case DPMPP2M:  // DPM++ (2M) from Karras et al (2022)
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float b                 = exp(-h) - 1.f;
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float r      = h_last / h;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case DPMPP2Mv2:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    float b = exp(-h) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float h_min  = std::min(h_last, h);
                    float h_max  = std::max(h_last, h);
                    float r      = h_max / h_min;
                    float h_d    = (h_max + h_min) / 2.f;
                    float b      = exp(-h_d) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case IPNDM:  // iPNDM sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order       = 4;
            ggml_tensor* x_next = x;
            std::vector<ggml_tensor*> buffer_model;

            for (int i = 0; i < steps; i++) {
                float sigma      = sigmas[i];
                float sigma_next = sigmas[i + 1];

                ggml_tensor* x_cur = x_next;
                float* vec_x_cur   = (float*)x_cur->data;
                float* vec_x_next  = (float*)x_next->data;

                // Denoising step
                ggml_tensor* denoised = model(x_cur, sigma, i + 1);
                float* vec_denoised   = (float*)denoised->data;
                // d_cur = (x_cur - denoised) / sigma
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x_cur);
                float* vec_d_cur          = (float*)d_cur->data;

                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x_cur[j] - vec_denoised[j]) / sigma;
                }

                int order = std::min(max_order, i + 1);

                // Calculate vec_x_next based on the order
                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * vec_d_cur[j];
                        }
                        break;

                    case 2:  // Use one history point
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (3 * vec_d_cur[j] - vec_d_prev1[j]) / 2;
                        }
                    } break;

                    case 3:  // Use two history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12;
                        }
                    } break;

                    case 4:  // Use three history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        float* vec_d_prev3 = (float*)buffer_model[buffer_model.size() - 3]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24;
                        }
                    } break;
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    // Shift elements to the left
                    for (int k = 0; k < max_order - 2; k++) {
                        buffer_model[k] = buffer_model[k + 1];
                    }
                    buffer_model.back() = d_cur;  // Replace the last element with d_cur
                } else {
                    buffer_model.push_back(d_cur);
                }
            }
        } break;
        case IPNDM_V:  // iPNDM_v sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order = 4;
            std::vector<ggml_tensor*> buffer_model;
            ggml_tensor* x_next = x;

            for (int i = 0; i < steps; i++) {
                float sigma  = sigmas[i];
                float t_next = sigmas[i + 1];

                // Denoising step
                ggml_tensor* denoised     = model(x, sigma, i + 1);
                float* vec_denoised       = (float*)denoised->data;
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x);
                float* vec_d_cur          = (float*)d_cur->data;
                float* vec_x              = (float*)x->data;

                // d_cur = (x - denoised) / sigma
                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                }

                int order   = std::min(max_order, i + 1);
                float h_n   = t_next - sigma;
                float h_n_1 = (i > 0) ? (sigma - sigmas[i - 1]) : h_n;

                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += vec_d_cur[j] * h_n;
                        }
                        break;

                    case 2: {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((2 + (h_n / h_n_1)) * vec_d_cur[j] - (h_n / h_n_1) * vec_d_prev1[j]) / 2;
                        }
                        break;
                    }

                    case 3: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12);
                        }
                        break;
                    }

                    case 4: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float h_n_3        = (i > 2) ? (sigmas[i - 2] - sigmas[i - 3]) : h_n_2;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        float* vec_d_prev3 = (buffer_model.size() > 2) ? (float*)buffer_model[buffer_model.size() - 3]->data : vec_d_prev2;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24);
                        }
                        break;
                    }
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    buffer_model.erase(buffer_model.begin());
                }
                buffer_model.push_back(d_cur);

                // Prepare the next d tensor
                d_cur = ggml_dup_tensor(work_ctx, x_next);
            }
        } break;
        case LCM:  // Latent Consistency Models
        {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);

                // x = denoised
                {
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                    ggml_tensor_set_f32_randn(noise, rng);
                    // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                        }
                    }
                }
            }
        } break;

        default:
            LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
            abort();
    }
}

#endif  // __DENOISER_HPP__
