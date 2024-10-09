#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "conditioner.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"

#include "tae.hpp"
#include "vae.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_STATIC
// #include "stb_image_write.h"

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 2.x",
    "SDXL",
    "SVD",
    "SD3 2B",
    "Flux Dev",
    "Flux Schnell"};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
};

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = nullptr;  // general backend
    ggml_backend_t clip_backend        = nullptr;
    ggml_backend_t control_net_backend = nullptr;
    ggml_backend_t vae_backend         = nullptr;

    ggml_type conditioner_wtype     = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype = GGML_TYPE_COUNT;
    ggml_type vae_wtype             = GGML_TYPE_COUNT;

    SDVersion version = VERSION_SDXL;

    std::shared_ptr<RNG> rng = std::make_shared<PhiloxRNG>();
    int n_threads            = 10;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() {
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    bool load_from_file(const std::string& model_path) {
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            LOG_ERROR("no backend");
            return false;
        }

        ModelLoader model_loader;

        LOG_INFO("loading model from '%s'", model_path.c_str());
        if (!model_loader.init_from_file(model_path)) {
            LOG_ERROR("init model loader from file failed: '%s'", model_path.c_str());
        }

        conditioner_wtype     = GGML_TYPE_F16;
        diffusion_model_wtype = GGML_TYPE_F16;
        vae_wtype             = GGML_TYPE_F32;
        scale_factor          = 0.13025f;

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        LOG_INFO("Conditioner weight type:     %s", ggml_type_name(conditioner_wtype));
        LOG_INFO("Diffusion model weight type: %s", ggml_type_name(diffusion_model_wtype));
        LOG_INFO("VAE weight type:             %s", ggml_type_name(vae_wtype));

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        {
            clip_backend = backend;

            cond_stage_model = std::make_shared<FrozenCLIPEmbedderWithCustomWords>(clip_backend, conditioner_wtype, "", version);
            diffusion_model  = std::make_shared<UNetModel>(backend, diffusion_model_wtype, version);

            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            {
                vae_backend = backend;

                first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend, vae_wtype, false, false, version);
                first_stage_model->alloc_params_buffer();
                first_stage_model->get_param_tensors(tensors, "first_stage_model");
            }
        }

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        bool success = model_loader.load_tensors(tensors, backend, ignore_tensors);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            return false;
        }

        auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
        if (comp_vis_denoiser) {
            float linear_start_sqrt = sqrtf(0.00085f);
            float amount            = sqrtf(0.0120) - linear_start_sqrt;  // linear_end_sqrt - linear_start_sqrt
            float product           = 1.0f;

            for (int i = 0; i < TIMESTEPS; i++) {
                float beta = linear_start_sqrt + amount * ((float)i / (TIMESTEPS - 1));
                product *= 1.0f - powf(beta, 2.0f);

                comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - product) / product);
                comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
            }
        }

        LOG_DEBUG("finished loaded file");
        return true;
    }

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        float min_cfg,
                        float cfg_scale,
                        float guidance,
                        sample_method_t method,
                        const std::vector<float>& sigmas,

                        size_t batch_num = 0) {
        size_t steps = sigmas.size() - 1;
        // noise = load_tensor_from_file(work_ctx, "./rand0.bin");
        // print_ggml_tensor(noise);
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);
        x = denoiser->noise_scaling(sigmas[0], noise, x);

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, noise);

        bool has_unconditioned = cfg_scale != 1.0 && uncond.c_crossattn != nullptr;

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond = nullptr;
        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            if (step == 1) {
                pretty_progress(0, (int)steps, 0);
            }
            int64_t t0 = ggml_time_us();

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec(x->ne[3], t);  // [N, ]
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(x->ne[3], guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_tensor_scale(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            diffusion_model->compute(n_threads,
                                     noised_input,
                                     timesteps,
                                     cond.c_crossattn,
                                     cond.c_concat,
                                     cond.c_vector,
                                     guidance_tensor,
                                     -1,
                                     controls,
                                     0.f,
                                     &out_cond);

            float* negative_data = nullptr;
            if (has_unconditioned) {
                diffusion_model->compute(n_threads,
                                         noised_input,
                                         timesteps,
                                         uncond.c_crossattn,
                                         uncond.c_concat,
                                         uncond.c_vector,
                                         guidance_tensor,
                                         -1,
                                         controls,
                                         0.f,
                                         &out_uncond);
                negative_data = (float*)out_uncond->data;
            }

            auto vec_denoised  = (float*)denoised->data;
            auto vec_input     = (float*)input->data;
            auto positive_data = (float*)out_cond->data;

            int ne_elements = (int)ggml_nelements(denoised);
            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    int64_t ne3 = out_cond->ne[3];
                    if (min_cfg != cfg_scale && ne3 != 1) {
                        // --
                    } else {
                        if (negative_data != nullptr) {
                            latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                        }
                    }
                }

                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }
            int64_t t1 = ggml_time_us();
            if (step > 0) {
                pretty_progress(step, (int)steps, (float)(t1 - t0) / 1000000.f);
            }

            send_result_step_callback(denoised, batch_num, step);

            return denoised;
        };

        sample_k_diffusion(method, denoise, work_ctx, x, sigmas, rng);

        x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);

        diffusion_model->free_compute_buffer();
        return x;
    }

    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode) {
        int64_t W = x->ne[0];
        int64_t H = x->ne[1];
        int64_t C = 8;

        ggml_tensor* result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                 decode ? (W * 8) : (W / 8),  // width
                                                 decode ? (H * 8) : (H / 8),  // height
                                                 decode ? 3 : C,
                                                 x->ne[3]);  // channels
        int64_t t0          = ggml_time_ms();
        {
            if (decode) {
                ggml_tensor_scale(x, 1.0f / scale_factor);
            } else {
                ggml_tensor_scale_input(x);
            }

            first_stage_model->compute(n_threads, x, decode, &result);
            first_stage_model->free_compute_buffer();

            if (decode) {
                ggml_tensor_scale_output(result);
            }
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae [mode: %s] graph completed, taking %.2fs", decode ? "DECODE" : "ENCODE", (t1 - t0) * 1.0f / 1000);
        if (decode) {
            ggml_tensor_clamp(result, 0.0f, 1.0f);
        }
        return result;
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        return compute_first_stage(work_ctx, x, true);
    }

    sd_result_cb_t result_cb = nullptr;
    void* result_cb_data     = nullptr;

    void send_result_callback(ggml_context* work_ctx, ggml_tensor* x, size_t number) {
        if (result_cb == nullptr) {
            return;
        }

        struct ggml_tensor* img = decode_first_stage(work_ctx, x);
        auto image_data         = sd_tensor_to_image(img);

        result_cb(number, image_data, result_cb_data);
    }

    sd_result_step_cb_t result_step_cb = nullptr;
    void* result_step_cb_data          = nullptr;

    void send_result_step_callback(ggml_tensor* x, size_t number, size_t step) {
        if (result_step_cb == nullptr) {
            return;
        }

        struct ggml_init_params params {};
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = false;

        struct ggml_context* work_ctx = ggml_init(params);
        if (!work_ctx) {
            return;
        }

        struct ggml_tensor* result = ggml_dup_tensor(work_ctx, x);
        copy_ggml_tensor(result, x);

        struct ggml_tensor* img = decode_first_stage(work_ctx, result);
        result_step_cb(number, step, sd_tensor_to_image(img), result_step_cb_data);

        ggml_free(work_ctx);
    }
};

/*================================================= SD API ==================================================*/

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

sd_ctx_t* new_sd_ctx(const char* model_path_c_str) {
    auto sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }
    std::string model_path(model_path_c_str);

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        return nullptr;
    }

    if (!sd_ctx->sd->load_from_file(model_path)) {
        free_sd_ctx(sd_ctx);
        return nullptr;
    }

    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

void sd_ctx_set_result_callback(sd_ctx_t* sd_ctx, sd_result_cb_t cb, void* data) {
    sd_ctx->sd->result_cb      = cb;
    sd_ctx->sd->result_cb_data = data;
}

void sd_ctx_set_result_step_callback(sd_ctx_t* sd_ctx, sd_result_step_cb_t cb, void* data) {
    sd_ctx->sd->result_step_cb      = cb;
    sd_ctx->sd->result_step_cb_data = data;
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx,
                           struct ggml_context* work_ctx,
                           ggml_tensor* init_latent,
                           std::string prompt,
                           std::string negative_prompt,
                           int clip_skip,
                           float cfg_scale,
                           float guidance,
                           int width,
                           int height,
                           enum sample_method_t sample_method,
                           const std::vector<float>& sigmas,
                           int64_t seed,
                           int batch_count) {
    auto result_pair = extract_and_remove_lora(prompt);

    prompt = result_pair.second;
    LOG_DEBUG("prompt after extract and remove lora: \"%s\"", prompt.c_str());

    // Get learned condition
    SDCondition cond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                           sd_ctx->sd->n_threads,
                                                                           prompt,
                                                                           clip_skip,
                                                                           width,
                                                                           height,
                                                                           sd_ctx->sd->diffusion_model->get_adm_in_channels());

    SDCondition uncond;
    if (cfg_scale != 1.0) {
        bool force_zero_embeddings = false;
        if (sd_ctx->sd->version == VERSION_SDXL && negative_prompt.empty()) {
            force_zero_embeddings = true;
        }
        uncond = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                     sd_ctx->sd->n_threads,
                                                                     negative_prompt,
                                                                     clip_skip,
                                                                     width,
                                                                     height,
                                                                     sd_ctx->sd->diffusion_model->get_adm_in_channels(),
                                                                     force_zero_embeddings);
    }

    // Sample
    int C = 4;
    int W = width / 8;
    int H = height / 8;
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor_set_f32_randn(noise, sd_ctx->sd->rng);

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     cfg_scale,
                                                     cfg_scale,
                                                     guidance,
                                                     sample_method,
                                                     sigmas,
                                                     b);

        int64_t sampling_end = ggml_time_ms();
        LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);

        if (sd_ctx->sd->result_cb != nullptr) {
            sd_ctx->sd->send_result_callback(work_ctx, x_0, b);
            continue;
        }
    }

    return nullptr;
}

sd_image_t* txt2img(sd_ctx_t* sd_ctx,
                    const char* prompt_c_str,
                    const char* negative_prompt_c_str,
                    int clip_skip,
                    float cfg_scale,
                    float guidance,
                    int width,
                    int height,
                    enum sample_method_t sample_method,
                    int sample_steps,
                    int64_t seed,
                    int batch_count) {
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    ggml_init_params params{};
    params.mem_size = static_cast<size_t>(10 * 1024 * 1024);  // 10 MB
    params.mem_size += width * height * 3 * sizeof(float);
    params.mem_size *= batch_count;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return nullptr;
    }

    auto sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps);

    int C = 4;
    int W = width / 8;
    int H = height / 8;

    ggml_tensor* init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
    ggml_set_f32(init_latent, 0.f);

    sd_image_t* result_images = generate_image(sd_ctx,
                                               work_ctx,
                                               init_latent,
                                               prompt_c_str,
                                               negative_prompt_c_str,
                                               clip_skip,
                                               cfg_scale,
                                               guidance,
                                               width,
                                               height,
                                               sample_method,
                                               sigmas,
                                               seed,
                                               batch_count);

    return result_images;
}
