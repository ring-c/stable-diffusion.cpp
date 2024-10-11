#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "unet.hpp"

struct UNetModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              ggml_type wtype,
              SDVersion version = VERSION_SD1)
        : unet(backend, wtype, version) {
    }

    void alloc_params_buffer() {
        unet.alloc_params_buffer();
    }

    void free_compute_buffer() {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    int64_t get_adm_in_channels() {
        return unet.unet.adm_in_channels;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL) {
        return unet.compute(n_threads, x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength, output, output_ctx);
    }
};

#endif