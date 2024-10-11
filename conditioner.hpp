#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include "clip.hpp"

struct SDCondition {
    struct ggml_tensor* c_crossattn = NULL;  // aka context
    struct ggml_tensor* c_vector    = NULL;  // aka y
    struct ggml_tensor* c_concat    = NULL;

    SDCondition() = default;
    SDCondition(struct ggml_tensor* c_crossattn, struct ggml_tensor* c_vector, struct ggml_tensor* c_concat)
        : c_crossattn(c_crossattn), c_vector(c_vector), c_concat(c_concat) {}
};

struct FrozenCLIPEmbedderWithCustomWords {
    SDVersion version = VERSION_SD1;
    CLIPTokenizer tokenizer;
    ggml_type wtype;
    std::shared_ptr<CLIPTextModelRunner> text_model;
    std::shared_ptr<CLIPTextModelRunner> text_model2;

    std::string trigger_word = "img";  // should be user settable
    std::string embd_dir;
    int32_t num_custom_embeddings = 0;
    std::vector<uint8_t> token_embed_custom;
    std::vector<std::string> readed_embeddings;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend, ggml_type wtype, const std::string& embd_dir, SDVersion version = VERSION_SD1, int clip_skip = -1)
        : version(version), tokenizer(version == VERSION_SD2 ? 0 : 49407), embd_dir(embd_dir), wtype(wtype) {
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_SD2 || version == VERSION_SDXL) {
                clip_skip = 2;
            }
        }

        text_model  = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPENAI_CLIP_VIT_L_14, clip_skip, false);
        text_model2 = std::make_shared<CLIPTextModelRunner>(backend, wtype, OPEN_CLIP_VIT_BIGG_14, clip_skip, false);
    }

    void set_clip_skip(int clip_skip) {
        text_model->set_clip_skip(clip_skip);
        if (version == VERSION_SDXL) {
            text_model2->set_clip_skip(clip_skip);
        }
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        text_model->get_param_tensors(tensors, "cond_stage_model.transformer.text_model");
        if (version == VERSION_SDXL) {
            text_model2->get_param_tensors(tensors, "cond_stage_model.1.transformer.text_model");
        }
    }

    void alloc_params_buffer() {
        text_model->alloc_params_buffer();
        if (version == VERSION_SDXL) {
            text_model2->alloc_params_buffer();
        }
    }

    bool load_embedding(std::string embd_name, std::string embd_path, std::vector<int32_t>& bpe_tokens) {
        // the order matters
        ModelLoader model_loader;
        if (!model_loader.init_from_file(embd_path)) {
            LOG_ERROR("embedding '%s' failed", embd_name.c_str());
            return false;
        }
        if (std::find(readed_embeddings.begin(), readed_embeddings.end(), embd_name) != readed_embeddings.end()) {
            LOG_DEBUG("embedding already read in: %s", embd_name.c_str());
            return true;
        }
        struct ggml_init_params params;
        params.mem_size               = 10 * 1024 * 1024;  // max for custom embeddings 10 MB
        params.mem_buffer             = NULL;
        params.no_alloc               = false;
        struct ggml_context* embd_ctx = ggml_init(params);
        struct ggml_tensor* embd      = NULL;
        int64_t hidden_size           = text_model->model.hidden_size;
        auto on_load                  = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) {
            if (tensor_storage.ne[0] != hidden_size) {
                LOG_DEBUG("embedding wrong hidden size, got %i, expected %i", tensor_storage.ne[0], hidden_size);
                return false;
            }
            embd        = ggml_new_tensor_2d(embd_ctx, wtype, hidden_size, tensor_storage.n_dims > 1 ? tensor_storage.ne[1] : 1);
            *dst_tensor = embd;
            return true;
        };
        model_loader.load_tensors(on_load, NULL);
        readed_embeddings.push_back(embd_name);
        token_embed_custom.resize(token_embed_custom.size() + ggml_nbytes(embd));
        memcpy((void*)(token_embed_custom.data() + num_custom_embeddings * hidden_size * ggml_type_size(wtype)),
               embd->data,
               ggml_nbytes(embd));
        for (int i = 0; i < embd->ne[1]; i++) {
            bpe_tokens.push_back(text_model->model.vocab_size + num_custom_embeddings);
            // LOG_DEBUG("new custom token: %i", text_model.vocab_size + num_custom_embeddings);
            num_custom_embeddings++;
        }
        LOG_DEBUG("embedding '%s' applied, custom embeddings: %i", embd_name.c_str(), num_custom_embeddings);
        return true;
    }

    std::string decode(const std::vector<int>& tokens) {
        return tokenizer.decode(tokens);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text, bool padding = false) {
        return tokenize(text, text_model->model.n_token, padding);
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text, size_t max_length = 0, bool padding = false) {
        auto parsed_attention = parse_prompt_attention(text);

        {
            std::stringstream ss;
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            LOG_DEBUG("parsed to %s", ss.str().c_str());
        }

        auto on_new_token_cb = [&](std::string& str, std::vector<int32_t>& bpe_tokens) -> bool {
            size_t word_end       = str.find(",");
            std::string embd_name = word_end == std::string::npos ? str : str.substr(0, word_end);
            embd_name             = trim(embd_name);
            std::string embd_path = get_full_path(embd_dir, embd_name + ".pt");
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".ckpt");
            }
            if (embd_path.size() == 0) {
                embd_path = get_full_path(embd_dir, embd_name + ".safetensors");
            }
            if (embd_path.size() > 0) {
                if (load_embedding(embd_name, embd_path, bpe_tokens)) {
                    if (word_end != std::string::npos) {
                        str = str.substr(word_end);
                    } else {
                        str = "";
                    }
                    return true;
                }
            }
            return false;
        };

        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text, on_new_token_cb);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer.pad_tokens(tokens, weights, max_length, padding);

        return {tokens, weights};
    }

    SDCondition get_learned_condition_common(ggml_context* work_ctx,
                                             int n_threads,
                                             std::vector<int>& tokens,
                                             std::vector<float>& weights,
                                             int clip_skip,
                                             int width,
                                             int height,
                                             int adm_in_channels        = -1,
                                             bool force_zero_embeddings = false) {
        set_clip_skip(clip_skip);
        int64_t t0                               = ggml_time_ms();
        struct ggml_tensor* hidden_states        = NULL;  // [N, n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states  = NULL;  // [n_token, hidden_size] or [n_token, hidden_size + hidden_size2]
        struct ggml_tensor* chunk_hidden_states1 = NULL;  // [n_token, hidden_size]
        struct ggml_tensor* chunk_hidden_states2 = NULL;  // [n_token, hidden_size2]
        struct ggml_tensor* pooled               = NULL;
        std::vector<float> hidden_states_vec;

        size_t chunk_len   = 77;
        size_t chunk_count = tokens.size() / chunk_len;
        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            std::vector<int> chunk_tokens(tokens.begin() + chunk_idx * chunk_len,
                                          tokens.begin() + (chunk_idx + 1) * chunk_len);
            std::vector<float> chunk_weights(weights.begin() + chunk_idx * chunk_len,
                                             weights.begin() + (chunk_idx + 1) * chunk_len);

            auto input_ids                 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);
            struct ggml_tensor* input_ids2 = NULL;
            size_t max_token_idx           = 0;
            if (version == VERSION_SDXL) {
                auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), tokenizer.EOS_TOKEN_ID);
                if (it != chunk_tokens.end()) {
                    std::fill(std::next(it), chunk_tokens.end(), 0);
                }

                max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

                input_ids2 = vector_to_ggml_tensor_i32(work_ctx, chunk_tokens);

                // for (int i = 0; i < chunk_tokens.size(); i++) {
                //     printf("%d ", chunk_tokens[i]);
                // }
                // printf("\n");
            }

            {
                text_model->compute(n_threads,
                                    input_ids,
                                    num_custom_embeddings,
                                    token_embed_custom.data(),
                                    max_token_idx,
                                    false,
                                    &chunk_hidden_states1,
                                    work_ctx);
                if (version == VERSION_SDXL) {
                    text_model2->compute(n_threads,
                                         input_ids2,
                                         0,
                                         NULL,
                                         max_token_idx,
                                         false,
                                         &chunk_hidden_states2, work_ctx);
                    // concat
                    chunk_hidden_states = ggml_tensor_concat(work_ctx, chunk_hidden_states1, chunk_hidden_states2, 0);

                    if (chunk_idx == 0) {
                        text_model2->compute(n_threads,
                                             input_ids2,
                                             0,
                                             NULL,
                                             max_token_idx,
                                             true,
                                             &pooled,
                                             work_ctx);
                    }
                } else {
                    chunk_hidden_states = chunk_hidden_states1;
                }
            }

            int64_t t1 = ggml_time_ms();
            LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
            ggml_tensor* result = ggml_dup_tensor(work_ctx, chunk_hidden_states);
            {
                float original_mean = ggml_tensor_mean(chunk_hidden_states);
                for (int i2 = 0; i2 < chunk_hidden_states->ne[2]; i2++) {
                    for (int i1 = 0; i1 < chunk_hidden_states->ne[1]; i1++) {
                        for (int i0 = 0; i0 < chunk_hidden_states->ne[0]; i0++) {
                            float value = ggml_tensor_get_f32(chunk_hidden_states, i0, i1, i2);
                            value *= chunk_weights[i1];
                            ggml_tensor_set_f32(result, value, i0, i1, i2);
                        }
                    }
                }
                float new_mean = ggml_tensor_mean(result);
                ggml_tensor_scale(result, (original_mean / new_mean));
            }
            if (force_zero_embeddings) {
                float* vec = (float*)result->data;
                for (int i = 0; i < ggml_nelements(result); i++) {
                    vec[i] = 0;
                }
            }
            hidden_states_vec.insert(hidden_states_vec.end(), (float*)result->data, ((float*)result->data) + ggml_nelements(result));
        }

        hidden_states = vector_to_ggml_tensor(work_ctx, hidden_states_vec);
        hidden_states = ggml_reshape_2d(work_ctx,
                                        hidden_states,
                                        chunk_hidden_states->ne[0],
                                        ggml_nelements(hidden_states) / chunk_hidden_states->ne[0]);

        ggml_tensor* vec = NULL;
        if (version == VERSION_SDXL) {
            int out_dim = 256;
            vec         = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, adm_in_channels);
            // [0:1280]
            size_t offset = 0;
            memcpy(vec->data, pooled->data, ggml_nbytes(pooled));
            offset += ggml_nbytes(pooled);

            // original_size_as_tuple
            float orig_width             = (float)width;
            float orig_height            = (float)height;
            std::vector<float> timesteps = {orig_height, orig_width};

            ggml_tensor* embed_view = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // crop_coords_top_left
            float crop_coord_top  = 0.f;
            float crop_coord_left = 0.f;
            timesteps             = {crop_coord_top, crop_coord_left};
            embed_view            = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            // target_size_as_tuple
            float target_width  = (float)width;
            float target_height = (float)height;
            timesteps           = {target_height, target_width};
            embed_view          = ggml_view_2d(work_ctx, vec, out_dim, 2, ggml_type_size(GGML_TYPE_F32) * out_dim, offset);
            offset += ggml_nbytes(embed_view);
            set_timestep_embedding(timesteps, embed_view, out_dim);
            // print_ggml_tensor(ggml_reshape_1d(work_ctx, embed_view, out_dim * 2));
            GGML_ASSERT(offset == ggml_nbytes(vec));
        }
        // print_ggml_tensor(result);
        return SDCondition(hidden_states, vec, NULL);
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const std::string& text,
                                      int clip_skip,
                                      int width,
                                      int height,
                                      int adm_in_channels        = -1,
                                      bool force_zero_embeddings = false) {
        auto tokens_and_weights     = tokenize(text, true);
        std::vector<int>& tokens    = tokens_and_weights.first;
        std::vector<float>& weights = tokens_and_weights.second;
        return get_learned_condition_common(work_ctx, n_threads, tokens, weights, clip_skip, width, height, adm_in_channels, force_zero_embeddings);
    }
};

#endif