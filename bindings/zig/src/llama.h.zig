const std = @import("std");
const ggml = @import("ggml.h");

pub const struct_llama_model = opaque {};
pub const struct_llama_context = opaque {};
pub const llama_pos = i32;
pub const llama_token = i32;
pub const llama_seq_id = i32;
pub const LLAMA_VOCAB_TYPE_SPM: c_int = 0;
pub const LLAMA_VOCAB_TYPE_BPE: c_int = 1;
pub const enum_llama_vocab_type = c_uint;
pub const LLAMA_TOKEN_TYPE_UNDEFINED: c_int = 0;
pub const LLAMA_TOKEN_TYPE_NORMAL: c_int = 1;
pub const LLAMA_TOKEN_TYPE_UNKNOWN: c_int = 2;
pub const LLAMA_TOKEN_TYPE_CONTROL: c_int = 3;
pub const LLAMA_TOKEN_TYPE_USER_DEFINED: c_int = 4;
pub const LLAMA_TOKEN_TYPE_UNUSED: c_int = 5;
pub const LLAMA_TOKEN_TYPE_BYTE: c_int = 6;
pub const enum_llama_token_type = c_uint;
pub const LLAMA_FTYPE_ALL_F32: c_int = 0;
pub const LLAMA_FTYPE_MOSTLY_F16: c_int = 1;
pub const LLAMA_FTYPE_MOSTLY_Q4_0: c_int = 2;
pub const LLAMA_FTYPE_MOSTLY_Q4_1: c_int = 3;
pub const LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16: c_int = 4;
pub const LLAMA_FTYPE_MOSTLY_Q8_0: c_int = 7;
pub const LLAMA_FTYPE_MOSTLY_Q5_0: c_int = 8;
pub const LLAMA_FTYPE_MOSTLY_Q5_1: c_int = 9;
pub const LLAMA_FTYPE_MOSTLY_Q2_K: c_int = 10;
pub const LLAMA_FTYPE_MOSTLY_Q3_K_S: c_int = 11;
pub const LLAMA_FTYPE_MOSTLY_Q3_K_M: c_int = 12;
pub const LLAMA_FTYPE_MOSTLY_Q3_K_L: c_int = 13;
pub const LLAMA_FTYPE_MOSTLY_Q4_K_S: c_int = 14;
pub const LLAMA_FTYPE_MOSTLY_Q4_K_M: c_int = 15;
pub const LLAMA_FTYPE_MOSTLY_Q5_K_S: c_int = 16;
pub const LLAMA_FTYPE_MOSTLY_Q5_K_M: c_int = 17;
pub const LLAMA_FTYPE_MOSTLY_Q6_K: c_int = 18;
pub const LLAMA_FTYPE_MOSTLY_IQ2_XXS: c_int = 19;
pub const LLAMA_FTYPE_MOSTLY_IQ2_XS: c_int = 20;
pub const LLAMA_FTYPE_MOSTLY_Q2_K_S: c_int = 21;
pub const LLAMA_FTYPE_MOSTLY_Q3_K_XS: c_int = 22;
pub const LLAMA_FTYPE_MOSTLY_IQ3_XXS: c_int = 23;
pub const LLAMA_FTYPE_GUESSED: c_int = 1024;
pub const enum_llama_ftype = c_uint;
pub const LLAMA_ROPE_SCALING_UNSPECIFIED: c_int = -1;
pub const LLAMA_ROPE_SCALING_NONE: c_int = 0;
pub const LLAMA_ROPE_SCALING_LINEAR: c_int = 1;
pub const LLAMA_ROPE_SCALING_YARN: c_int = 2;
pub const LLAMA_ROPE_SCALING_MAX_VALUE: c_int = 2;
pub const enum_llama_rope_scaling_type = c_int;
pub const LLAMA_SPLIT_NONE: c_int = 0;
pub const LLAMA_SPLIT_LAYER: c_int = 1;
pub const LLAMA_SPLIT_ROW: c_int = 2;
pub const enum_llama_split_mode = c_uint;
pub const struct_llama_token_data = extern struct {
    id: llama_token = @import("std").mem.zeroes(llama_token),
    logit: f32 = @import("std").mem.zeroes(f32),
    p: f32 = @import("std").mem.zeroes(f32),
};
pub const llama_token_data = struct_llama_token_data;
pub const struct_llama_token_data_array = extern struct {
    data: [*c]llama_token_data = @import("std").mem.zeroes([*c]llama_token_data),
    size: usize = @import("std").mem.zeroes(usize),
    sorted: bool = @import("std").mem.zeroes(bool),
};
pub const llama_token_data_array = struct_llama_token_data_array;
pub const llama_progress_callback = ?*const fn (f32, ?*anyopaque) callconv(.C) bool;
pub const struct_llama_batch = extern struct {
    n_tokens: i32 = @import("std").mem.zeroes(i32),
    token: [*c]llama_token = @import("std").mem.zeroes([*c]llama_token),
    embd: [*c]f32 = @import("std").mem.zeroes([*c]f32),
    pos: [*c]llama_pos = @import("std").mem.zeroe([*c]llama_pos),
    n_seq_id: [*c]i32 = @import("std").mem.zeroes([*c]i32),
    seq_id: [*c][*c]llama_seq_id = @import("std").mem.zeroes([*c][*c]llama_seq_id),
    logits: [*c]i8 = @import("std").mem.zeroes([*c]i8),
    all_pos_0: llama_pos = @import("std").mem.zeroes(llama_pos),
    all_pos_1: llama_pos = @import("std").mem.zeroes(llama_pos),
    all_seq_id: llama_seq_id = @import("std").mem.zeroes(llama_seq_id),
};
pub const llama_batch = struct_llama_batch;
pub const LLAMA_KV_OVERRIDE_INT: c_int = 0;
pub const LLAMA_KV_OVERRIDE_FLOAT: c_int = 1;
pub const LLAMA_KV_OVERRIDE_BOOL: c_int = 2;
pub const enum_llama_model_kv_override_type = c_uint;
const union_unnamed_7 = extern union {
    int_value: i64,
    float_value: f64,
    bool_value: bool,
};
pub const struct_llama_model_kv_override = extern struct {
    key: [128]u8 = @import("std").mem.zeroes([128]u8),
    tag: enum_llama_model_kv_override_type = @import("std").mem.zeroes(enum_llama_model_kv_override_type),
    unnamed_0: union_unnamed_7 = @import("std").mem.zeroes(union_unnamed_7),
};
pub const struct_llama_model_params = extern struct {
    n_gpu_layers: i32 = @import("std").mem.zeroes(i32),
    split_mode: enum_llama_split_mode = @import("std").mem.zeroes(enum_llama_split_mode),
    main_gpu: i32 = @import("std").mem.zeroes(i32),
    tensor_split: [*c]const f32 = @import("std").mem.zeroes([*c]const f32),
    progress_callback: llama_progress_callback = @import("std").mem.zeroes(llama_progress_callback),
    progress_callback_user_data: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    kv_overrides: [*c]const struct_llama_model_kv_override = @import("std").mem.zeroes([*c]const struct_llama_model_kv_override),
    vocab_only: bool = @import("std").mem.zeroes(bool),
    use_mmap: bool = @import("std").mem.zeroes(bool),
    use_mlock: bool = @import("std").mem.zeroes(bool),
};
pub const struct_llama_context_params = extern struct {
    seed: u32 = @import("std").mem.zeroes(u32),
    n_ctx: u32 = @import("std").mem.zeroes(u32),
    n_batch: u32 = @import("std").mem.zeroes(u32),
    n_threads: u32 = @import("std").mem.zeroes(u32),
    n_threads_batch: u32 = @import("std").mem.zeroes(u32),
    rope_scaling_type: i8 = @import("std").mem.zeroes(i8),
    rope_freq_base: f32 = @import("std").mem.zeroes(f32),
    rope_freq_scale: f32 = @import("std").mem.zeroes(f32),
    yarn_ext_factor: f32 = @import("std").mem.zeroes(f32),
    yarn_attn_factor: f32 = @import("std").mem.zeroes(f32),
    yarn_beta_fast: f32 = @import("std").mem.zeroes(f32),
    yarn_beta_slow: f32 = @import("std").mem.zeroes(f32),
    yarn_orig_ctx: u32 = @import("std").mem.zeroes(u32),
    cb_eval: ggml.backend_sched_eval_callback = @import("std").mem.zeroes(ggml.backend_sched_eval_callback),
    cb_eval_user_data: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    type_k: ggml.type = @import("std").mem.zeroes(ggml.type),
    type_v: ggml.type = @import("std").mem.zeroes(ggml.type),
    mul_mat_q: bool = @import("std").mem.zeroes(bool),
    logits_all: bool = @import("std").mem.zeroes(bool),
    embedding: bool = @import("std").mem.zeroes(bool),
    offload_kqv: bool = @import("std").mem.zeroes(bool),
};
pub const struct_llama_model_quantize_params = extern struct {
    nthread: i32 = @import("std").mem.zeroes(i32),
    ftype: enum_llama_ftype = @import("std").mem.zeroes(enum_llama_ftype),
    allow_requantize: bool = @import("std").mem.zeroes(bool),
    quantize_output_tensor: bool = @import("std").mem.zeroes(bool),
    only_copy: bool = @import("std").mem.zeroes(bool),
    pure: bool = @import("std").mem.zeroes(bool),
    imatrix: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const llama_model_quantize_params = struct_llama_model_quantize_params;
pub const struct_llama_grammar = opaque {};
pub const LLAMA_GRETYPE_END: c_int = 0;
pub const LLAMA_GRETYPE_ALT: c_int = 1;
pub const LLAMA_GRETYPE_RULE_REF: c_int = 2;
pub const LLAMA_GRETYPE_CHAR: c_int = 3;
pub const LLAMA_GRETYPE_CHAR_NOT: c_int = 4;
pub const LLAMA_GRETYPE_CHAR_RNG_UPPER: c_int = 5;
pub const LLAMA_GRETYPE_CHAR_ALT: c_int = 6;
pub const enum_llama_gretype = c_uint;
pub const struct_llama_grammar_element = extern struct {
    type: enum_llama_gretype = @import("std").mem.zeroes(enum_llama_gretype),
    value: u32 = @import("std").mem.zeroes(u32),
};
pub const llama_grammar_element = struct_llama_grammar_element;
pub const struct_llama_timings = extern struct {
    t_start_ms: f64 = @import("std").mem.zeroes(f64),
    t_end_ms: f64 = @import("std").mem.zeroes(f64),
    t_load_ms: f64 = @import("std").mem.zeroes(f64),
    t_sample_ms: f64 = @import("std").mem.zeroes(f64),
    t_p_eval_ms: f64 = @import("std").mem.zeroes(f64),
    t_eval_ms: f64 = @import("std").mem.zeroes(f64),
    n_sample: i32 = @import("std").mem.zeroes(i32),
    n_p_eval: i32 = @import("std").mem.zeroes(i32),
    n_eval: i32 = @import("std").mem.zeroes(i32),
};
pub extern fn llama_model_default_params() struct_llama_model_params;
pub extern fn llama_context_default_params() struct_llama_context_params;
pub extern fn llama_model_quantize_default_params() struct_llama_model_quantize_params;
pub extern fn llama_backend_init(numa: bool) void;
pub extern fn llama_backend_free() void;
pub extern fn llama_load_model_from_file(path_model: [*c]const u8, params: struct_llama_model_params) ?*struct_llama_model;
pub extern fn llama_free_model(model: ?*struct_llama_model) void;
pub extern fn llama_new_context_with_model(model: ?*struct_llama_model, params: struct_llama_context_params) ?*struct_llama_context;
pub extern fn llama_free(ctx: ?*struct_llama_context) void;
pub extern fn llama_time_us() i64;
pub extern fn llama_max_devices() usize;
pub extern fn llama_supports_mmap() bool;
pub extern fn llama_supports_mlock() bool;
pub extern fn llama_supports_gpu_offload() bool;
pub extern fn llama_mmap_supported() bool;
pub extern fn llama_mlock_supported() bool;
pub extern fn llama_get_model(ctx: ?*const struct_llama_context) ?*const struct_llama_model;
pub extern fn llama_n_ctx(ctx: ?*const struct_llama_context) u32;
pub extern fn llama_n_batch(ctx: ?*const struct_llama_context) u32;
pub extern fn llama_vocab_type(model: ?*const struct_llama_model) enum_llama_vocab_type;
pub extern fn llama_n_vocab(model: ?*const struct_llama_model) i32;
pub extern fn llama_n_ctx_train(model: ?*const struct_llama_model) i32;
pub extern fn llama_n_embd(model: ?*const struct_llama_model) i32;
pub extern fn llama_rope_freq_scale_train(model: ?*const struct_llama_model) f32;
pub extern fn llama_model_meta_val_str(model: ?*const struct_llama_model, key: [*c]const u8, buf: [*c]u8, buf_size: usize) i32;
pub extern fn llama_model_meta_count(model: ?*const struct_llama_model) i32;
pub extern fn llama_model_meta_key_by_index(model: ?*const struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) i32;
pub extern fn llama_model_meta_val_str_by_index(model: ?*const struct_llama_model, i: i32, buf: [*c]u8, buf_size: usize) i32;
pub extern fn llama_model_desc(model: ?*const struct_llama_model, buf: [*c]u8, buf_size: usize) i32;
pub extern fn llama_model_size(model: ?*const struct_llama_model) u64;
pub extern fn llama_model_n_params(model: ?*const struct_llama_model) u64;
pub extern fn llama_get_model_tensor(model: ?*struct_llama_model, name: [*c]const u8) [*c]ggml.tensor;
pub extern fn llama_model_quantize(fname_inp: [*c]const u8, fname_out: [*c]const u8, params: [*c]const llama_model_quantize_params) u32;
pub extern fn llama_apply_lora_from_file(ctx: ?*struct_llama_context, path_lora: [*c]const u8, scale: f32, path_base_model: [*c]const u8, n_threads: i32) i32;
pub extern fn llama_model_apply_lora_from_file(model: ?*const struct_llama_model, path_lora: [*c]const u8, scale: f32, path_base_model: [*c]const u8, n_threads: i32) i32;
pub const struct_llama_kv_cache_view_cell = extern struct {
    pos: llama_pos = @import("std").mem.zeroes(llama_pos),
};
pub const struct_llama_kv_cache_view = extern struct {
    n_cells: i32 = @import("std").mem.zeroes(i32),
    n_max_seq: i32 = @import("std").mem.zeroes(i32),
    token_count: i32 = @import("std").mem.zeroes(i32),
    used_cells: i32 = @import("std").mem.zeroes(i32),
    max_contiguous: i32 = @import("std").mem.zeroes(i32),
    max_contiguous_idx: i32 = @import("std").mem.zeroes(i32),
    cells: [*c]struct_llama_kv_cache_view_cell = @import("std").mem.zeroes([*c]struct_llama_kv_cache_view_cell),
    cells_sequences: [*c]llama_seq_id = @import("std").mem.zeroes([*c]llama_seq_id),
};
pub extern fn llama_kv_cache_view_init(ctx: ?*const struct_llama_context, n_max_seq: i32) struct_llama_kv_cache_view;
pub extern fn llama_kv_cache_view_free(view: [*c]struct_llama_kv_cache_view) void;
pub extern fn llama_kv_cache_view_update(ctx: ?*const struct_llama_context, view: [*c]struct_llama_kv_cache_view) void;
pub extern fn llama_get_kv_cache_token_count(ctx: ?*const struct_llama_context) i32;
pub extern fn llama_get_kv_cache_used_cells(ctx: ?*const struct_llama_context) i32;
pub extern fn llama_kv_cache_clear(ctx: ?*struct_llama_context) void;
pub extern fn llama_kv_cache_seq_rm(ctx: ?*struct_llama_context, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) void;
pub extern fn llama_kv_cache_seq_cp(ctx: ?*struct_llama_context, seq_id_src: llama_seq_id, seq_id_dst: llama_seq_id, p0: llama_pos, p1: llama_pos) void;
pub extern fn llama_kv_cache_seq_keep(ctx: ?*struct_llama_context, seq_id: llama_seq_id) void;
pub extern fn llama_kv_cache_seq_shift(ctx: ?*struct_llama_context, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, delta: llama_pos) void;
pub extern fn llama_kv_cache_seq_div(ctx: ?*struct_llama_context, seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, d: c_int) void;
pub extern fn llama_get_state_size(ctx: ?*const struct_llama_context) usize;
pub extern fn llama_copy_state_data(ctx: ?*struct_llama_context, dst: [*c]u8) usize;
pub extern fn llama_set_state_data(ctx: ?*struct_llama_context, src: [*c]u8) usize;
pub extern fn llama_load_session_file(ctx: ?*struct_llama_context, path_session: [*c]const u8, tokens_out: [*c]llama_token, n_token_capacity: usize, n_token_count_out: [*c]usize) bool;
pub extern fn llama_save_session_file(ctx: ?*struct_llama_context, path_session: [*c]const u8, tokens: [*c]const llama_token, n_token_count: usize) bool;
pub extern fn llama_eval(ctx: ?*struct_llama_context, tokens: [*c]llama_token, n_tokens: i32, n_past: i32) c_int;
pub extern fn llama_eval_embd(ctx: ?*struct_llama_context, embd: [*c]f32, n_tokens: i32, n_past: i32) c_int;
pub extern fn llama_batch_get_one(tokens: [*c]llama_token, n_tokens: i32, pos_0: llama_pos, seq_id: llama_seq_id) struct_llama_batch;
pub extern fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) struct_llama_batch;
pub extern fn llama_batch_free(batch: struct_llama_batch) void;
pub extern fn llama_decode(ctx: ?*struct_llama_context, batch: struct_llama_batch) i32;
pub extern fn llama_set_n_threads(ctx: ?*struct_llama_context, n_threads: u32, n_threads_batch: u32) void;
pub extern fn llama_get_logits(ctx: ?*struct_llama_context) [*c]f32;
pub extern fn llama_get_logits_ith(ctx: ?*struct_llama_context, i: i32) [*c]f32;
pub extern fn llama_get_embeddings(ctx: ?*struct_llama_context) [*c]f32;
pub extern fn llama_token_get_text(model: ?*const struct_llama_model, token: llama_token) [*c]const u8;
pub extern fn llama_token_get_score(model: ?*const struct_llama_model, token: llama_token) f32;
pub extern fn llama_token_get_type(model: ?*const struct_llama_model, token: llama_token) enum_llama_token_type;
pub extern fn llama_token_bos(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_token_eos(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_token_nl(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_add_bos_token(model: ?*const struct_llama_model) i32;
pub extern fn llama_add_eos_token(model: ?*const struct_llama_model) i32;
pub extern fn llama_token_prefix(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_token_middle(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_token_suffix(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_token_eot(model: ?*const struct_llama_model) llama_token;
pub extern fn llama_tokenize(model: ?*const struct_llama_model, text: [*c]const u8, text_len: i32, tokens: [*c]llama_token, n_max_tokens: i32, add_bos: bool, special: bool) i32;
pub extern fn llama_token_to_piece(model: ?*const struct_llama_model, token: llama_token, buf: [*c]u8, length: i32) i32;
pub extern fn llama_grammar_init(rules: [*c][*c]const llama_grammar_element, n_rules: usize, start_rule_index: usize) ?*struct_llama_grammar;
pub extern fn llama_grammar_free(grammar: ?*struct_llama_grammar) void;
pub extern fn llama_grammar_copy(grammar: ?*const struct_llama_grammar) ?*struct_llama_grammar;
pub extern fn llama_set_rng_seed(ctx: ?*struct_llama_context, seed: u32) void;
pub extern fn llama_sample_repetition_penalties(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, last_tokens: [*c]const llama_token, penalty_last_n: usize, penalty_repeat: f32, penalty_freq: f32, penalty_present: f32) void;
pub extern fn llama_sample_apply_guidance(ctx: ?*struct_llama_context, logits: [*c]f32, logits_guidance: [*c]f32, scale: f32) void;
pub extern fn llama_sample_classifier_free_guidance(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, guidance_ctx: ?*struct_llama_context, scale: f32) void;
pub extern fn llama_sample_softmax(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array) void;
pub extern fn llama_sample_top_k(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, k: i32, min_keep: usize) void;
pub extern fn llama_sample_top_p(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, p: f32, min_keep: usize) void;
pub extern fn llama_sample_min_p(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, p: f32, min_keep: usize) void;
pub extern fn llama_sample_tail_free(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, z: f32, min_keep: usize) void;
pub extern fn llama_sample_typical(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, p: f32, min_keep: usize) void;
pub extern fn llama_sample_entropy(ctx: ?*struct_llama_context, candidates_p: [*c]llama_token_data_array, min_temp: f32, max_temp: f32, exponent_val: f32) void;
pub extern fn llama_sample_temp(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, temp: f32) void;
pub extern fn llama_sample_temperature(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, temp: f32) void;
pub extern fn llama_sample_grammar(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, grammar: ?*const struct_llama_grammar) void;
pub extern fn llama_sample_token_mirostat(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, tau: f32, eta: f32, m: i32, mu: [*c]f32) llama_token;
pub extern fn llama_sample_token_mirostat_v2(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array, tau: f32, eta: f32, mu: [*c]f32) llama_token;
pub extern fn llama_sample_token_greedy(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array) llama_token;
pub extern fn llama_sample_token(ctx: ?*struct_llama_context, candidates: [*c]llama_token_data_array) llama_token;
pub extern fn llama_grammar_accept_token(ctx: ?*struct_llama_context, grammar: ?*struct_llama_grammar, token: llama_token) void;
pub const struct_llama_beam_view = extern struct {
    tokens: [*c]const llama_token = @import("std").mem.zeroes([*c]const llama_token),
    n_tokens: usize = @import("std").mem.zeroes(usize),
    p: f32 = @import("std").mem.zeroes(f32),
    eob: bool = @import("std").mem.zeroes(bool),
};
pub const struct_llama_beams_state = extern struct {
    beam_views: [*c]struct_llama_beam_view = @import("std").mem.zeroes([*c]struct_llama_beam_view),
    n_beams: usize = @import("std").mem.zeroes(usize),
    common_prefix_length: usize = @import("std").mem.zeroes(usize),
    last_call: bool = @import("std").mem.zeroes(bool),
};
pub const llama_beam_search_callback_fn_t = ?*const fn (?*anyopaque, struct_llama_beams_state) callconv(.C) void;
pub extern fn llama_beam_search(ctx: ?*struct_llama_context, callback: llama_beam_search_callback_fn_t, callback_data: ?*anyopaque, n_beams: usize, n_past: i32, n_predict: i32) void;
pub extern fn llama_get_timings(ctx: ?*struct_llama_context) struct_llama_timings;
pub extern fn llama_print_timings(ctx: ?*struct_llama_context) void;
pub extern fn llama_reset_timings(ctx: ?*struct_llama_context) void;
pub extern fn llama_print_system_info() [*c]const u8;
pub extern fn llama_log_set(log_callback: ggml.log_callback, user_data: ?*anyopaque) void;
pub extern fn llama_dump_timing_info_yaml(stream: [*c]std.os.fd_t, ctx: ?*const struct_llama_context) void;

pub const LLAMA_API = "";
pub const LLAMA_DEFAULT_SEED = @import("std").zig.c_translation.promoteIntLiteral(c_int, 0xFFFFFFFF, .hex);
pub const LLAMA_MAX_RNG_STATE = @as(c_int, 64) * @as(c_int, 1024);
pub const LLAMA_FILE_MAGIC_GGLA = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 0x67676c61, .hex);
pub const LLAMA_FILE_MAGIC_GGSN = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 0x6767736e, .hex);
pub const LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN;
pub const LLAMA_SESSION_VERSION = @as(c_int, 4);

pub const llama_model = struct_llama_model;
pub const llama_context = struct_llama_context;
pub const llama_token_type = enum_llama_token_type;
pub const llama_ftype = enum_llama_ftype;
pub const llama_rope_scaling_type = enum_llama_rope_scaling_type;
pub const llama_split_mode = enum_llama_split_mode;
pub const llama_model_kv_override_type = enum_llama_model_kv_override_type;
pub const llama_model_kv_override = struct_llama_model_kv_override;
pub const llama_model_params = struct_llama_model_params;
pub const llama_context_params = struct_llama_context_params;
pub const llama_grammar = struct_llama_grammar;
pub const llama_gretype = enum_llama_gretype;
pub const llama_timings = struct_llama_timings;
pub const llama_kv_cache_view_cell = struct_llama_kv_cache_view_cell;
pub const llama_kv_cache_view = struct_llama_kv_cache_view;
pub const llama_beam_view = struct_llama_beam_view;
pub const llama_beams_state = struct_llama_beams_state;
