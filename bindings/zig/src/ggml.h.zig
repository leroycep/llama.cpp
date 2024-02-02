pub const fp16_t = u16;
pub extern fn fp16_to_fp32(x: fp16_t) f32;
pub extern fn fp32_to_fp16(x: f32) fp16_t;
pub extern fn fp16_to_fp32_row(x: [*c]const fp16_t, y: [*c]f32, n: c_int) void;
pub extern fn fp32_to_fp16_row(x: [*c]const f32, y: [*c]fp16_t, n: c_int) void;
pub const OBJECT_TENSOR: c_int = 0;
pub const OBJECT_GRAPH: c_int = 1;
pub const OBJECT_WORK_BUFFER: c_int = 2;
pub const object_type = c_uint;
pub const object = extern struct {
    offs: usize = @import("std").mem.zeroes(usize),
    size: usize = @import("std").mem.zeroes(usize),
    next: [*c]object = @import("std").mem.zeroes([*c]object),
    type: object_type = @import("std").mem.zeroes(object_type),
    padding: [4]u8 = @import("std").mem.zeroes([4]u8),
};
pub const context = opaque {};
pub const TYPE_F32: c_int = 0;
pub const TYPE_F16: c_int = 1;
pub const TYPE_Q4_0: c_int = 2;
pub const TYPE_Q4_1: c_int = 3;
pub const TYPE_Q5_0: c_int = 6;
pub const TYPE_Q5_1: c_int = 7;
pub const TYPE_Q8_0: c_int = 8;
pub const TYPE_Q8_1: c_int = 9;
pub const TYPE_Q2_K: c_int = 10;
pub const TYPE_Q3_K: c_int = 11;
pub const TYPE_Q4_K: c_int = 12;
pub const TYPE_Q5_K: c_int = 13;
pub const TYPE_Q6_K: c_int = 14;
pub const TYPE_Q8_K: c_int = 15;
pub const TYPE_IQ2_XXS: c_int = 16;
pub const TYPE_IQ2_XS: c_int = 17;
pub const TYPE_IQ3_XXS: c_int = 18;
pub const TYPE_I8: c_int = 19;
pub const TYPE_I16: c_int = 20;
pub const TYPE_I32: c_int = 21;
pub const TYPE_COUNT: c_int = 22;
pub const @"type" = c_uint;
pub const PREC_DEFAULT: c_int = 0;
pub const PREC_F32: c_int = 1;
pub const prec = c_uint;
pub const BACKEND_CPU: c_int = 0;
pub const BACKEND_GPU: c_int = 10;
pub const BACKEND_GPU_SPLIT: c_int = 20;
pub const backend_type = c_uint;
pub const FTYPE_UNKNOWN: c_int = -1;
pub const FTYPE_ALL_F32: c_int = 0;
pub const FTYPE_MOSTLY_F16: c_int = 1;
pub const FTYPE_MOSTLY_Q4_0: c_int = 2;
pub const FTYPE_MOSTLY_Q4_1: c_int = 3;
pub const FTYPE_MOSTLY_Q4_1_SOME_F16: c_int = 4;
pub const FTYPE_MOSTLY_Q8_0: c_int = 7;
pub const FTYPE_MOSTLY_Q5_0: c_int = 8;
pub const FTYPE_MOSTLY_Q5_1: c_int = 9;
pub const FTYPE_MOSTLY_Q2_K: c_int = 10;
pub const FTYPE_MOSTLY_Q3_K: c_int = 11;
pub const FTYPE_MOSTLY_Q4_K: c_int = 12;
pub const FTYPE_MOSTLY_Q5_K: c_int = 13;
pub const FTYPE_MOSTLY_Q6_K: c_int = 14;
pub const FTYPE_MOSTLY_IQ2_XXS: c_int = 15;
pub const FTYPE_MOSTLY_IQ2_XS: c_int = 16;
pub const FTYPE_MOSTLY_IQ3_XXS: c_int = 17;
pub const ftype = c_int;
pub const OP_NONE: c_int = 0;
pub const OP_DUP: c_int = 1;
pub const OP_ADD: c_int = 2;
pub const OP_ADD1: c_int = 3;
pub const OP_ACC: c_int = 4;
pub const OP_SUB: c_int = 5;
pub const OP_MUL: c_int = 6;
pub const OP_DIV: c_int = 7;
pub const OP_SQR: c_int = 8;
pub const OP_SQRT: c_int = 9;
pub const OP_LOG: c_int = 10;
pub const OP_SUM: c_int = 11;
pub const OP_SUM_ROWS: c_int = 12;
pub const OP_MEAN: c_int = 13;
pub const OP_ARGMAX: c_int = 14;
pub const OP_REPEAT: c_int = 15;
pub const OP_REPEAT_BACK: c_int = 16;
pub const OP_CONCAT: c_int = 17;
pub const OP_SILU_BACK: c_int = 18;
pub const OP_NORM: c_int = 19;
pub const OP_RMS_NORM: c_int = 20;
pub const OP_RMS_NORM_BACK: c_int = 21;
pub const OP_GROUP_NORM: c_int = 22;
pub const OP_MUL_MAT: c_int = 23;
pub const OP_MUL_MAT_ID: c_int = 24;
pub const OP_OUT_PROD: c_int = 25;
pub const OP_SCALE: c_int = 26;
pub const OP_SET: c_int = 27;
pub const OP_CPY: c_int = 28;
pub const OP_CONT: c_int = 29;
pub const OP_RESHAPE: c_int = 30;
pub const OP_VIEW: c_int = 31;
pub const OP_PERMUTE: c_int = 32;
pub const OP_TRANSPOSE: c_int = 33;
pub const OP_GET_ROWS: c_int = 34;
pub const OP_GET_ROWS_BACK: c_int = 35;
pub const OP_DIAG: c_int = 36;
pub const OP_DIAG_MASK_INF: c_int = 37;
pub const OP_DIAG_MASK_ZERO: c_int = 38;
pub const OP_SOFT_MAX: c_int = 39;
pub const OP_SOFT_MAX_BACK: c_int = 40;
pub const OP_ROPE: c_int = 41;
pub const OP_ROPE_BACK: c_int = 42;
pub const OP_ALIBI: c_int = 43;
pub const OP_CLAMP: c_int = 44;
pub const OP_CONV_TRANSPOSE_1D: c_int = 45;
pub const OP_IM2COL: c_int = 46;
pub const OP_CONV_TRANSPOSE_2D: c_int = 47;
pub const OP_POOL_1D: c_int = 48;
pub const OP_POOL_2D: c_int = 49;
pub const OP_UPSCALE: c_int = 50;
pub const OP_PAD: c_int = 51;
pub const OP_ARGSORT: c_int = 52;
pub const OP_LEAKY_RELU: c_int = 53;
pub const OP_FLASH_ATTN: c_int = 54;
pub const OP_FLASH_FF: c_int = 55;
pub const OP_FLASH_ATTN_BACK: c_int = 56;
pub const OP_WIN_PART: c_int = 57;
pub const OP_WIN_UNPART: c_int = 58;
pub const OP_GET_REL_POS: c_int = 59;
pub const OP_ADD_REL_POS: c_int = 60;
pub const OP_UNARY: c_int = 61;
pub const OP_MAP_UNARY: c_int = 62;
pub const OP_MAP_BINARY: c_int = 63;
pub const OP_MAP_CUSTOM1_F32: c_int = 64;
pub const OP_MAP_CUSTOM2_F32: c_int = 65;
pub const OP_MAP_CUSTOM3_F32: c_int = 66;
pub const OP_MAP_CUSTOM1: c_int = 67;
pub const OP_MAP_CUSTOM2: c_int = 68;
pub const OP_MAP_CUSTOM3: c_int = 69;
pub const OP_CROSS_ENTROPY_LOSS: c_int = 70;
pub const OP_CROSS_ENTROPY_LOSS_BACK: c_int = 71;
pub const OP_COUNT: c_int = 72;
pub const enum_ggml_op = c_uint;
pub const UNARY_OP_ABS: c_int = 0;
pub const UNARY_OP_SGN: c_int = 1;
pub const UNARY_OP_NEG: c_int = 2;
pub const UNARY_OP_STEP: c_int = 3;
pub const UNARY_OP_TANH: c_int = 4;
pub const UNARY_OP_ELU: c_int = 5;
pub const UNARY_OP_RELU: c_int = 6;
pub const UNARY_OP_GELU: c_int = 7;
pub const UNARY_OP_GELU_QUICK: c_int = 8;
pub const UNARY_OP_SILU: c_int = 9;
pub const UNARY_OP_HARDSWISH: c_int = 10;
pub const UNARY_OP_HARDSIGMOID: c_int = 11;
pub const UNARY_OP_COUNT: c_int = 12;
pub const enum_ggml_unary_op = c_uint;
pub const LOG_LEVEL_ERROR: c_int = 2;
pub const LOG_LEVEL_WARN: c_int = 3;
pub const LOG_LEVEL_INFO: c_int = 4;
pub const LOG_LEVEL_DEBUG: c_int = 5;
pub const enum_ggml_log_level = c_uint;
pub const OBJECT_SIZE: usize = @sizeOf(object);
pub const struct_ggml_backend_buffer = opaque {};
pub const struct_ggml_tensor = extern struct {
    type: @"type" = @import("std").mem.zeroes(@"type"),
    backend: backend_type = @import("std").mem.zeroes(backend_type),
    buffer: ?*struct_ggml_backend_buffer = @import("std").mem.zeroes(?*struct_ggml_backend_buffer),
    ne: [4]i64 = @import("std").mem.zeroes([4]i64),
    nb: [4]usize = @import("std").mem.zeroes([4]usize),
    op: enum_ggml_op = @import("std").mem.zeroes(enum_ggml_op),
    op_params: [16]i32 = @import("std").mem.zeroes([16]i32),
    is_param: bool = @import("std").mem.zeroes(bool),
    grad: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    src: [10][*c]struct_ggml_tensor = @import("std").mem.zeroes([10][*c]struct_ggml_tensor),
    perf_runs: c_int = @import("std").mem.zeroes(c_int),
    perf_cycles: i64 = @import("std").mem.zeroes(i64),
    perf_time_us: i64 = @import("std").mem.zeroes(i64),
    view_src: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    view_offs: usize = @import("std").mem.zeroes(usize),
    data: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    name: [64]u8 = @import("std").mem.zeroes([64]u8),
    extra: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    padding: [8]u8 = @import("std").mem.zeroes([8]u8),
};
pub const TENSOR_SIZE: usize = @sizeOf(struct_ggml_tensor);
pub const struct_ggml_cplan = extern struct {
    work_size: usize = @import("std").mem.zeroes(usize),
    work_data: [*c]u8 = @import("std").mem.zeroes([*c]u8),
    n_threads: c_int = @import("std").mem.zeroes(c_int),
    abort_callback: ?*const fn (?*anyopaque) callconv(.C) bool = @import("std").mem.zeroes(?*const fn (?*anyopaque) callconv(.C) bool),
    abort_callback_data: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT: c_int = 0;
pub const CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT: c_int = 1;
pub const CGRAPH_EVAL_ORDER_COUNT: c_int = 2;
pub const enum_ggml_cgraph_eval_order = c_uint;
pub const struct_ggml_hash_set = extern struct {
    size: usize = @import("std").mem.zeroes(usize),
    keys: [*c][*c]struct_ggml_tensor = @import("std").mem.zeroes([*c][*c]struct_ggml_tensor),
};
pub const struct_ggml_cgraph = extern struct {
    size: c_int = @import("std").mem.zeroes(c_int),
    n_nodes: c_int = @import("std").mem.zeroes(c_int),
    n_leafs: c_int = @import("std").mem.zeroes(c_int),
    nodes: [*c][*c]struct_ggml_tensor = @import("std").mem.zeroes([*c][*c]struct_ggml_tensor),
    grads: [*c][*c]struct_ggml_tensor = @import("std").mem.zeroes([*c][*c]struct_ggml_tensor),
    leafs: [*c][*c]struct_ggml_tensor = @import("std").mem.zeroes([*c][*c]struct_ggml_tensor),
    visited_hash_table: struct_ggml_hash_set = @import("std").mem.zeroes(struct_ggml_hash_set),
    order: enum_ggml_cgraph_eval_order = @import("std").mem.zeroes(enum_ggml_cgraph_eval_order),
    perf_runs: c_int = @import("std").mem.zeroes(c_int),
    perf_cycles: i64 = @import("std").mem.zeroes(i64),
    perf_time_us: i64 = @import("std").mem.zeroes(i64),
};
pub const struct_ggml_scratch = extern struct {
    offs: usize = @import("std").mem.zeroes(usize),
    size: usize = @import("std").mem.zeroes(usize),
    data: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub const struct_ggml_init_params = extern struct {
    mem_size: usize = @import("std").mem.zeroes(usize),
    mem_buffer: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
    no_alloc: bool = @import("std").mem.zeroes(bool),
};
pub const TASK_INIT: c_int = 0;
pub const TASK_COMPUTE: c_int = 1;
pub const TASK_FINALIZE: c_int = 2;
pub const enum_ggml_task_type = c_uint;
pub const struct_ggml_compute_params = extern struct {
    type: enum_ggml_task_type = @import("std").mem.zeroes(enum_ggml_task_type),
    ith: c_int = @import("std").mem.zeroes(c_int),
    nth: c_int = @import("std").mem.zeroes(c_int),
    wsize: usize = @import("std").mem.zeroes(usize),
    wdata: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub extern fn time_init() void;
pub extern fn time_ms() i64;
pub extern fn time_us() i64;
pub extern fn cycles() i64;
pub extern fn cycles_per_ms() i64;
pub extern fn print_backtrace() void;
pub extern fn numa_init() void;
pub extern fn is_numa() bool;
pub extern fn print_object(obj: [*c]const object) void;
pub extern fn print_objects(ctx: ?*const context) void;
pub extern fn nelements(tensor: [*c]const struct_ggml_tensor) i64;
pub extern fn nrows(tensor: [*c]const struct_ggml_tensor) i64;
pub extern fn nbytes(tensor: [*c]const struct_ggml_tensor) usize;
pub extern fn nbytes_pad(tensor: [*c]const struct_ggml_tensor) usize;
pub extern fn blck_size(@"type": type) c_int;
pub extern fn type_size(@"type": type) usize;
pub extern fn row_size(@"type": type, ne: i64) usize;
pub extern fn type_sizef(@"type": type) f64;
pub extern fn type_name(@"type": type) [*c]const u8;
pub extern fn op_name(op: enum_ggml_op) [*c]const u8;
pub extern fn op_symbol(op: enum_ggml_op) [*c]const u8;
pub extern fn unary_op_name(op: enum_ggml_unary_op) [*c]const u8;
pub extern fn op_desc(t: [*c]const struct_ggml_tensor) [*c]const u8;
pub extern fn element_size(tensor: [*c]const struct_ggml_tensor) usize;
pub extern fn is_quantized(@"type": type) bool;
pub extern fn ftype_to_ggml_type(ftype: ftype) type;
pub extern fn is_transposed(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_contiguous(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_permuted(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_scalar(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_vector(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_matrix(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn is_3d(tensor: [*c]const struct_ggml_tensor) bool;
pub extern fn n_dims(tensor: [*c]const struct_ggml_tensor) c_int;
pub extern fn are_same_shape(t0: [*c]const struct_ggml_tensor, t1: [*c]const struct_ggml_tensor) bool;
pub extern fn tensor_overhead() usize;
pub extern fn init(params: struct_ggml_init_params) ?*context;
pub extern fn free(ctx: ?*context) void;
pub extern fn used_mem(ctx: ?*const context) usize;
pub extern fn set_scratch(ctx: ?*context, scratch: struct_ggml_scratch) usize;
pub extern fn get_no_alloc(ctx: ?*context) bool;
pub extern fn set_no_alloc(ctx: ?*context, no_alloc: bool) void;
pub extern fn get_mem_buffer(ctx: ?*const context) ?*anyopaque;
pub extern fn get_mem_size(ctx: ?*const context) usize;
pub extern fn get_max_tensor_size(ctx: ?*const context) usize;
pub extern fn new_tensor(ctx: ?*context, @"type": type, n_dims: c_int, ne: [*c]const i64) [*c]struct_ggml_tensor;
pub extern fn new_tensor_1d(ctx: ?*context, @"type": type, ne0: i64) [*c]struct_ggml_tensor;
pub extern fn new_tensor_2d(ctx: ?*context, @"type": type, ne0: i64, ne1: i64) [*c]struct_ggml_tensor;
pub extern fn new_tensor_3d(ctx: ?*context, @"type": type, ne0: i64, ne1: i64, ne2: i64) [*c]struct_ggml_tensor;
pub extern fn new_tensor_4d(ctx: ?*context, @"type": type, ne0: i64, ne1: i64, ne2: i64, ne3: i64) [*c]struct_ggml_tensor;
pub extern fn new_i32(ctx: ?*context, value: i32) [*c]struct_ggml_tensor;
pub extern fn new_f32(ctx: ?*context, value: f32) [*c]struct_ggml_tensor;
pub extern fn dup_tensor(ctx: ?*context, src: [*c]const struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn view_tensor(ctx: ?*context, src: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn get_first_tensor(ctx: ?*const context) [*c]struct_ggml_tensor;
pub extern fn get_next_tensor(ctx: ?*const context, tensor: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn get_tensor(ctx: ?*context, name: [*c]const u8) [*c]struct_ggml_tensor;
pub extern fn set_zero(tensor: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn set_i32(tensor: [*c]struct_ggml_tensor, value: i32) [*c]struct_ggml_tensor;
pub extern fn set_f32(tensor: [*c]struct_ggml_tensor, value: f32) [*c]struct_ggml_tensor;
pub extern fn unravel_index(tensor: [*c]const struct_ggml_tensor, i: i64, @"i0": [*c]i64, @"i1": [*c]i64, @"i2": [*c]i64, @"i3": [*c]i64) void;
pub extern fn get_i32_1d(tensor: [*c]const struct_ggml_tensor, i: c_int) i32;
pub extern fn set_i32_1d(tensor: [*c]const struct_ggml_tensor, i: c_int, value: i32) void;
pub extern fn get_i32_nd(tensor: [*c]const struct_ggml_tensor, @"i0": c_int, @"i1": c_int, @"i2": c_int, @"i3": c_int) i32;
pub extern fn set_i32_nd(tensor: [*c]const struct_ggml_tensor, @"i0": c_int, @"i1": c_int, @"i2": c_int, @"i3": c_int, value: i32) void;
pub extern fn get_f32_1d(tensor: [*c]const struct_ggml_tensor, i: c_int) f32;
pub extern fn set_f32_1d(tensor: [*c]const struct_ggml_tensor, i: c_int, value: f32) void;
pub extern fn get_f32_nd(tensor: [*c]const struct_ggml_tensor, @"i0": c_int, @"i1": c_int, @"i2": c_int, @"i3": c_int) f32;
pub extern fn set_f32_nd(tensor: [*c]const struct_ggml_tensor, @"i0": c_int, @"i1": c_int, @"i2": c_int, @"i3": c_int, value: f32) void;
pub extern fn get_data(tensor: [*c]const struct_ggml_tensor) ?*anyopaque;
pub extern fn get_data_f32(tensor: [*c]const struct_ggml_tensor) [*c]f32;
pub extern fn get_unary_op(tensor: [*c]const struct_ggml_tensor) enum_ggml_unary_op;
pub extern fn get_name(tensor: [*c]const struct_ggml_tensor) [*c]const u8;
pub extern fn set_name(tensor: [*c]struct_ggml_tensor, name: [*c]const u8) [*c]struct_ggml_tensor;
pub extern fn format_name(tensor: [*c]struct_ggml_tensor, fmt: [*c]const u8, ...) [*c]struct_ggml_tensor;
pub extern fn dup(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn dup_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn add(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn add_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn add_cast(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, @"type": type) [*c]struct_ggml_tensor;
pub extern fn add1(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn add1_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn acc(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn acc_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn sub(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sub_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn mul(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn mul_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn div(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn div_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sqr(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sqr_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sqrt(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sqrt_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn log(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn log_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sum(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sum_rows(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn mean(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn argmax(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn repeat(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn repeat_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn concat(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn abs(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn abs_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sgn(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn sgn_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn neg(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn neg_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn step(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn step_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn tanh(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn tanh_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn elu(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn elu_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn relu(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn leaky_relu(ctx: ?*context, a: [*c]struct_ggml_tensor, negative_slope: f32, inplace: bool) [*c]struct_ggml_tensor;
pub extern fn relu_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn gelu(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn gelu_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn gelu_quick(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn gelu_quick_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn silu(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn silu_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn silu_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn hardswish(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn hardsigmoid(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn norm(ctx: ?*context, a: [*c]struct_ggml_tensor, eps: f32) [*c]struct_ggml_tensor;
pub extern fn norm_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, eps: f32) [*c]struct_ggml_tensor;
pub extern fn rms_norm(ctx: ?*context, a: [*c]struct_ggml_tensor, eps: f32) [*c]struct_ggml_tensor;
pub extern fn rms_norm_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, eps: f32) [*c]struct_ggml_tensor;
pub extern fn group_norm(ctx: ?*context, a: [*c]struct_ggml_tensor, n_groups: c_int) [*c]struct_ggml_tensor;
pub extern fn group_norm_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, n_groups: c_int) [*c]struct_ggml_tensor;
pub extern fn rms_norm_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, eps: f32) [*c]struct_ggml_tensor;
pub extern fn mul_mat(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn mul_mat_set_prec(a: [*c]struct_ggml_tensor, prec: prec) void;
pub extern fn mul_mat_id(ctx: ?*context, as: [*c]const [*c]struct_ggml_tensor, n_as: c_int, ids: [*c]struct_ggml_tensor, id: c_int, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn out_prod(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn scale(ctx: ?*context, a: [*c]struct_ggml_tensor, s: f32) [*c]struct_ggml_tensor;
pub extern fn scale_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, s: f32) [*c]struct_ggml_tensor;
pub extern fn set(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn set_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, nb2: usize, nb3: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn set_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, offset: usize) [*c]struct_ggml_tensor;
pub extern fn set_1d_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, offset: usize) [*c]struct_ggml_tensor;
pub extern fn set_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn set_2d_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, nb1: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn cpy(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn cast(ctx: ?*context, a: [*c]struct_ggml_tensor, @"type": type) [*c]struct_ggml_tensor;
pub extern fn cont(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn cont_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64) [*c]struct_ggml_tensor;
pub extern fn cont_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64) [*c]struct_ggml_tensor;
pub extern fn cont_3d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64) [*c]struct_ggml_tensor;
pub extern fn cont_4d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) [*c]struct_ggml_tensor;
pub extern fn reshape(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn reshape_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64) [*c]struct_ggml_tensor;
pub extern fn reshape_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64) [*c]struct_ggml_tensor;
pub extern fn reshape_3d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64) [*c]struct_ggml_tensor;
pub extern fn reshape_4d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) [*c]struct_ggml_tensor;
pub extern fn view_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, offset: usize) [*c]struct_ggml_tensor;
pub extern fn view_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn view_3d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn view_4d(ctx: ?*context, a: [*c]struct_ggml_tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) [*c]struct_ggml_tensor;
pub extern fn permute(ctx: ?*context, a: [*c]struct_ggml_tensor, axis0: c_int, axis1: c_int, axis2: c_int, axis3: c_int) [*c]struct_ggml_tensor;
pub extern fn transpose(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn get_rows(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn get_rows_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn diag(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn diag_mask_inf(ctx: ?*context, a: [*c]struct_ggml_tensor, n_past: c_int) [*c]struct_ggml_tensor;
pub extern fn diag_mask_inf_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, n_past: c_int) [*c]struct_ggml_tensor;
pub extern fn diag_mask_zero(ctx: ?*context, a: [*c]struct_ggml_tensor, n_past: c_int) [*c]struct_ggml_tensor;
pub extern fn diag_mask_zero_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, n_past: c_int) [*c]struct_ggml_tensor;
pub extern fn soft_max(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn soft_max_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn soft_max_ext(ctx: ?*context, a: [*c]struct_ggml_tensor, mask: [*c]struct_ggml_tensor, scale: f32) [*c]struct_ggml_tensor;
pub extern fn soft_max_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn soft_max_back_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn rope(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, mode: c_int, n_ctx: c_int) [*c]struct_ggml_tensor;
pub extern fn rope_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, mode: c_int, n_ctx: c_int) [*c]struct_ggml_tensor;
pub extern fn rope_custom(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, mode: c_int, n_ctx: c_int, n_orig_ctx: c_int, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) [*c]struct_ggml_tensor;
pub extern fn rope_custom_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, mode: c_int, n_ctx: c_int, n_orig_ctx: c_int, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) [*c]struct_ggml_tensor;
pub extern fn rope_yarn_corr_dims(n_dims: c_int, n_orig_ctx: c_int, freq_base: f32, beta_fast: f32, beta_slow: f32, dims: [*c]f32) void;
pub extern fn rope_xpos_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, base: f32, down: bool) [*c]struct_ggml_tensor;
pub extern fn rope_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, n_dims: c_int, mode: c_int, n_ctx: c_int, n_orig_ctx: c_int, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, xpos_base: f32, xpos_down: bool) [*c]struct_ggml_tensor;
pub extern fn alibi(ctx: ?*context, a: [*c]struct_ggml_tensor, n_past: c_int, n_head: c_int, bias_max: f32) [*c]struct_ggml_tensor;
pub extern fn clamp(ctx: ?*context, a: [*c]struct_ggml_tensor, min: f32, max: f32) [*c]struct_ggml_tensor;
pub extern fn im2col(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int, is_2D: bool, dst_type: type) [*c]struct_ggml_tensor;
pub extern fn conv_depthwise_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) [*c]struct_ggml_tensor;
pub extern fn conv_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s0: c_int, p0: c_int, d0: c_int) [*c]struct_ggml_tensor;
pub extern fn conv_1d_ph(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s: c_int, d: c_int) [*c]struct_ggml_tensor;
pub extern fn conv_transpose_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s0: c_int, p0: c_int, d0: c_int) [*c]struct_ggml_tensor;
pub extern fn conv_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) [*c]struct_ggml_tensor;
pub extern fn conv_2d_sk_p0(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn conv_2d_s1_ph(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn conv_transpose_2d_p0(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, stride: c_int) [*c]struct_ggml_tensor;
pub const OP_POOL_MAX: c_int = 0;
pub const OP_POOL_AVG: c_int = 1;
pub const OP_POOL_COUNT: c_int = 2;
pub const enum_ggml_op_pool = c_uint;
pub extern fn pool_1d(ctx: ?*context, a: [*c]struct_ggml_tensor, op: enum_ggml_op_pool, k0: c_int, s0: c_int, p0: c_int) [*c]struct_ggml_tensor;
pub extern fn pool_2d(ctx: ?*context, a: [*c]struct_ggml_tensor, op: enum_ggml_op_pool, k0: c_int, k1: c_int, s0: c_int, s1: c_int, p0: f32, p1: f32) [*c]struct_ggml_tensor;
pub extern fn upscale(ctx: ?*context, a: [*c]struct_ggml_tensor, scale_factor: c_int) [*c]struct_ggml_tensor;
pub extern fn pad(ctx: ?*context, a: [*c]struct_ggml_tensor, p0: c_int, p1: c_int, p2: c_int, p3: c_int) [*c]struct_ggml_tensor;
pub const SORT_ASC: c_int = 0;
pub const SORT_DESC: c_int = 1;
pub const enum_ggml_sort_order = c_uint;
pub extern fn argsort(ctx: ?*context, a: [*c]struct_ggml_tensor, order: enum_ggml_sort_order) [*c]struct_ggml_tensor;
pub extern fn top_k(ctx: ?*context, a: [*c]struct_ggml_tensor, k: c_int) [*c]struct_ggml_tensor;
pub extern fn flash_attn(ctx: ?*context, q: [*c]struct_ggml_tensor, k: [*c]struct_ggml_tensor, v: [*c]struct_ggml_tensor, masked: bool) [*c]struct_ggml_tensor;
pub extern fn flash_attn_back(ctx: ?*context, q: [*c]struct_ggml_tensor, k: [*c]struct_ggml_tensor, v: [*c]struct_ggml_tensor, d: [*c]struct_ggml_tensor, masked: bool) [*c]struct_ggml_tensor;
pub extern fn flash_ff(ctx: ?*context, a: [*c]struct_ggml_tensor, b0: [*c]struct_ggml_tensor, b1: [*c]struct_ggml_tensor, c0: [*c]struct_ggml_tensor, c1: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn win_part(ctx: ?*context, a: [*c]struct_ggml_tensor, w: c_int) [*c]struct_ggml_tensor;
pub extern fn win_unpart(ctx: ?*context, a: [*c]struct_ggml_tensor, w0: c_int, h0: c_int, w: c_int) [*c]struct_ggml_tensor;
pub extern fn unary(ctx: ?*context, a: [*c]struct_ggml_tensor, op: enum_ggml_unary_op) [*c]struct_ggml_tensor;
pub extern fn unary_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, op: enum_ggml_unary_op) [*c]struct_ggml_tensor;
pub extern fn get_rel_pos(ctx: ?*context, a: [*c]struct_ggml_tensor, qh: c_int, kh: c_int) [*c]struct_ggml_tensor;
pub extern fn add_rel_pos(ctx: ?*context, a: [*c]struct_ggml_tensor, pw: [*c]struct_ggml_tensor, ph: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn add_rel_pos_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, pw: [*c]struct_ggml_tensor, ph: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub const unary_op_f32_t = ?*const fn (c_int, [*c]f32, [*c]const f32) callconv(.C) void;
pub const binary_op_f32_t = ?*const fn (c_int, [*c]f32, [*c]const f32, [*c]const f32) callconv(.C) void;
pub const custom1_op_f32_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor) callconv(.C) void;
pub const custom2_op_f32_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor) callconv(.C) void;
pub const custom3_op_f32_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor) callconv(.C) void;
pub extern fn map_unary_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: unary_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_unary_inplace_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: unary_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_binary_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: binary_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_binary_inplace_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: binary_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom1_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: custom1_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom1_inplace_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: custom1_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom2_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: custom2_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom2_inplace_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: custom2_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom3_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor, fun: custom3_op_f32_t) [*c]struct_ggml_tensor;
pub extern fn map_custom3_inplace_f32(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor, fun: custom3_op_f32_t) [*c]struct_ggml_tensor;
pub const custom1_op_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor, c_int, c_int, ?*anyopaque) callconv(.C) void;
pub const custom2_op_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor, c_int, c_int, ?*anyopaque) callconv(.C) void;
pub const custom3_op_t = ?*const fn ([*c]struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor, [*c]const struct_ggml_tensor, c_int, c_int, ?*anyopaque) callconv(.C) void;
pub extern fn map_custom1(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: custom1_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn map_custom1_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, fun: custom1_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn map_custom2(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: custom2_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn map_custom2_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, fun: custom2_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn map_custom3(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor, fun: custom3_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn map_custom3_inplace(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor, fun: custom3_op_t, n_tasks: c_int, userdata: ?*anyopaque) [*c]struct_ggml_tensor;
pub extern fn cross_entropy_loss(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn cross_entropy_loss_back(ctx: ?*context, a: [*c]struct_ggml_tensor, b: [*c]struct_ggml_tensor, c: [*c]struct_ggml_tensor) [*c]struct_ggml_tensor;
pub extern fn set_param(ctx: ?*context, tensor: [*c]struct_ggml_tensor) void;
pub extern fn build_forward_expand(cgraph: [*c]struct_ggml_cgraph, tensor: [*c]struct_ggml_tensor) void;
pub extern fn build_backward_expand(ctx: ?*context, gf: [*c]struct_ggml_cgraph, gb: [*c]struct_ggml_cgraph, keep: bool) void;
pub extern fn new_graph(ctx: ?*context) [*c]struct_ggml_cgraph;
pub extern fn new_graph_custom(ctx: ?*context, size: usize, grads: bool) [*c]struct_ggml_cgraph;
pub extern fn graph_dup(ctx: ?*context, cgraph: [*c]struct_ggml_cgraph) [*c]struct_ggml_cgraph;
pub extern fn graph_view(cgraph: [*c]struct_ggml_cgraph, @"i0": c_int, @"i1": c_int) struct_ggml_cgraph;
pub extern fn graph_cpy(src: [*c]struct_ggml_cgraph, dst: [*c]struct_ggml_cgraph) void;
pub extern fn graph_reset(cgraph: [*c]struct_ggml_cgraph) void;
pub extern fn graph_clear(cgraph: [*c]struct_ggml_cgraph) void;
pub extern fn graph_overhead() usize;
pub extern fn graph_overhead_custom(size: usize, grads: bool) usize;
pub extern fn graph_plan(cgraph: [*c]const struct_ggml_cgraph, n_threads: c_int) struct_ggml_cplan;
pub extern fn graph_compute(cgraph: [*c]struct_ggml_cgraph, cplan: [*c]struct_ggml_cplan) c_int;
pub extern fn graph_compute_with_ctx(ctx: ?*context, cgraph: [*c]struct_ggml_cgraph, n_threads: c_int) void;
pub extern fn graph_get_tensor(cgraph: [*c]struct_ggml_cgraph, name: [*c]const u8) [*c]struct_ggml_tensor;
pub extern fn graph_export(cgraph: [*c]const struct_ggml_cgraph, fname: [*c]const u8) void;
pub extern fn graph_import(fname: [*c]const u8, ctx_data: [*c]?*context, ctx_eval: [*c]?*context) [*c]struct_ggml_cgraph;
pub extern fn graph_print(cgraph: [*c]const struct_ggml_cgraph) void;
pub extern fn graph_dump_dot(gb: [*c]const struct_ggml_cgraph, gf: [*c]const struct_ggml_cgraph, filename: [*c]const u8) void;
pub extern fn build_backward_gradient_checkpointing(ctx: ?*context, gf: [*c]struct_ggml_cgraph, gb: [*c]struct_ggml_cgraph, gb_tmp: [*c]struct_ggml_cgraph, checkpoints: [*c][*c]struct_ggml_tensor, n_checkpoints: c_int) void;
pub const OPT_ADAM: c_int = 0;
pub const OPT_LBFGS: c_int = 1;
pub const enum_ggml_opt_type = c_uint;
pub const LINESEARCH_DEFAULT: c_int = 1;
pub const LINESEARCH_BACKTRACKING_ARMIJO: c_int = 0;
pub const LINESEARCH_BACKTRACKING_WOLFE: c_int = 1;
pub const LINESEARCH_BACKTRACKING_STRONG_WOLFE: c_int = 2;
pub const enum_ggml_linesearch = c_uint;
pub const OPT_OK: c_int = 0;
pub const OPT_DID_NOT_CONVERGE: c_int = 1;
pub const OPT_NO_CONTEXT: c_int = 2;
pub const OPT_INVALID_WOLFE: c_int = 3;
pub const OPT_FAIL: c_int = 4;
pub const OPT_CANCEL: c_int = 5;
pub const LINESEARCH_FAIL: c_int = -128;
pub const LINESEARCH_MINIMUM_STEP: c_int = -127;
pub const LINESEARCH_MAXIMUM_STEP: c_int = -126;
pub const LINESEARCH_MAXIMUM_ITERATIONS: c_int = -125;
pub const LINESEARCH_INVALID_PARAMETERS: c_int = -124;
pub const enum_ggml_opt_result = c_int;
pub const opt_callback = ?*const fn (?*anyopaque, c_int, [*c]f32, [*c]bool) callconv(.C) void;
pub const log_callback = ?*const fn (enum_ggml_log_level, [*c]const u8, ?*anyopaque) callconv(.C) void;
const struct_unnamed_1 = extern struct {
    n_iter: c_int = @import("std").mem.zeroes(c_int),
    sched: f32 = @import("std").mem.zeroes(f32),
    decay: f32 = @import("std").mem.zeroes(f32),
    decay_min_ndim: c_int = @import("std").mem.zeroes(c_int),
    alpha: f32 = @import("std").mem.zeroes(f32),
    beta1: f32 = @import("std").mem.zeroes(f32),
    beta2: f32 = @import("std").mem.zeroes(f32),
    eps: f32 = @import("std").mem.zeroes(f32),
    eps_f: f32 = @import("std").mem.zeroes(f32),
    eps_g: f32 = @import("std").mem.zeroes(f32),
    gclip: f32 = @import("std").mem.zeroes(f32),
};
const struct_unnamed_2 = extern struct {
    m: c_int = @import("std").mem.zeroes(c_int),
    n_iter: c_int = @import("std").mem.zeroes(c_int),
    max_linesearch: c_int = @import("std").mem.zeroes(c_int),
    eps: f32 = @import("std").mem.zeroes(f32),
    ftol: f32 = @import("std").mem.zeroes(f32),
    wolfe: f32 = @import("std").mem.zeroes(f32),
    min_step: f32 = @import("std").mem.zeroes(f32),
    max_step: f32 = @import("std").mem.zeroes(f32),
    linesearch: enum_ggml_linesearch = @import("std").mem.zeroes(enum_ggml_linesearch),
};
pub const struct_ggml_opt_params = extern struct {
    type: enum_ggml_opt_type = @import("std").mem.zeroes(enum_ggml_opt_type),
    graph_size: usize = @import("std").mem.zeroes(usize),
    n_threads: c_int = @import("std").mem.zeroes(c_int),
    past: c_int = @import("std").mem.zeroes(c_int),
    delta: f32 = @import("std").mem.zeroes(f32),
    max_no_improvement: c_int = @import("std").mem.zeroes(c_int),
    print_forward_graph: bool = @import("std").mem.zeroes(bool),
    print_backward_graph: bool = @import("std").mem.zeroes(bool),
    n_gradient_accumulation: c_int = @import("std").mem.zeroes(c_int),
    adam: struct_unnamed_1 = @import("std").mem.zeroes(struct_unnamed_1),
    lbfgs: struct_unnamed_2 = @import("std").mem.zeroes(struct_unnamed_2),
};
const struct_unnamed_3 = extern struct {
    g: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    m: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    v: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    pf: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    fx_best: f32 = @import("std").mem.zeroes(f32),
    fx_prev: f32 = @import("std").mem.zeroes(f32),
    n_no_improvement: c_int = @import("std").mem.zeroes(c_int),
};
const struct_unnamed_4 = extern struct {
    x: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    xp: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    g: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    gp: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    d: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    pf: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    lmal: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    lmys: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    lms: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    lmy: [*c]struct_ggml_tensor = @import("std").mem.zeroes([*c]struct_ggml_tensor),
    fx_best: f32 = @import("std").mem.zeroes(f32),
    step: f32 = @import("std").mem.zeroes(f32),
    j: c_int = @import("std").mem.zeroes(c_int),
    k: c_int = @import("std").mem.zeroes(c_int),
    end: c_int = @import("std").mem.zeroes(c_int),
    n_no_improvement: c_int = @import("std").mem.zeroes(c_int),
};
pub const struct_ggml_opt_context = extern struct {
    ctx: ?*context = @import("std").mem.zeroes(?*context),
    params: struct_ggml_opt_params = @import("std").mem.zeroes(struct_ggml_opt_params),
    iter: c_int = @import("std").mem.zeroes(c_int),
    nx: i64 = @import("std").mem.zeroes(i64),
    just_initialized: bool = @import("std").mem.zeroes(bool),
    loss_before: f32 = @import("std").mem.zeroes(f32),
    loss_after: f32 = @import("std").mem.zeroes(f32),
    adam: struct_unnamed_3 = @import("std").mem.zeroes(struct_unnamed_3),
    lbfgs: struct_unnamed_4 = @import("std").mem.zeroes(struct_unnamed_4),
};
pub extern fn opt_default_params(@"type": enum_ggml_opt_type) struct_ggml_opt_params;
pub extern fn opt(ctx: ?*context, params: struct_ggml_opt_params, f: [*c]struct_ggml_tensor) enum_ggml_opt_result;
pub extern fn opt_init(ctx: ?*context, opt: [*c]struct_ggml_opt_context, params: struct_ggml_opt_params, nx: i64) void;
pub extern fn opt_resume(ctx: ?*context, opt: [*c]struct_ggml_opt_context, f: [*c]struct_ggml_tensor) enum_ggml_opt_result;
pub extern fn opt_resume_g(ctx: ?*context, opt: [*c]struct_ggml_opt_context, f: [*c]struct_ggml_tensor, gf: [*c]struct_ggml_cgraph, gb: [*c]struct_ggml_cgraph, callback: opt_callback, callback_data: ?*anyopaque) enum_ggml_opt_result;
pub extern fn quantize_init(@"type": type) void;
pub extern fn quantize_free() void;
pub extern fn quantize_q4_0(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q4_1(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q5_0(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q5_1(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q8_0(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q2_K(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q3_K(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q4_K(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q5_K(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_q6_K(src: [*c]const f32, dst: ?*anyopaque, n: c_int, k: c_int, hist: [*c]i64) usize;
pub extern fn quantize_requires_imatrix(@"type": type) bool;
pub extern fn quantize_chunk(@"type": type, src: [*c]const f32, dst: ?*anyopaque, start: c_int, nrows: c_int, n_per_row: c_int, hist: [*c]i64, imatrix: [*c]const f32) usize;
pub const GGUF_TYPE_UINT8: c_int = 0;
pub const GGUF_TYPE_INT8: c_int = 1;
pub const GGUF_TYPE_UINT16: c_int = 2;
pub const GGUF_TYPE_INT16: c_int = 3;
pub const GGUF_TYPE_UINT32: c_int = 4;
pub const GGUF_TYPE_INT32: c_int = 5;
pub const GGUF_TYPE_FLOAT32: c_int = 6;
pub const GGUF_TYPE_BOOL: c_int = 7;
pub const GGUF_TYPE_STRING: c_int = 8;
pub const GGUF_TYPE_ARRAY: c_int = 9;
pub const GGUF_TYPE_UINT64: c_int = 10;
pub const GGUF_TYPE_INT64: c_int = 11;
pub const GGUF_TYPE_FLOAT64: c_int = 12;
pub const GGUF_TYPE_COUNT: c_int = 13;
pub const enum_gguf_type = c_uint;
pub const struct_gguf_context = opaque {};
pub const struct_gguf_init_params = extern struct {
    no_alloc: bool = @import("std").mem.zeroes(bool),
    ctx: [*c]?*context = @import("std").mem.zeroes([*c]?*context),
};
pub extern fn gguf_init_empty() ?*struct_gguf_context;
pub extern fn gguf_init_from_file(fname: [*c]const u8, params: struct_gguf_init_params) ?*struct_gguf_context;
pub extern fn gguf_free(ctx: ?*struct_gguf_context) void;
pub extern fn gguf_type_name(@"type": enum_gguf_type) [*c]const u8;
pub extern fn gguf_get_version(ctx: ?*const struct_gguf_context) c_int;
pub extern fn gguf_get_alignment(ctx: ?*const struct_gguf_context) usize;
pub extern fn gguf_get_data_offset(ctx: ?*const struct_gguf_context) usize;
pub extern fn gguf_get_data(ctx: ?*const struct_gguf_context) ?*anyopaque;
pub extern fn gguf_get_n_kv(ctx: ?*const struct_gguf_context) c_int;
pub extern fn gguf_find_key(ctx: ?*const struct_gguf_context, key: [*c]const u8) c_int;
pub extern fn gguf_get_key(ctx: ?*const struct_gguf_context, key_id: c_int) [*c]const u8;
pub extern fn gguf_get_kv_type(ctx: ?*const struct_gguf_context, key_id: c_int) enum_gguf_type;
pub extern fn gguf_get_arr_type(ctx: ?*const struct_gguf_context, key_id: c_int) enum_gguf_type;
pub extern fn gguf_get_val_u8(ctx: ?*const struct_gguf_context, key_id: c_int) u8;
pub extern fn gguf_get_val_i8(ctx: ?*const struct_gguf_context, key_id: c_int) i8;
pub extern fn gguf_get_val_u16(ctx: ?*const struct_gguf_context, key_id: c_int) u16;
pub extern fn gguf_get_val_i16(ctx: ?*const struct_gguf_context, key_id: c_int) i16;
pub extern fn gguf_get_val_u32(ctx: ?*const struct_gguf_context, key_id: c_int) u32;
pub extern fn gguf_get_val_i32(ctx: ?*const struct_gguf_context, key_id: c_int) i32;
pub extern fn gguf_get_val_f32(ctx: ?*const struct_gguf_context, key_id: c_int) f32;
pub extern fn gguf_get_val_u64(ctx: ?*const struct_gguf_context, key_id: c_int) u64;
pub extern fn gguf_get_val_i64(ctx: ?*const struct_gguf_context, key_id: c_int) i64;
pub extern fn gguf_get_val_f64(ctx: ?*const struct_gguf_context, key_id: c_int) f64;
pub extern fn gguf_get_val_bool(ctx: ?*const struct_gguf_context, key_id: c_int) bool;
pub extern fn gguf_get_val_str(ctx: ?*const struct_gguf_context, key_id: c_int) [*c]const u8;
pub extern fn gguf_get_val_data(ctx: ?*const struct_gguf_context, key_id: c_int) ?*const anyopaque;
pub extern fn gguf_get_arr_n(ctx: ?*const struct_gguf_context, key_id: c_int) c_int;
pub extern fn gguf_get_arr_data(ctx: ?*const struct_gguf_context, key_id: c_int) ?*const anyopaque;
pub extern fn gguf_get_arr_str(ctx: ?*const struct_gguf_context, key_id: c_int, i: c_int) [*c]const u8;
pub extern fn gguf_get_n_tensors(ctx: ?*const struct_gguf_context) c_int;
pub extern fn gguf_find_tensor(ctx: ?*const struct_gguf_context, name: [*c]const u8) c_int;
pub extern fn gguf_get_tensor_offset(ctx: ?*const struct_gguf_context, i: c_int) usize;
pub extern fn gguf_get_tensor_name(ctx: ?*const struct_gguf_context, i: c_int) [*c]u8;
pub extern fn gguf_get_tensor_type(ctx: ?*const struct_gguf_context, i: c_int) type;
pub extern fn gguf_set_val_u8(ctx: ?*struct_gguf_context, key: [*c]const u8, val: u8) void;
pub extern fn gguf_set_val_i8(ctx: ?*struct_gguf_context, key: [*c]const u8, val: i8) void;
pub extern fn gguf_set_val_u16(ctx: ?*struct_gguf_context, key: [*c]const u8, val: u16) void;
pub extern fn gguf_set_val_i16(ctx: ?*struct_gguf_context, key: [*c]const u8, val: i16) void;
pub extern fn gguf_set_val_u32(ctx: ?*struct_gguf_context, key: [*c]const u8, val: u32) void;
pub extern fn gguf_set_val_i32(ctx: ?*struct_gguf_context, key: [*c]const u8, val: i32) void;
pub extern fn gguf_set_val_f32(ctx: ?*struct_gguf_context, key: [*c]const u8, val: f32) void;
pub extern fn gguf_set_val_u64(ctx: ?*struct_gguf_context, key: [*c]const u8, val: u64) void;
pub extern fn gguf_set_val_i64(ctx: ?*struct_gguf_context, key: [*c]const u8, val: i64) void;
pub extern fn gguf_set_val_f64(ctx: ?*struct_gguf_context, key: [*c]const u8, val: f64) void;
pub extern fn gguf_set_val_bool(ctx: ?*struct_gguf_context, key: [*c]const u8, val: bool) void;
pub extern fn gguf_set_val_str(ctx: ?*struct_gguf_context, key: [*c]const u8, val: [*c]const u8) void;
pub extern fn gguf_set_arr_data(ctx: ?*struct_gguf_context, key: [*c]const u8, @"type": enum_gguf_type, data: ?*const anyopaque, n: c_int) void;
pub extern fn gguf_set_arr_str(ctx: ?*struct_gguf_context, key: [*c]const u8, data: [*c][*c]const u8, n: c_int) void;
pub extern fn gguf_set_kv(ctx: ?*struct_gguf_context, src: ?*struct_gguf_context) void;
pub extern fn gguf_add_tensor(ctx: ?*struct_gguf_context, tensor: [*c]const struct_ggml_tensor) void;
pub extern fn gguf_set_tensor_type(ctx: ?*struct_gguf_context, name: [*c]const u8, @"type": type) void;
pub extern fn gguf_set_tensor_data(ctx: ?*struct_gguf_context, name: [*c]const u8, data: ?*const anyopaque, size: usize) void;
pub extern fn gguf_write_to_file(ctx: ?*const struct_gguf_context, fname: [*c]const u8, only_meta: bool) void;
pub extern fn gguf_get_meta_size(ctx: ?*const struct_gguf_context) usize;
pub extern fn gguf_get_meta_data(ctx: ?*const struct_gguf_context, data: ?*anyopaque) void;
pub extern fn cpu_has_avx() c_int;
pub extern fn cpu_has_avx_vnni() c_int;
pub extern fn cpu_has_avx2() c_int;
pub extern fn cpu_has_avx512() c_int;
pub extern fn cpu_has_avx512_vbmi() c_int;
pub extern fn cpu_has_avx512_vnni() c_int;
pub extern fn cpu_has_fma() c_int;
pub extern fn cpu_has_neon() c_int;
pub extern fn cpu_has_arm_fma() c_int;
pub extern fn cpu_has_metal() c_int;
pub extern fn cpu_has_f16c() c_int;
pub extern fn cpu_has_fp16_va() c_int;
pub extern fn cpu_has_wasm_simd() c_int;
pub extern fn cpu_has_blas() c_int;
pub extern fn cpu_has_cublas() c_int;
pub extern fn cpu_has_clblast() c_int;
pub extern fn cpu_has_vulkan() c_int;
pub extern fn cpu_has_kompute() c_int;
pub extern fn cpu_has_gpublas() c_int;
pub extern fn cpu_has_sse3() c_int;
pub extern fn cpu_has_ssse3() c_int;
pub extern fn cpu_has_sycl() c_int;
pub extern fn cpu_has_vsx() c_int;
pub const to_float_t = ?*const fn (noalias ?*const anyopaque, noalias [*c]f32, c_int) callconv(.C) void;
pub const from_float_t = ?*const fn (noalias [*c]const f32, noalias ?*anyopaque, c_int) callconv(.C) void;
pub const vec_dot_t = ?*const fn (c_int, noalias [*c]f32, noalias ?*const anyopaque, noalias ?*const anyopaque) callconv(.C) void;
pub const type_traits_t = extern struct {
    type_name: [*c]const u8 = @import("std").mem.zeroes([*c]const u8),
    blck_size: c_int = @import("std").mem.zeroes(c_int),
    type_size: usize = @import("std").mem.zeroes(usize),
    is_quantized: bool = @import("std").mem.zeroes(bool),
    to_float: to_float_t = @import("std").mem.zeroes(to_float_t),
    from_float: from_float_t = @import("std").mem.zeroes(from_float_t),
    from_float_reference: from_float_t = @import("std").mem.zeroes(from_float_t),
    vec_dot: vec_dot_t = @import("std").mem.zeroes(vec_dot_t),
    vec_dot_type: type = @import("std").mem.zeroes(type),
};
pub extern fn internal_get_type_traits(@"type": type) type_traits_t;
pub const struct_ggml_backend = opaque {};
pub const struct_ggml_backend_buffer_type = opaque {};
pub const struct_ggml_allocr = opaque {};
pub const allocr_t = ?*struct_ggml_allocr;
pub extern fn allocr_new(data: ?*anyopaque, size: usize, alignment: usize) allocr_t;
pub extern fn allocr_new_measure(alignment: usize) allocr_t;
pub extern fn allocr_new_from_buffer(buffer: ?*struct_ggml_backend_buffer) allocr_t;
pub extern fn allocr_new_from_backend(backend: ?*struct_ggml_backend, size: usize) allocr_t;
pub extern fn allocr_new_measure_from_backend(backend: ?*struct_ggml_backend) allocr_t;
pub extern fn allocr_get_buffer(alloc: allocr_t) ?*struct_ggml_backend_buffer;
pub extern fn allocr_set_parse_seq(alloc: allocr_t, list: [*c]const c_int, n: c_int) void;
pub extern fn allocr_free(alloc: allocr_t) void;
pub extern fn allocr_is_measure(alloc: allocr_t) bool;
pub extern fn allocr_reset(alloc: allocr_t) void;
pub extern fn allocr_alloc(alloc: allocr_t, tensor: [*c]struct_ggml_tensor) void;
pub extern fn allocr_max_size(alloc: allocr_t) usize;
pub extern fn allocr_alloc_graph(alloc: allocr_t, graph: [*c]struct_ggml_cgraph) usize;
pub const struct_ggml_tallocr = opaque {};
pub const tallocr_t = ?*struct_ggml_tallocr;
pub extern fn tallocr_new(data: ?*anyopaque, size: usize, alignment: usize) tallocr_t;
pub extern fn tallocr_new_measure(alignment: usize) tallocr_t;
pub extern fn tallocr_new_from_buft(buft: ?*struct_ggml_backend_buffer_type, size: usize) tallocr_t;
pub extern fn tallocr_new_from_backend(backend: ?*struct_ggml_backend, size: usize) tallocr_t;
pub extern fn tallocr_new_from_buffer(buffer: ?*struct_ggml_backend_buffer) tallocr_t;
pub extern fn tallocr_new_measure_from_buft(buft: ?*struct_ggml_backend_buffer_type) tallocr_t;
pub extern fn tallocr_new_measure_from_backend(backend: ?*struct_ggml_backend) tallocr_t;
pub extern fn tallocr_get_buffer(talloc: tallocr_t) ?*struct_ggml_backend_buffer;
pub extern fn tallocr_free(talloc: tallocr_t) void;
pub extern fn tallocr_is_measure(talloc: tallocr_t) bool;
pub extern fn tallocr_reset(talloc: tallocr_t) void;
pub extern fn tallocr_alloc(talloc: tallocr_t, tensor: [*c]struct_ggml_tensor) void;
pub extern fn tallocr_max_size(talloc: tallocr_t) usize;
pub const struct_ggml_gallocr = opaque {};
pub const gallocr_t = ?*struct_ggml_gallocr;
pub extern fn gallocr_new() gallocr_t;
pub extern fn gallocr_free(galloc: gallocr_t) void;
pub extern fn gallocr_set_parse_seq(galloc: gallocr_t, list: [*c]const c_int, n: c_int) void;
pub extern fn gallocr_alloc_graph(galloc: gallocr_t, talloc: tallocr_t, graph: [*c]struct_ggml_cgraph) usize;
pub extern fn gallocr_alloc_graph_n(galloc: gallocr_t, graph: [*c]struct_ggml_cgraph, hash_set: struct_ggml_hash_set, hash_node_talloc: [*c]tallocr_t) void;
pub extern fn backend_alloc_ctx_tensors_from_buft(ctx: ?*context, buft: ?*struct_ggml_backend_buffer_type) ?*struct_ggml_backend_buffer;
pub extern fn backend_alloc_ctx_tensors(ctx: ?*context, backend: ?*struct_ggml_backend) ?*struct_ggml_backend_buffer;
pub const backend_buffer_type_t = ?*struct_ggml_backend_buffer_type;
pub const backend_buffer_t = ?*struct_ggml_backend_buffer;
pub const backend_t = ?*struct_ggml_backend;
pub const backend_graph_plan_t = ?*anyopaque;
pub extern fn backend_buft_name(buft: backend_buffer_type_t) [*c]const u8;
pub extern fn backend_buft_alloc_buffer(buft: backend_buffer_type_t, size: usize) backend_buffer_t;
pub extern fn backend_buft_get_alignment(buft: backend_buffer_type_t) usize;
pub extern fn backend_buft_get_max_size(buft: backend_buffer_type_t) usize;
pub extern fn backend_buft_get_alloc_size(buft: backend_buffer_type_t, tensor: [*c]struct_ggml_tensor) usize;
pub extern fn backend_buft_supports_backend(buft: backend_buffer_type_t, backend: backend_t) bool;
pub extern fn backend_buft_is_host(buft: backend_buffer_type_t) bool;
pub const BACKEND_BUFFER_USAGE_ANY: c_int = 0;
pub const BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub const enum_ggml_backend_buffer_usage = c_uint;
pub extern fn backend_buffer_name(buffer: backend_buffer_t) [*c]const u8;
pub extern fn backend_buffer_free(buffer: backend_buffer_t) void;
pub extern fn backend_buffer_get_base(buffer: backend_buffer_t) ?*anyopaque;
pub extern fn backend_buffer_get_size(buffer: backend_buffer_t) usize;
pub extern fn backend_buffer_init_tensor(buffer: backend_buffer_t, tensor: [*c]struct_ggml_tensor) void;
pub extern fn backend_buffer_get_alignment(buffer: backend_buffer_t) usize;
pub extern fn backend_buffer_get_max_size(buffer: backend_buffer_t) usize;
pub extern fn backend_buffer_get_alloc_size(buffer: backend_buffer_t, tensor: [*c]struct_ggml_tensor) usize;
pub extern fn backend_buffer_clear(buffer: backend_buffer_t, value: u8) void;
pub extern fn backend_buffer_is_host(buffer: backend_buffer_t) bool;
pub extern fn backend_buffer_set_usage(buffer: backend_buffer_t, usage: enum_ggml_backend_buffer_usage) void;
pub extern fn backend_buffer_get_type(buffer: backend_buffer_t) backend_buffer_type_t;
pub extern fn backend_buffer_reset(buffer: backend_buffer_t) void;
pub extern fn backend_name(backend: backend_t) [*c]const u8;
pub extern fn backend_free(backend: backend_t) void;
pub extern fn backend_get_default_buffer_type(backend: backend_t) backend_buffer_type_t;
pub extern fn backend_alloc_buffer(backend: backend_t, size: usize) backend_buffer_t;
pub extern fn backend_get_alignment(backend: backend_t) usize;
pub extern fn backend_get_max_size(backend: backend_t) usize;
pub extern fn backend_tensor_set_async(backend: backend_t, tensor: [*c]struct_ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn backend_tensor_get_async(backend: backend_t, tensor: [*c]const struct_ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn backend_tensor_set(tensor: [*c]struct_ggml_tensor, data: ?*const anyopaque, offset: usize, size: usize) void;
pub extern fn backend_tensor_get(tensor: [*c]const struct_ggml_tensor, data: ?*anyopaque, offset: usize, size: usize) void;
pub extern fn backend_synchronize(backend: backend_t) void;
pub extern fn backend_graph_plan_create(backend: backend_t, cgraph: [*c]struct_ggml_cgraph) backend_graph_plan_t;
pub extern fn backend_graph_plan_free(backend: backend_t, plan: backend_graph_plan_t) void;
pub extern fn backend_graph_plan_compute(backend: backend_t, plan: backend_graph_plan_t) void;
pub extern fn backend_graph_compute(backend: backend_t, cgraph: [*c]struct_ggml_cgraph) bool;
pub extern fn backend_supports_op(backend: backend_t, op: [*c]const struct_ggml_tensor) bool;
pub extern fn backend_tensor_copy(src: [*c]struct_ggml_tensor, dst: [*c]struct_ggml_tensor) void;
pub extern fn backend_tensor_copy_async(backend: backend_t, src: [*c]struct_ggml_tensor, dst: [*c]struct_ggml_tensor) void;
pub extern fn backend_cpu_init() backend_t;
pub extern fn backend_is_cpu(backend: backend_t) bool;
pub extern fn backend_cpu_set_n_threads(backend_cpu: backend_t, n_threads: c_int) void;
pub extern fn backend_cpu_buffer_from_ptr(ptr: ?*anyopaque, size: usize) backend_buffer_t;
pub extern fn backend_cpu_buffer_type() backend_buffer_type_t;
pub extern fn backend_reg_get_count() usize;
pub extern fn backend_reg_find_by_name(name: [*c]const u8) usize;
pub extern fn backend_reg_init_backend_from_str(backend_str: [*c]const u8) backend_t;
pub extern fn backend_reg_get_name(i: usize) [*c]const u8;
pub extern fn backend_reg_init_backend(i: usize, params: [*c]const u8) backend_t;
pub extern fn backend_reg_get_default_buffer_type(i: usize) backend_buffer_type_t;
pub extern fn backend_reg_alloc_buffer(i: usize, size: usize) backend_buffer_t;
pub const struct_ggml_backend_sched = opaque {};
pub const backend_sched_t = ?*struct_ggml_backend_sched;
pub const backend_sched_eval_callback = ?*const fn ([*c]struct_ggml_tensor, bool, ?*anyopaque) callconv(.C) bool;
pub extern fn backend_sched_new(backends: [*c]backend_t, bufts: [*c]backend_buffer_type_t, n_backends: c_int, graph_size: usize) backend_sched_t;
pub extern fn backend_sched_free(sched: backend_sched_t) void;
pub extern fn backend_sched_init_measure(sched: backend_sched_t, measure_graph: [*c]struct_ggml_cgraph) void;
pub extern fn backend_sched_get_n_splits(sched: backend_sched_t) c_int;
pub extern fn backend_sched_get_tallocr(sched: backend_sched_t, backend: backend_t) tallocr_t;
pub extern fn backend_sched_get_buffer(sched: backend_sched_t, backend: backend_t) backend_buffer_t;
pub extern fn backend_sched_set_node_backend(sched: backend_sched_t, node: [*c]struct_ggml_tensor, backend: backend_t) void;
pub extern fn backend_sched_get_node_backend(sched: backend_sched_t, node: [*c]struct_ggml_tensor) backend_t;
pub extern fn backend_sched_graph_compute(sched: backend_sched_t, graph: [*c]struct_ggml_cgraph) void;
pub extern fn backend_sched_reset(sched: backend_sched_t) void;
pub extern fn backend_sched_set_eval_callback(sched: backend_sched_t, callback: backend_sched_eval_callback, user_data: ?*anyopaque) void;
pub const struct_ggml_backend_graph_copy = extern struct {
    buffer: backend_buffer_t = @import("std").mem.zeroes(backend_buffer_t),
    ctx_allocated: ?*context = @import("std").mem.zeroes(?*context),
    ctx_unallocated: ?*context = @import("std").mem.zeroes(?*context),
    graph: [*c]struct_ggml_cgraph = @import("std").mem.zeroes([*c]struct_ggml_cgraph),
};
pub extern fn backend_graph_copy(backend: backend_t, graph: [*c]struct_ggml_cgraph) struct_ggml_backend_graph_copy;
pub extern fn backend_graph_copy_free(copy: struct_ggml_backend_graph_copy) void;
pub const backend_eval_callback = ?*const fn (c_int, [*c]struct_ggml_tensor, [*c]struct_ggml_tensor, ?*anyopaque) callconv(.C) bool;
pub extern fn backend_compare_graph_backend(backend1: backend_t, backend2: backend_t, graph: [*c]struct_ggml_cgraph, callback: backend_eval_callback, user_data: ?*anyopaque) bool;
pub extern fn backend_tensor_alloc(buffer: backend_buffer_t, tensor: [*c]struct_ggml_tensor, addr: ?*anyopaque) void;
pub extern fn backend_view_init(buffer: backend_buffer_t, tensor: [*c]struct_ggml_tensor) void;

pub const GGML_FILE_MAGIC = @import("std").zig_c_translation_promoteIntLiteral(c_int, 0x67676d6c, .hex);
pub const GGML_FILE_VERSION = @as(c_int, 1);
pub const GGML_QNT_VERSION = @as(c_int, 2);
pub const GGML_QNT_VERSION_FACTOR = @as(c_int, 1000);
pub const GGML_MAX_DIMS = @as(c_int, 4);
pub const GGML_MAX_PARAMS = @as(c_int, 2048);
pub const GGML_MAX_CONTEXTS = @as(c_int, 64);
pub const GGML_MAX_SRC = @as(c_int, 10);
pub const GGML_MAX_NAME = @as(c_int, 64);
pub const GGML_MAX_OP_PARAMS = @as(c_int, 64);
pub const GGML_DEFAULT_N_THREADS = @as(c_int, 4);
pub const GGML_DEFAULT_GRAPH_SIZE = @as(c_int, 2048);
pub const GGML_MEM_ALIGN = @as(c_int, 16);
pub const GGML_EXIT_SUCCESS = @as(c_int, 0);
pub const GGML_EXIT_ABORTED = @as(c_int, 1);
pub const GGUF_MAGIC = "GGUF";
pub const GGUF_VERSION = @as(c_int, 3);
pub const GGUF_DEFAULT_ALIGNMENT = @as(c_int, 32);
pub const GGML_UNUSED = @import("std").zig_c_translation_Macros_DISCARD;
pub inline fn GGML_PAD(x: anytype, n: anytype) @TypeOf(((x + n) - @as(c_int, 1)) & ~(n - @as(c_int, 1))) {
    _ = &x;
    _ = &n;
    return ((x + n) - @as(c_int, 1)) & ~(n - @as(c_int, 1));
}
pub inline fn GGML_UNREACHABLE() noreturn {
    unreachable;
}
pub const GGML_N_TASKS_MAX = -@as(c_int, 1);
