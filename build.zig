const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_clblast = b.option(bool, "clblast", "Enable clblast for GPU accelerated layers (default: false)") orelse false;

    const clblast = b.dependency("clblast", .{
        .target = target,
        .optimize = optimize,
    });

    const zig_version = @import("builtin").zig_version_string;

    // If git fails to find a version, use a dummy hash value instead.
    // We won't be in a git repository when the file is downloaded as an archive
    // by zig. Also, zig 0.12.0-dev.2059+42389cb9c doesn't have any way to get
    // the url it was fetched from or the package's hash, as far as I know.
    var git_hash_exit_code: u8 = undefined;
    const commit_hash = b.runAllowFail(&.{ "git", "rev-parse", "HEAD" }, &git_hash_exit_code, .Ignore) catch "0000000000000000000000000000000000000000";

    const generated_files = b.addWriteFiles();
    const llama_build_info_cpp = generated_files.add("common/build-info.cpp", b.fmt(
        \\int LLAMA_BUILD_NUMBER = {};
        \\char const *LLAMA_COMMIT = "{s}";
        \\char const *LLAMA_COMPILER = "Zig {s}";
        \\char const *LLAMA_BUILD_TARGET = "{s}";
        \\
    , .{ 0, std.mem.trim(u8, commit_hash, "\n\r"), zig_version, try target.result.zigTriple(b.allocator) }));

    const ggml = b.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });
    if (ggml.rootModuleTarget().abi != .msvc)
        ggml.defineCMacro("_GNU_SOURCE", null);
    ggml.root_module.c_std = .C11;
    ggml.addCSourceFiles(.{ .files = &.{
        "ggml.c",
        "ggml-quants.c",
        "ggml-alloc.c",
        "ggml-backend.c",
    } });
    ggml.addIncludePath(.{ .path = "./" });
    ggml.installHeader("ggml.h", "ggml.h");
    ggml.installHeader("ggml-alloc.h", "ggml-alloc.h");
    ggml.installHeader("ggml-backend.h", "ggml-backend.h");

    if (enable_clblast) {
        ggml.addCSourceFile(.{ .file = .{ .path = "ggml-opencl.cpp" } });
        ggml.installHeader("ggml-opencl.h", "ggml-opencl.h");
        ggml.root_module.addCMacro("GGML_USE_CLBLAST", "1");
        ggml.root_module.linkLibrary(clblast.artifact("clblast"));
    }

    ggml.linkLibC();
    b.installArtifact(ggml);

    const llama = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    llama.addCSourceFile(.{ .file = .{ .path = "llama.cpp" } });
    llama.addCSourceFile(.{ .file = llama_build_info_cpp });
    llama.linkLibrary(ggml);
    llama.installHeader("llama.h", "llama.h");
    llama.linkLibCpp();
    b.installArtifact(llama);

    // raw header file bindings
    const ggml_h_zig = b.addModule("ggml.h", .{
        .root_source_file = .{ .path = "bindings/zig/src/ggml.h.zig" },
        .target = target,
        .optimize = optimize,
    });
    // raw header file bindings
    const llama_h_zig = b.addModule("llama.h", .{
        .root_source_file = .{ .path = "bindings/zig/src/llama.h.zig" },
        .target = target,
        .optimize = optimize,
    });
    llama_h_zig.addImport("ggml.h", ggml_h_zig);

    const zig_bindings = b.addModule("llama", .{
        .root_source_file = .{ .path = "bindings/zig/src/llama.zig" },
        .target = target,
        .optimize = optimize,
    });
    zig_bindings.linkLibrary(llama);
    zig_bindings.addImport("llama.h", llama_h_zig);

    // zig examples
    {
        const zig_simple = b.addExecutable(.{
            .name = "zig-simple",
            .root_source_file = .{ .path = "bindings/zig/examples/simple.zig" },
            .target = target,
            .optimize = optimize,
        });
        zig_simple.root_module.addImport("llama", zig_bindings);
        b.installArtifact(zig_simple);
    }

    const common = b.addStaticLibrary(.{
        .name = "common",
        .target = target,
        .optimize = optimize,
    });
    common.addCSourceFiles(.{ .files = &.{
        "common/common.cpp",
        "common/console.cpp",
        "common/sampling.cpp",
        "common/grammar-parser.cpp",
        "common/train.cpp",
    } });
    common.linkLibrary(llama);
    common.addIncludePath(.{ .path = "common" });
    common.installHeader("common/common.h", "common.h");
    common.installHeader("common/console.h", "console.h");
    common.installHeader("common/sampling.h", "sampling.h");
    common.installHeader("common/grammar-parser.h", "grammar-parser.h");
    common.installHeader("common/log.h", "log.h");
    common.installHeader("common/train.h", "train.h");
    common.installHeader("common/stb_image.h", "stb_image.h");

    addExample(b, .{
        .name = "main",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    addExample(b, .{
        .name = "quantize",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    addExample(b, .{
        .name = "perplexity",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    addExample(b, .{
        .name = "embedding",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    addExample(b, .{
        .name = "finetune",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    addExample(b, .{
        .name = "train-text-from-scratch",
        .target = target,
        .optimize = optimize,
        .libraries = &.{ llama, common },
    });

    const server = b.addExecutable(.{
        .name = "server",
        .target = target,
        .optimize = optimize,
    });
    server.addCSourceFiles(.{ .files = &.{
        "examples/server/server.cpp",
        "examples/llava/clip.cpp",
    } });
    server.linkLibrary(llama);
    server.linkLibrary(common);
    if (server.rootModuleTarget().os.tag == .windows) {
        server.linkSystemLibrary("ws2_32");
    }
    b.installArtifact(server);
}

fn addExample(b: *std.Build, options: struct {
    name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Mode,
    libraries: []const *std.Build.Step.Compile,
}) void {
    const example = b.addExecutable(.{
        .name = options.name,
        .target = options.target,
        .optimize = options.optimize,
    });
    example.addCSourceFile(.{ .file = .{ .path = b.fmt("examples/{s}/{s}.cpp", .{ options.name, options.name }) } });
    for (options.libraries) |lib| {
        example.linkLibrary(lib);
    }
    b.installArtifact(example);
}
