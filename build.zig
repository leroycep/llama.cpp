const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zig_version = @import("builtin").zig_version_string;
    const commit_hash = b.run(&.{ "git", "rev-parse", "HEAD" });

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
