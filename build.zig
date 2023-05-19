const std = @import("std");

const ComputeFramework = enum {
    accelerate,
    openblas,
    cublas,
    clblast,
};

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const want_lto = b.option(bool, "lto", "Want -fLTO");
    const compute_framework = b.option(ComputeFramework, "compute-framework", "Specify a library to perform computations (default: none)");

    const libggml = b.addStaticLibrary(.{
        .name = "ggml",
        .target = target,
        .optimize = optimize,
    });
    libggml.want_lto = want_lto;
    libggml.linkLibC();
    libggml.installHeader("ggml.h", "ggml.h");
    libggml.addCSourceFiles(&.{
        "ggml.c",
    }, &.{});
    if (compute_framework) |framework| {
        switch (framework) {
            .accelerate => {
                libggml.defineCMacro("GGML_USE_ACCELERATE", "1");
            },
            .openblas => {
                libggml.defineCMacro("GGML_USE_OPENBLAS", "1");
                libggml.linkSystemLibrary("openblas");
            },
            .cublas => {
                libggml.defineCMacro("GGML_USE_CUBLAS", "1");
            },
            .clblast => {
                const clblast = b.dependency("clblast", .{
                    .target = target,
                    .optimize = optimize,
                });

                libggml.addCSourceFile("ggml-opencl.c", &.{});
                libggml.defineCMacro("GGML_USE_CLBLAST", "1");
                libggml.linkLibrary(clblast.artifact("clblast"));
            },
        }
    }

    const libllama = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    libllama.want_lto = want_lto;
    libllama.linkLibCpp();
    libllama.installHeader("llama.h", "llama.h");
    libllama.linkLibrary(libggml);
    libllama.addCSourceFiles(&.{
        "llama.cpp",
    }, &.{"-std=c++11"});
    b.installArtifact(libllama);

    const config_header = b.addConfigHeader(.{}, .{
        .BUILD_NUMBER = std.mem.trim(u8, b.exec(&.{ "git", "rev-list", "--count", "HEAD" }), "\n\r "),
        .BUILD_COMMIT = std.mem.trim(u8, b.exec(&.{ "git", "rev-parse", "--short", "HEAD" }), "\n\r "),
    });

    const build_args = .{ .b = b, .lib = libllama, .target = target, .optimize = optimize, .want_lto = want_lto, .config_header = config_header };

    const exe = build_example("main", build_args);
    _ = build_example("quantize", build_args);
    _ = build_example("perplexity", build_args);
    _ = build_example("embedding", build_args);

    // create "zig build run" command for ./main

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn build_example(name: []const u8, args: anytype) *std.build.LibExeObjStep {
    const b = args.b;
    const lib = args.lib;
    const want_lto = args.want_lto;

    const exe = b.addExecutable(.{
        .name = name,
        .target = args.target,
        .optimize = args.optimize,
    });
    exe.want_lto = want_lto;
    exe.addIncludePath("examples");
    exe.addConfigHeader(args.config_header);
    exe.addCSourceFiles(&.{
        b.fmt("examples/{s}/{s}.cpp", .{ name, name }),
        "examples/common.cpp",
    }, &.{"-std=c++11"});
    exe.linkLibrary(lib);
    b.installArtifact(exe);

    return exe;
}
