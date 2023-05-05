const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const want_lto = b.option(bool, "lto", "Want -fLTO");

    const lib = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    lib.want_lto = want_lto;
    lib.linkLibCpp();
    lib.installHeader("llama.h", "llama.h");
    lib.installHeader("ggml.h", "ggml.h");
    lib.addCSourceFiles(&.{
        "ggml.c",
    }, &.{});
    lib.addCSourceFiles(&.{
        "llama.cpp",
    }, &.{"-std=c++11"});
    b.installArtifact(lib);

    const config_header = b.addConfigHeader(.{}, .{
        .BUILD_NUMBER = std.mem.trim(u8, b.exec(&.{ "git", "rev-list", "--count", "HEAD" }), "\n\r "),
        .BUILD_COMMIT = std.mem.trim(u8, b.exec(&.{ "git", "rev-parse", "--short", "HEAD" }), "\n\r "),
    });

    const build_args = .{ .b = b, .lib = lib, .target = target, .optimize = optimize, .want_lto = want_lto, .config_header = config_header };

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
