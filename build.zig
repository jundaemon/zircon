const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const functions = b.addModule("functions", .{
        .root_source_file = b.path("src/functions.zig"),
        .target = target,
        .optimize = optimize,
    });
    const layer = b.createModule(.{
        .root_source_file = b.path("src/layer.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "functions", .module = functions },
        },
    });
    const conf = b.addModule("conf", .{
        .root_source_file = b.path("src/conf.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "functions", .module = functions },
            .{ .name = "layer", .module = layer },
        },
    });
    const optim = b.addModule("optim", .{
        .root_source_file = b.path("src/optim.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "layer", .module = layer },
            .{ .name = "conf", .module = conf },
        },
    });
    const mlp = b.addModule("mlp", .{
        .root_source_file = b.path("src/mlp.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "functions", .module = functions },
            .{ .name = "layer", .module = layer },
            .{ .name = "conf", .module = conf },
            .{ .name = "optim", .module = optim },
        },
    });

    const mlp_tests = b.addTest(.{ .root_module = mlp });
    const run_mlp_tests = b.addRunArtifact(mlp_tests);
    const test_step = b.step("test", "run tests");
    test_step.dependOn(&run_mlp_tests.step);
}
