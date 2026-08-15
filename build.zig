const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const function = b.addModule("function", .{
        .root_source_file = b.path("src/function.zig"),
        .target = target,
        .optimize = optimize,
    });
    const layer = b.createModule(.{
        .root_source_file = b.path("src/layer.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "function", .module = function },
        },
    });
    const mlp = b.addModule("mlp", .{
        .root_source_file = b.path("src/mlp.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "function", .module = function },
            .{ .name = "layer", .module = layer },
        },
    });
    const optim = b.addModule("optim", .{
        .root_source_file = b.path("src/optim.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "mlp", .module = mlp },
        },
    });
    mlp.addImport("optim", optim);

    const mlp_tests = b.addTest(.{ .root_module = mlp });
    const optim_tests = b.addTest(.{ .root_module = optim });
    const run_mlp_tests = b.addRunArtifact(mlp_tests);
    const run_optim_tests = b.addRunArtifact(optim_tests);

    const test_step = b.step("test", "run tests");
    test_step.dependOn(&run_mlp_tests.step);
    test_step.dependOn(&run_optim_tests.step);
}
