const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lossf = b.createModule(.{
        .root_source_file = b.path("src/lossf.zig"),
        .target = target,
        .optimize = optimize,
    });
    const mlp = b.addModule("mlp", .{
        .root_source_file = b.path("src/mlp.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "lossf", .module = lossf },
        },
    });

    const mlp_tests = b.addTest(.{ .root_module = mlp });
    const run_mlp_tests = b.addRunArtifact(mlp_tests);
    const test_step = b.step("test", "run tests");
    test_step.dependOn(&run_mlp_tests.step);
}
