const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mlp = b.addModule("mlp", .{
        .root_source_file = b.path("src/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    const loss = b.addModule("loss", .{
        .root_source_file = b.path("src/loss.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mlp_tests = b.addTest(.{ .root_module = mlp });
    const loss_tests = b.addTest(.{ .root_module = loss });

    const run_mlp_tests = b.addRunArtifact(mlp_tests);
    const run_loss_tests = b.addRunArtifact(loss_tests);

    const test_step = b.step("test", "run tests");
    test_step.dependOn(&run_mlp_tests.step);
    test_step.dependOn(&run_loss_tests.step);
}
