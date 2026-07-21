const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const nn = b.addModule("nn", .{
        .root_source_file = b.path("src/nn.zig"),
        .target = target,
        .optimize = optimize,
    });

    const nn_tests = b.addTest(.{ .root_module = nn });
    const run_nn_tests = b.addRunArtifact(nn_tests);
    const test_step = b.step("test", "run tests");
    test_step.dependOn(&run_nn_tests.step);
}
