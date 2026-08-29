const std = @import("std");
const fmt = std.fmt;
const Io = std.Io;
const testing = std.testing;

pub const function = @import("function.zig");
pub const optim = @import("optim.zig");
pub const layer = @import("layer.zig");
pub const mlp = @import("mlp.zig");

const Loss = function.Loss;
const Optimizer = optim.Optimizer;
const MLPConfig = mlp.MLPConfig;
const MLP = mlp.MLP;

test "mlp double passes 1" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_1_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 3, 5, 7, 4, 1 },
        .f = &.{ .Tanh, .Tanh, .Tanh, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_1_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{});

    const pred = model.forward(.{ 1, 2 });
    const loss = loss_fn.eval(pred, .{0.1});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_1_updated_weights");

    const pred_ = model.forward(.{ 3, 2 });
    const loss_ = loss_fn.eval(pred_, .{0.2});
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_1_final_weights");
}

test "mlp double passes 2" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_2_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 3, 5, 7, 4, 1 },
        .f = &.{ .Tanh, .Tanh, .Tanh, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_2_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{ .lr = 1e-4 });

    const pred = model.forward(.{ 1, 2 });
    const loss = loss_fn.eval(pred, .{0.1});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_2_updated_weights");

    const pred_ = model.forward(.{ 3, 2 });
    const loss_ = loss_fn.eval(pred_, .{0.2});
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_2_final_weights");
}

test "mlp double passes 3" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_3_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 1,
        .outs = &.{ 2, 4, 6, 5, 3 },
        .f = &.{ .ReLU, .ReLU, .ReLU, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_3_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .SGD,
    }) = try .init(&model, .{ .lr = 1e-4, .momentum = 0.9 });

    const pred = model.forward(.{2});
    const loss = loss_fn.eval(pred, .{ 0.5, 0.2, 0.3 });
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_3_updated_weights");

    const pred_ = model.forward(.{3});
    const loss_ = loss_fn.eval(pred_, .{ 0.7, 0.1, 0.2 });
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_3_final_weights");
}

test "mlp double passes 4" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_4_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 3,
        .outs = &.{ 9, 7, 5, 3, 1 },
        .f = &.{ .ReLU, .Tanh, .ReLU, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_4_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = try .init(&model, .{});

    const pred = model.forward(.{ 1, 2, 3 });
    const loss = loss_fn.eval(pred, .{0.4});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_4_updated_weights");

    const pred_ = model.forward(.{ 5, 4, 3 });
    const loss_ = loss_fn.eval(pred_, .{0.3});
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_4_final_weights");
}

test "mlp double passes 5" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_5_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 3,
        .outs = &.{ 9, 7, 5, 3, 1 },
        .f = &.{ .ReLU, .Tanh, .ReLU, .Tanh, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_5_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .RMSprop,
    }) = try .init(&model, .{ .lr = 1e-4, .alpha = 0.9, .epsilon = 1e-7 });

    const pred = model.forward(.{ 1, 2, 3 });
    const loss = loss_fn.eval(pred, .{0.4});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_5_updated_weights");

    const pred_ = model.forward(.{ 5, 4, 3 });
    const loss_ = loss_fn.eval(pred_, .{0.3});
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_5_final_weights");
}

test "mlp double passes 6" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_6_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 4,
        .outs = &.{ 10, 8, 6, 4, 2 },
        .f = &.{ .Tanh, .ReLU, .Tanh, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_6_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .Adam,
    }) = try .init(&model, .{});

    const pred = model.forward(.{ 4, 3, 2, 1 });
    const loss = loss_fn.eval(pred, .{ 0.1, 0.2 });
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_6_updated_weights");

    const pred_ = model.forward(.{ 3, 4, 5, 6 });
    const loss_ = loss_fn.eval(pred_, .{ 0.4, 0.3 });
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_6_final_weights");
}

test "mlp double passes 7" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_7_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 4,
        .outs = &.{ 10, 8, 6, 4, 2 },
        .f = &.{ .Tanh, .ReLU, .Tanh, .ReLU, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_7_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .MSE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .Adam,
    }) = try .init(&model, .{ .lr = 1e-4, .beta = 0.99, .beta_2 = 0.99, .epsilon = 1e-7 });

    const pred = model.forward(.{ 4, 3, 2, 1 });
    const loss = loss_fn.eval(pred, .{ 0.1, 0.2 });
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_7_updated_weights");

    const pred_ = model.forward(.{ 3, 4, 5, 6 });
    const loss_ = loss_fn.eval(pred_, .{ 0.4, 0.3 });
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_7_final_weights");
}

test "mlp double passes 8" {
    const io = testing.io;
    var file: Io.File = try Io.Dir.cwd().createFile(io, "tests/cases/mlp_8_losses", .{});
    defer file.close(io);

    const mlp_config = MLPConfig{
        .in = 2,
        .outs = &.{ 2, 2, 1 },
        .f = &.{ .Sigmoid, .Sigmoid, .None },
        .seed = 1,
    };
    var model: MLP(mlp_config) = .init();
    try model.save(io, "tests/cases/mlp_8_initial_weights");

    const loss_fn: Loss(.{
        .mlp_config = mlp_config,
        .loss_function = .BCE,
    }) = .init(&model);
    var optimizer: Optimizer(.{
        .mlp_config = mlp_config,
        .optimizer = .Adam,
    }) = try .init(&model, .{});

    const pred = model.forward(.{ 1, 1 });
    const loss = loss_fn.eval(pred, .{0});
    var buf: [30]u8 = undefined;
    const loss_str = try fmt.bufPrint(&buf, "{d}\n", .{loss.item});
    try file.writeStreamingAll(io, loss_str);

    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    try model.save(io, "tests/cases/mlp_8_updated_weights");

    const pred_ = model.forward(.{ 0, 1 });
    const loss_ = loss_fn.eval(pred_, .{1});
    var buf_: [30]u8 = undefined;
    const loss_str_ = try fmt.bufPrint(&buf_, "{d}\n", .{loss_.item});
    try file.writeStreamingAll(io, loss_str_);

    loss_.backward();
    optimizer.step();
    try model.save(io, "tests/cases/mlp_8_final_weights");
}
