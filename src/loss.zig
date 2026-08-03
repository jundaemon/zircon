/// returns mean squared error loss then derivative of mean squared error loss
pub fn MSE(comptime n: usize, predicted: [n]f32, expected: [n]f32) struct { f32, [n]f32 } {
    if (n < 1) @compileError("output should be of size 1 or larger");

    const ae = @as(@Vector(n, f32), predicted) - expected;
    return .{ @reduce(.Add, ae * ae) / n, ae * @as(@Vector(n, f32), @splat(2 / @as(f32, n))) };
}
