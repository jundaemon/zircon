const std = @import("std");
const Random = std.Random;

pub fn uniform_float(rand: Random) f32 {
    return rand.float(f32) * 2 - 1;
}
