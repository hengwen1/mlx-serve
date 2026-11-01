//! HF-transformers export packs at load: the layout `transformers` writes for a composite checkpoint
//! (`Glm5NextForConditionalGeneration`-style) against the layout this server reads. Each move is a no-op on a pack
//! that already matches, and the FP8 moves run only for a config that declares `quant_method: "fp8"` with
//! `weight_block_size [128, 128]` — a 64 x 128 grid on a short projection is indistinguishable from a 128 x 128 one
//! by geometry alone.
//!
//! - `stackExperts`: per-expert MoE tensors (`experts.<i>.gate_proj.weight`) stacked along axis 0
//!   (`experts.gate_proj.weight`), as mlx-lm writes them.
//! - `requantBlockFp8Banks`: a block-FP8 expert bank (e4m3 codes x f32 128 x 128 scales) re-encoded affine 8-bit on
//!   the SAME 128-wide groups with f32 scales — near-lossless (the group geometry and scale dtype match the source;
//!   only the code alphabet changes), and the bank rides every affine gather and one-token kernel.
//! - `expandBlockFp8RowScales`: a dense block-FP8 projection's scale grid broadcast to the per-row scales the
//!   row-scaled e4m3 reader takes; the codes stay untouched and the values stay exact.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");
const Weights = @import("resident.zig").Weights;

const gpa = std.heap.c_allocator;
const block: c_int = 128;

/// A U8 code tensor `[.., rows, cols]` beside an F32 scale grid `[.., ceil(rows / 128), ceil(cols / 128)]`.
/// Only trusted under a config that declares the 128 x 128 block (see the module doc).
pub fn isBlockFp8Grid(w: mlx.mlx_array, grid: mlx.mlx_array) bool {
    if (w.ctx == null or grid.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint8 or mlx.mlx_array_dtype(grid) != .float32) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(grid);
    if (ws.len < 2 or ss.len != ws.len) return false;
    return ss[ss.len - 2] == @divFloor(ws[ws.len - 2] + block - 1, block) and
        ss[ss.len - 1] == @divFloor(ws[ws.len - 1] + block - 1, block);
}

/// A tile/broadcast view's raw buffer holds only its unique elements (binary ops propagate its
/// strides and add-scalar-zero folds), so a flat reshape — which a non-row-major layout can never
/// be viewed as — is the one reliable materializer (see dsv4's `toHostF32`). Returns `x` itself when
/// it is already a dense row-major buffer.
fn materialize(x: mlx.mlx_array, count: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{@intCast(count)}, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    return flat;
}

const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const leaves = [_][]const u8{ ".weight", ".weight_scale_inv", ".scales", ".biases" };

/// One `experts.<i>.<proj><leaf>` key split into its stacked name, expert index and tail.
const ExpertKey = struct {
    stacked: []const u8,
    expert: u32,
    key: []const u8,
};

fn parseExpertKey(key: []const u8) ?ExpertKey {
    const at = std.mem.indexOf(u8, key, ".mlp.experts.") orelse return null;
    const rest = key[at + ".mlp.experts.".len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const expert = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const tail = rest[dot + 1 ..];
    var matched: ?[]const u8 = null;
    for (projs) |p| if (std.mem.startsWith(u8, tail, p)) {
        matched = p;
        break;
    };
    const proj = matched orelse return null;
    const leaf = tail[proj.len..];
    for (leaves) |l| if (std.mem.eql(u8, leaf, l)) {
        const stacked = std.fmt.allocPrint(gpa, "{s}{s}{s}", .{ key[0 .. at + ".mlp.experts.".len], proj, leaf }) catch return null;
        return .{ .stacked = stacked, .expert = expert, .key = key };
    };
    return null;
}

/// `experts.<i>.<proj><leaf>` -> stacked `experts.<proj><leaf>` along axis 0, in expert order. A group with a gap
/// (an expert missing one of its tensors) is refused rather than stacked short.
pub fn stackExperts(weights: *Weights, s: mlx.mlx_stream) !void {
    const a = weights.allocator;
    var groups = std.StringHashMap(std.ArrayList(ExpertKey)).init(a);
    defer {
        var it = groups.iterator();
        while (it.next()) |e| {
            for (e.value_ptr.items) |k| a.free(k.stacked);
            e.value_ptr.deinit(a);
        }
        groups.deinit();
    }
    const keys = try collectKeys(weights, a);
    defer a.free(keys);
    defer freeKeys(a, keys);
    for (keys) |key| {
        const ek = parseExpertKey(key) orelse continue;
        const g = try groups.getOrPut(ek.stacked);
        if (!g.found_existing) g.value_ptr.* = .empty;
        try g.value_ptr.append(a, .{ .stacked = try a.dupe(u8, ek.stacked), .expert = ek.expert, .key = key });
    }
    var stacked_any = false;
    var git = groups.iterator();
    while (git.next()) |e| {
        const list = e.value_ptr.items;
        std.mem.sort(ExpertKey, list, {}, struct {
            fn less(_: void, x: ExpertKey, y: ExpertKey) bool {
                return x.expert < y.expert;
            }
        }.less);
        for (list, 0..) |k, i| if (k.expert != i) return error.HfExpertGap;
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (list) |k| try mlx.check(mlx.mlx_vector_array_append_value(vec, weights.get(k.key).?));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        // STACK, not concat: the per-expert tensors are 2-D `[R, K]` and the reader wants the
        // expert axis `[E, R, K]`.
        try mlx.check(mlx.mlx_stack_axis(&out, vec, 0, s));
        try mlx.check(mlx.mlx_array_eval(out));
        for (list) |k| weights.remove(k.key);
        try weights.put(e.key_ptr.*, out);
        stacked_any = true;
    }
    if (stacked_any) log.info("[hf-pack] per-expert tensors stacked\n", .{});
}

/// Block-FP8 expert banks (U8 codes `[E, rows, cols]` x F32 128 x 128 grids) as affine 8-bit on the same groups:
/// codes u32, f32 scales, zero f32 biases. Near-lossless: the dequantized values only re-round from e4m3 to the
/// finer int8 grid over the same 128 columns and scales.
pub fn requantBlockFp8Banks(weights: *Weights, s: mlx.mlx_stream) !void {
    const a = weights.allocator;
    const keys = try collectKeys(weights, a);
    defer a.free(keys);
    defer freeKeys(a, keys);
    var banks: std.ArrayList([]const u8) = .empty;
    defer banks.deinit(a);
    defer freeKeys(a, banks.items);
    for (keys) |key| {
        if (!std.mem.endsWith(u8, key, ".weight")) continue;
        const w = weights.get(key).?;
        const ws = mlx.getShape(w);
        if (ws.len != 3) continue;
        var nb: [320]u8 = undefined;
        const grid_name = std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{key[0 .. key.len - ".weight".len]}) catch continue;
        const grid = weights.get(grid_name) orelse continue;
        if (!isBlockFp8Grid(w, grid)) continue;
        try banks.append(a, try a.dupe(u8, key));
    }
    for (banks.items) |key| {
        const base = key[0 .. key.len - ".weight".len];
        const w = weights.get(key).?;
        var nb: [320]u8 = undefined;
        const grid = weights.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base})).?;
        const ws = mlx.getShape(w);
        const rows = ws[ws.len - 2];
        const cols = ws[ws.len - 1];
        // The multiply reshapes codes to [E, rows/128, 128, cols/128, 128]; a ragged edge has no slot.
        if (@mod(rows, block) != 0 or @mod(cols, block) != 0) return error.HfFp8RaggedBank;

        var codes = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(codes);
        try mlx.check(mlx.mlx_from_fp8(&codes, w, .float32, s));
        var c5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c5);
        try mlx.check(mlx.mlx_reshape(&c5, codes, &[_]c_int{ ws[0], @divTrunc(rows, block), block, @divTrunc(cols, block), block }, 5, s));
        var g5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g5);
        try mlx.check(mlx.mlx_reshape(&g5, grid, &[_]c_int{ ws[0], @divTrunc(rows, block), 1, @divTrunc(cols, block), 1 }, 5, s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, c5, g5, s));
        // The multiply over a broadcast grid strides its output; a reshape that merges the stride-0
        // axes cannot be a view, so it copies. Two axes: a bank's flat count passes 2^31.
        const per_expert: c_int = rows * cols;
        var dense_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense_flat);
        try mlx.check(mlx.mlx_reshape(&dense_flat, scaled, &[_]c_int{ ws[0], per_expert }, 2, s));
        try mlx.check(mlx.mlx_array_eval(dense_flat));
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_reshape(&dense, dense_flat, &[_]c_int{ ws[0], rows, cols }, 3, s));

        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_quantize(&qv, dense, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
        // affine quantize emits [codes, scales, biases].
        var codes_q = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(codes_q);
        try mlx.check(mlx.mlx_vector_array_get(&codes_q, qv, 0));
        var scales_q = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scales_q);
        try mlx.check(mlx.mlx_vector_array_get(&scales_q, qv, 1));
        var biases_q = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(biases_q);
        try mlx.check(mlx.mlx_vector_array_get(&biases_q, qv, 2));
        try mlx.check(mlx.mlx_array_eval(codes_q));
        try mlx.check(mlx.mlx_array_eval(scales_q));
        try mlx.check(mlx.mlx_array_eval(biases_q));

        try weights.put(key, codes_q);
        try weights.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), scales_q);
        try weights.put(try std.fmt.bufPrint(&nb, "{s}.biases", .{base}), biases_q);
        weights.remove(try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base}));
    }
    if (banks.items.len > 0) log.info("[hf-pack] {d} block-FP8 banks re-encoded affine 8-bit (same 128 groups)\n", .{banks.items.len});
}

/// A dense block-FP8 projection's f32 scale grid `[ceil(rows / 128), ceil(cols / 128)]` broadcast to per-row scales
/// `[rows, cols / 128]` under the same name: the row-scaled e4m3 reader's layout, values unchanged. `cols` must be a
/// multiple of 128 (every projection this pack ships is).
pub fn expandBlockFp8RowScales(weights: *Weights, s: mlx.mlx_stream) !void {
    const keys = try collectKeys(weights, std.heap.page_allocator);
    defer std.heap.page_allocator.free(keys);
    defer freeKeys(std.heap.page_allocator, keys);
    var expanded: usize = 0;
    for (keys) |key| {
        if (!std.mem.endsWith(u8, key, ".weight_scale_inv")) continue;
        const grid = weights.get(key) orelse continue;
        const gs = mlx.getShape(grid);
        if (gs.len != 2) continue;
        var nb: [320]u8 = undefined;
        const w_name = std.fmt.bufPrint(&nb, "{s}.weight", .{key[0 .. key.len - ".weight_scale_inv".len]}) catch continue;
        const w = weights.get(w_name) orelse continue;
        const ws = mlx.getShape(w);
        if (ws.len != 2 or mlx.mlx_array_dtype(w) != .uint8) continue;
        if (gs[0] * block < ws[0] or gs[1] * block != ws[1]) continue;

        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_repeat_axis(&wide, grid, block, 0, s));
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_slice(&rows, wide, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ws[0], gs[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const flat = try materialize(rows, @intCast(ws[0] * gs[1]), s);
        defer _ = mlx.mlx_array_free(flat);
        var owned = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(owned);
        try mlx.check(mlx.mlx_reshape(&owned, flat, &[_]c_int{ ws[0], gs[1] }, 2, s));
        weights.replace(key, owned);
        expanded += 1;
    }
    if (expanded > 0) log.info("[hf-pack] {d} block-FP8 scale grids expanded to per-row scales\n", .{expanded});
}

/// Resident-bytes overhead the two FP8 moves add over the stored bytes, scanned from the shard headers
/// (the bill runs before the load): a bank gains f32 scales and zero biases (each `codes / 4`), a dense
/// projection gains per-row f32 scales (`codes / 8`).
pub fn fp8OverheadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var grids = std.StringHashMap(void).init(a);
    var weights: std.ArrayList(struct { key: []const u8, bytes: u64, ndim: usize }) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        var file = try dir.openFile(io, entry.name, .{});
        defer file.close(io);
        var rbuf: [8192]u8 = undefined;
        var rs = file.reader(io, &rbuf);
        const len = try rs.interface.takeInt(u64, .little);
        if (len > 64 * 1024 * 1024) return error.InvalidSafetensorsHeader;
        const raw = try rs.interface.readAlloc(a, @intCast(len));
        var parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
        if (parsed != .object) return error.InvalidSafetensorsHeader;
        var pit = parsed.object.iterator();
        while (pit.next()) |e| {
            if (e.value_ptr.* != .object) continue;
            const offs = e.value_ptr.object.get("data_offsets") orelse continue;
            const shape = e.value_ptr.object.get("shape") orelse continue;
            if (offs != .array or offs.array.items.len != 2 or shape != .array) continue;
            const bytes: u64 = @intCast(offs.array.items[1].integer - offs.array.items[0].integer);
            if (std.mem.endsWith(u8, e.key_ptr.*, ".weight_scale_inv")) {
                try grids.put(e.key_ptr.*, {});
            } else if (std.mem.endsWith(u8, e.key_ptr.*, ".weight")) {
                try weights.append(a, .{ .key = try a.dupe(u8, e.key_ptr.*), .bytes = bytes, .ndim = shape.array.items.len });
            }
        }
    }
    var total: u64 = 0;
    for (weights.items) |wt| {
        const base = wt.key[0 .. wt.key.len - ".weight".len];
        var nb: [320]u8 = undefined;
        const grid = std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base}) catch continue;
        if (!grids.contains(grid)) continue;
        total += wt.bytes / (if (wt.ndim == 3) @as(u64, 4) else 8);
    }
    return total;
}

fn e4m3Value(code: u8) f32 {
    const sign: f32 = if (code & 0x80 != 0) -1 else 1;
    const exp: u8 = (code >> 3) & 0xf;
    const mant: u8 = code & 0x7;
    if (exp == 0) return sign * @as(f32, @floatFromInt(mant)) / 8.0 * 0.015625;
    if (exp == 0xf and mant == 0x7) return std.math.nan(f32);
    return sign * std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, exp) - 7))) * (1.0 + @as(f32, @floatFromInt(mant)) / 8.0);
}

fn collectKeys(weights: *const Weights, a: std.mem.Allocator) ![][]const u8 {
    var keys: std.ArrayList([]const u8) = .empty;
    errdefer freeKeys(a, keys.items);
    var it = weights.map.keyIterator();
    while (it.next()) |k| try keys.append(a, try a.dupe(u8, k.*));
    return keys.toOwnedSlice(a);
}

fn freeKeys(a: std.mem.Allocator, keys: [][]const u8) void {
    for (keys) |k| a.free(k);
}

// ── tests ──

const testing = std.testing;

fn cpuStream() !mlx.mlx_stream {
    const s = mlx.mlx_default_cpu_stream_new();
    errdefer _ = mlx.mlx_stream_free(s);
    return s;
}

test "hf_pack: isBlockFp8Grid checks the 128 x 128 grid on the last two axes" {
    const s = try cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var codes: [4 * 256]u8 = undefined;
    var grid: [1 * 2]f32 = @splat(1.0);
    const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 256 }, 2, .uint8);
    const g = mlx.mlx_array_new_data(&grid, &[_]c_int{ 1, 2 }, 2, .float32);
    try testing.expect(isBlockFp8Grid(w, g));
    const g3 = mlx.mlx_array_new_data(&grid, &[_]c_int{ 2, 2 }, 2, .float32);
    try testing.expect(!isBlockFp8Grid(w, g3));
    const wf = mlx.mlx_array_new_data(&codes, &[_]c_int{ 4, 256 }, 2, .bfloat16);
    try testing.expect(!isBlockFp8Grid(wf, g));
}

test "hf_pack: per-expert tensors stack in expert order" {
    const s = try cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    const e0 = [_]f32{ 1, 2, 3, 4 };
    const e1 = [_]f32{ 5, 6, 7, 8 };
    // Ship them out of order on purpose; the stack must sort.
    try w.put("m.layers.0.mlp.experts.1.gate_proj.weight", mlx.mlx_array_new_data(&e1, &[_]c_int{ 1, 4 }, 2, .float32));
    try w.put("m.layers.0.mlp.experts.0.gate_proj.weight", mlx.mlx_array_new_data(&e0, &[_]c_int{ 1, 4 }, 2, .float32));
    try w.put("m.layers.0.mlp.experts.1.up_proj.weight", mlx.mlx_array_new_data(&e1, &[_]c_int{ 1, 4 }, 2, .float32));
    try w.put("m.layers.0.mlp.experts.0.up_proj.weight", mlx.mlx_array_new_data(&e0, &[_]c_int{ 1, 4 }, 2, .float32));
    try stackExperts(&w, s);
    try testing.expect(w.get("m.layers.0.mlp.experts.0.gate_proj.weight") == null);
    const got = w.get("m.layers.0.mlp.experts.gate_proj.weight").?;
    // STACK adds the expert axis (a concat of 2-D tensors would flatten the experts in).
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 1, 4 }, mlx.getShape(got));
    const host = try testing.allocator.alloc(f32, 8);
    defer testing.allocator.free(host);
    try mlx.check(mlx.mlx_array_eval(got));
    _ = mlx.mlx_synchronize(s);
    const ptr = mlx.mlx_array_data_float32(got).?;
    @memcpy(host, ptr);
    try testing.expectEqualSlices(f32, &e0, host[0..4]);
    try testing.expectEqualSlices(f32, &e1, host[4..8]);
    // A gap is refused, not stacked short.
    try w.put("m.layers.1.mlp.experts.2.gate_proj.weight", mlx.mlx_array_new_data(&e1, &[_]c_int{ 1, 4 }, 2, .float32));
    try testing.expectError(error.HfExpertGap, stackExperts(&w, s));
}

test "hf_pack: a block-FP8 bank re-encodes affine 8-bit near-exactly" {
    const s = try cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    // E=2, R=128, K=256: one grid row per expert.
    var codes: [2 * 128 * 256]u8 = undefined;
    for (&codes) |*c| c.* = rnd.intRangeAtMost(u8, 0, 0x7e); // finite e4m3 codes, no NaN
    var grid: [2 * 1 * 2]f32 = undefined;
    for (&grid) |*g| g.* = rnd.float(f32) * 0.01 + 0.001;
    try w.put("m.layers.0.mlp.experts.gate_proj.weight", mlx.mlx_array_new_data(&codes, &[_]c_int{ 2, 128, 256 }, 3, .uint8));
    try w.put("m.layers.0.mlp.experts.gate_proj.weight_scale_inv", mlx.mlx_array_new_data(&grid, &[_]c_int{ 2, 1, 2 }, 3, .float32));

    // The reference host-side: e4m3 code x grid value (the load.zig decode).
    const want = try testing.allocator.alloc(f32, 2 * 128 * 256);
    defer testing.allocator.free(want);
    for (0..2) |e| for (0..128) |r| for (0..256) |c| {
        want[(e * 128 + r) * 256 + c] = e4m3Value(codes[(e * 128 + r) * 256 + c]) * grid[e * 2 + c / 128];
    };

    try requantBlockFp8Banks(&w, s);
    try testing.expect(w.get("m.layers.0.mlp.experts.gate_proj.weight_scale_inv") == null);
    const qw = w.get("m.layers.0.mlp.experts.gate_proj.weight").?;
    const qs = w.get("m.layers.0.mlp.experts.gate_proj.scales").?;
    const qb = w.get("m.layers.0.mlp.experts.gate_proj.biases").?;
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(qw));
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 128, 64 }, mlx.getShape(qw));
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(qs));
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 128, 2 }, mlx.getShape(qs));
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 128, 2 }, mlx.getShape(qb));

    // The requantized bank dequantized matches the e4m3 x grid values within one int8 code step
    // (each group's amax / 127): the group geometry and scales are the source's own.
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, qw, qs, qb, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_array_eval(deq));
    _ = mlx.mlx_synchronize(s);
    const got = mlx.mlx_array_data_float32(deq).?[0 .. 2 * 128 * 256];
    for (0..2) |e| for (0..128) |r| for (0..2) |kb| {
        var amax: f32 = 0;
        for (0..128) |cc| amax = @max(amax, @abs(want[((e * 128 + r) * 256) + kb * 128 + cc]));
        // MLX affine clamps codes to [-128, 127] over an amax/127 scale, so the low tail may
        // overshoot by up to 1/127 more than a step.
        const bar = 2.0 * amax / 127.0 + 1e-6;
        for (0..128) |cc| {
            const at = ((e * 128 + r) * 256) + kb * 128 + cc;
            try testing.expect(@abs(got[at] - want[at]) <= bar);
        }
    };
}

test "hf_pack: a dense block-FP8 grid expands to exact per-row scales" {
    const s = try cpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    var codes: [128 * 256]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i % 0x7f);
    var grid = [_]f32{ 0.5, 2.0 };
    try w.put("m.layers.0.self_attn.q_proj.weight", mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 256 }, 2, .uint8));
    try w.put("m.layers.0.self_attn.q_proj.weight_scale_inv", mlx.mlx_array_new_data(&grid, &[_]c_int{ 1, 2 }, 2, .float32));
    try expandBlockFp8RowScales(&w, s);
    const got = w.get("m.layers.0.self_attn.q_proj.weight_scale_inv").?;
    try testing.expectEqualSlices(c_int, &[_]c_int{ 128, 2 }, mlx.getShape(got));
    try mlx.check(mlx.mlx_array_eval(got));
    _ = mlx.mlx_synchronize(s);
    const ptr = mlx.mlx_array_data_float32(got).?;
    for (0..128) |r| {
        try testing.expectEqual(@as(f32, 0.5), ptr[r * 2]);
        try testing.expectEqual(@as(f32, 2.0), ptr[r * 2 + 1]);
    }
}



