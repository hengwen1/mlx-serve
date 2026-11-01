//! Sushi packs at load: the tensor moves `sdk/quant/exl3/sushi_pack.zig` provides, applied per the model's config
//! (GLM-5-Next and MiMo layouts).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const Weights = @import("../../../sdk/weights/resident.zig").Weights;
const ModelConfig = model_mod.ModelConfig;
const adaptMimoHeads = @import("../../../sdk/quant/exl3/sushi_pack.zig").adaptMimoHeads;
const decodeDenseMlp = @import("../../../sdk/quant/exl3/sushi_pack.zig").decodeDenseMlp;
const renameHyperConnections = @import("../../../sdk/quant/exl3/sushi_pack.zig").renameHyperConnections;
const splitFusedQkv = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitFusedQkv;
const splitKvB = @import("../../../sdk/quant/exl3/sushi_pack.zig").splitKvB;
const hf_pack = @import("../../../sdk/weights/hf_pack.zig");

/// Also serves NVIDIA's ModelOpt GLM pack and GLM's streaming packs (`expert_bank`), which ship the same source layout.
pub fn adapt(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    if (config.exl3 == null and !config.modelopt and !config.expert_bank and !config.isGlm5() and !config.block_fp8) return;
    if (config.isMimo() and config.exl3 != null) {
        try dequantFp8Trunk(config, weights, s);
    }
    if (config.isGlm5()) {
        // An HF-transformers export names the NextN layer `layers.<N>` (`shared_head.norm` beside it), ships the
        // Sinkhorn tables as `hc_attn_`/`hc_ffn_`, and keeps `kv_b_proj` whole; all three are no-ops on a pack
        // that already matches (mlx-community, ModelOpt, Sushi).
        try @import("../../../sdk/quant/exl3/sushi_pack.zig").renameGlmMtp(weights, "model.language_model", config.num_hidden_layers);
        try renameHyperConnections(weights);
        try splitKvB(weights, .{
            .heads = config.num_attention_heads,
            .nope = config.dsa_head_dim,
            .v = config.dsa_head_dim,
            .latent = config.dsa_kv_lora_rank,
        }, s);
    }
    if (config.block_fp8) {
        // The export's per-expert block-FP8 banks: stacked along axis 0, then re-encoded affine 8-bit on the
        // same 128 groups (near-lossless, rides every affine gather); the dense projections' 128 x 128 grids
        // broadcast to the per-row scales the row-scaled e4m3 reader takes, codes untouched.
        try hf_pack.stackExperts(weights, s);
        try hf_pack.requantBlockFp8Banks(weights, s);
        try hf_pack.expandBlockFp8RowScales(weights, s);
    }
    if (config.isGlm5() and config.expert_bank) {
        // A streaming pack: the NextN layer's experts in their own NVFP4 bank.
        if (config.pack_dir) |dir| try @import("../../../sdk/quant/nvfp4/stream_bank.zig").load(weights, dir, "mtp-expert-manifest-nvfp4.json", "model.language_model.mtp.0.block");
    }
}

/// MiMo's trunk is block-FP8 as the source stores it; this engine reads dense projections, so
/// the fused rank-local `qkv_proj` becomes q/k/v and the layer-0 MLP dense, both bf16.
fn dequantFp8Trunk(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    var buf: [160]u8 = undefined;
    for (0..config.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        const base = try std.fmt.bufPrint(&buf, "{s}.layers.{d}", .{ config.weight_prefix, li });
        try splitFusedQkv(weights, base, qkvRows(config, li), s);
        if (li < config.first_k_dense_replace) try decodeDenseMlp(weights, base, s);
    }
    // The MTP heads are stored the same way and attend in a sliding layer's geometry.
    try adaptMimoHeads(weights, qkvRows(config, config.firstSlidingLayer() orelse return), s);
}

fn qkvRows(config: *const ModelConfig, li: u32) [3]u64 {
    return .{
        @as(u64, config.layerNumHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerHeadDim(li),
        @as(u64, config.layerKVHeads(li)) * config.layerVHeadDim(li),
    };
}

// ── tests ──

const testing = std.testing;

test "sushi pack: the MiMo MTP heads get the trunk's split QKV and decoded MLP" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // Layer 0 slides (pattern 2): its geometry is the heads'. q 32 + k 32 + v 32 rows over 4 ranks.
    var config: ModelConfig = .{ .model_type = "mimo_v2", .num_hidden_layers = 2, .num_attention_heads = 4, .num_key_value_heads = 4, .head_dim = 8, .has_sliding_window = true, .sliding_window_pattern = 2, .weight_prefix = "model" };
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    var codes: [128 * 128]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 13 + 3) % 0x70);
    const ones = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    try w.put("model.mtp.layers.0.eh_proj.weight", mlx.mlx_array_new_float(1));
    try w.put("model.mtp.layers.0.self_attn.qkv_proj.weight", mlx.mlx_array_new_data(&codes, &[_]c_int{ 96, 128 }, 2, .uint8));
    try w.put("model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv", mlx.mlx_array_new_data(&ones, &[_]c_int{ 4, 1 }, 2, .float32));
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        var k2: [96]u8 = undefined;
        try w.put(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight", .{proj}), mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 128 }, 2, .uint8));
        try w.put(try std.fmt.bufPrint(&k2, "model.mtp.layers.0.mlp.{s}.weight_scale_inv", .{proj}), mlx.mlx_array_new_data(&ones, &[_]c_int{ 1, 1 }, 2, .float32));
    }
    try dequantFp8Trunk(&config, &w, s);

    try testing.expect(w.get("model.mtp.layers.0.self_attn.qkv_proj.weight") == null);
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        const part = w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.self_attn.{s}.weight", .{proj})).?;
        try testing.expectEqualSlices(c_int, &.{ 32, 128 }, mlx.getShape(part));
        try testing.expect(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.self_attn.{s}.scales", .{proj})) != null);
    }
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var k1: [96]u8 = undefined;
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight", .{proj})).?));
        try testing.expect(w.get(try std.fmt.bufPrint(&k1, "model.mtp.layers.0.mlp.{s}.weight_scale_inv", .{proj})) == null);
    }
}

test "sushi adapt: a plain GLM pack gets the export's renames, and block-FP8 banks re-encode" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var config: ModelConfig = .{ .model_type = "glm5_next", .num_hidden_layers = 1, .num_attention_heads = 4, .dsa_head_dim = 1, .dsa_kv_lora_rank = 2, .block_fp8 = true };
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    // HF-export names: the NextN layer under `layers.1`, the Sinkhorn tables as `hc_attn_`, a whole `kv_b_proj`.
    try w.put("model.language_model.layers.1.hc_attn_fn", mlx.mlx_array_new_float(1));
    try w.put("model.language_model.layers.1.shared_head.norm.weight", mlx.mlx_array_new_float(1));
    const kvb = [_]f32{ 1, 2, 3, 4 };
    try w.put("model.language_model.layers.0.self_attn.kv_b_proj.weight", mlx.mlx_array_new_data(&kvb, &[_]c_int{ 8, 2 }, 2, .float32));
    // One per-expert block-FP8 bank: e4m3 codes x a 128 x 128 grid.
    var codes: [128 * 128]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 13 + 3) % 0x70);
    var grid: f32 = 0.25;
    try w.put("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 128 }, 2, .uint8));
    try w.put("model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", mlx.mlx_array_new_data(&grid, &[_]c_int{ 1, 1 }, 2, .float32));
    try adapt(&config, &w, s);

    try testing.expect(w.get("model.language_model.mtp.0.norm.weight") != null);
    // The NextN layer moved to `mtp.0`, its hc table under the head's block.
    try testing.expect(w.get("model.language_model.layers.1.hc_attn_fn") == null);
    try testing.expect(w.get("model.language_model.mtp.0.block.attn_hc.fn") != null);
    try testing.expect(w.get("model.language_model.layers.0.self_attn.embed_q.weight") != null);
    // The bank stacked (the expert axis exists) and re-encoded affine 8-bit on the same groups.
    const bank = w.get("model.language_model.layers.0.mlp.experts.gate_proj.weight").?;
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, 128, 32 }, mlx.getShape(bank));
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(bank));
    try testing.expect(w.get("model.language_model.layers.0.mlp.experts.gate_proj.scales") != null);
    try testing.expect(w.get("model.language_model.layers.0.mlp.experts.gate_proj.biases") != null);
    try testing.expect(w.get("model.language_model.layers.0.mlp.experts.gate_proj.weight_scale_inv") == null);
}
