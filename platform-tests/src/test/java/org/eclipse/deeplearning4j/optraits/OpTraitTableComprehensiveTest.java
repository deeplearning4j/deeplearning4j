/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.optraits;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Comprehensive JUnit 5 coverage for the libnd4j op trait classification system.
 *
 * <p>This test exists to catch the class of regression where ops are misclassified
 * against the table maintained in {@code libnd4j/include/ops/impl/OpTraitTable.cpp}
 * and {@code libnd4j/include/ops/declarable/OpDescriptor.h}. Specifically, it
 * guards against the recent drift where {@code gather}, {@code gather_nd}, and
 * {@code repeat} were incorrectly tagged as {@code OP_TRAIT_VALUE_DEPENDENT_SHAPE}
 * and caused SHAPES_FROZEN {@code LIFECYCLE_ERROR} false positives.
 *
 * <p>The mask-bit Java-side constants mirror the C++ enum {@code OpTraits} in
 * {@code OpDescriptor.h}. All 29 named bits (0..28) are modelled below.
 */
@DisplayName("libnd4j OpTraitTable — comprehensive trait coverage")
public class OpTraitTableComprehensiveTest {

    // ── Trait bits — keep 1:1 with OpDescriptor.h ───────────────────────────────
    private static final int OP_TRAIT_UNARY_ELEMENTWISE     = 1 << 0;
    private static final int OP_TRAIT_BINARY_ELEMENTWISE    = 1 << 1;
    private static final int OP_TRAIT_TERNARY_ELEMENTWISE   = 1 << 2;
    private static final int OP_TRAIT_REDUCTION             = 1 << 3;
    private static final int OP_TRAIT_NORMALIZATION         = 1 << 4;
    private static final int OP_TRAIT_MATMUL                = 1 << 5;
    private static final int OP_TRAIT_FULLY_WRITING         = 1 << 6;
    private static final int OP_TRAIT_VIEW_PRODUCING        = 1 << 7;
    private static final int OP_TRAIT_VALUE_DEPENDENT_SHAPE = 1 << 8;
    private static final int OP_TRAIT_DATA_DEPENDENT        = 1 << 9;
    private static final int OP_TRAIT_SHAPE_ONLY_OUTPUT     = 1 << 10;
    private static final int OP_TRAIT_ACTIVATION            = 1 << 11;
    private static final int OP_TRAIT_COMPARISON            = 1 << 12;
    private static final int OP_TRAIT_LOGICAL               = 1 << 13;
    private static final int OP_TRAIT_IDENTITY              = 1 << 14;
    private static final int OP_TRAIT_DATA_MOVEMENT         = 1 << 15;
    private static final int OP_TRAIT_CONSTANT_GENERATION   = 1 << 16;
    private static final int OP_TRAIT_ATTENTION             = 1 << 17;
    private static final int OP_TRAIT_GATHER                = 1 << 18;
    private static final int OP_TRAIT_GATHER_ND             = 1 << 19;
    private static final int OP_TRAIT_CONCAT                = 1 << 20;
    private static final int OP_TRAIT_SPLIT                 = 1 << 21;
    private static final int OP_TRAIT_SPLIT_V               = 1 << 22;
    private static final int OP_TRAIT_STACK                 = 1 << 23;
    private static final int OP_TRAIT_SLICE                 = 1 << 24;
    private static final int OP_TRAIT_TILE                  = 1 << 25;
    private static final int OP_TRAIT_SCATTER_ND            = 1 << 26;
    private static final int OP_TRAIT_SCATTER_ND_UPDATE     = 1 << 27;
    private static final int OP_TRAIT_CAST                  = 1 << 28;

    // ── Native-op trait query ────────────────────────────────────────────────────
    private static int traits(String opName) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits(opName);
    }

    private static boolean has(int bits, int bit) {
        return (bits & bit) == bit;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 1. Trait-bit enumeration — canonical representative per trait bit
    // ────────────────────────────────────────────────────────────────────────────
    static Stream<Arguments> traitBitReps() {
        return Stream.of(
                Arguments.of("UNARY_ELEMENTWISE",     "abs",              OP_TRAIT_UNARY_ELEMENTWISE),
                Arguments.of("BINARY_ELEMENTWISE",    "add",              OP_TRAIT_BINARY_ELEMENTWISE),
                Arguments.of("TERNARY_ELEMENTWISE",   "select",           OP_TRAIT_TERNARY_ELEMENTWISE),
                Arguments.of("REDUCTION",             "reduce_sum",       OP_TRAIT_REDUCTION),
                Arguments.of("NORMALIZATION",         "softmax",          OP_TRAIT_NORMALIZATION),
                Arguments.of("MATMUL",                "matmul",           OP_TRAIT_MATMUL),
                Arguments.of("FULLY_WRITING",         "abs",              OP_TRAIT_FULLY_WRITING),
                Arguments.of("VIEW_PRODUCING",        "reshape",          OP_TRAIT_VIEW_PRODUCING),
                Arguments.of("VALUE_DEPENDENT_SHAPE", "fill",             OP_TRAIT_VALUE_DEPENDENT_SHAPE),
                Arguments.of("DATA_DEPENDENT",        "unique",           OP_TRAIT_DATA_DEPENDENT),
                Arguments.of("SHAPE_ONLY_OUTPUT",     "shape_of",         OP_TRAIT_SHAPE_ONLY_OUTPUT),
                Arguments.of("ACTIVATION",            "relu",             OP_TRAIT_ACTIVATION),
                Arguments.of("COMPARISON",            "equals",           OP_TRAIT_COMPARISON),
                Arguments.of("LOGICAL",               "boolean_and",      OP_TRAIT_LOGICAL),
                Arguments.of("IDENTITY",              "identity",         OP_TRAIT_IDENTITY),
                Arguments.of("DATA_MOVEMENT",         "gather",           OP_TRAIT_DATA_MOVEMENT),
                Arguments.of("CONSTANT_GENERATION",   "range",            OP_TRAIT_CONSTANT_GENERATION),
                Arguments.of("ATTENTION",             "flash_attention",  OP_TRAIT_ATTENTION),
                Arguments.of("GATHER",                "gather",           OP_TRAIT_GATHER),
                Arguments.of("GATHER_ND",             "gather_nd",        OP_TRAIT_GATHER_ND),
                Arguments.of("CONCAT",                "concat",           OP_TRAIT_CONCAT),
                Arguments.of("SPLIT",                 "split",            OP_TRAIT_SPLIT),
                Arguments.of("SPLIT_V",               "split_v",          OP_TRAIT_SPLIT_V),
                Arguments.of("STACK",                 "stack",            OP_TRAIT_STACK),
                Arguments.of("SLICE",                 "slice",            OP_TRAIT_SLICE),
                Arguments.of("TILE",                  "tile",             OP_TRAIT_TILE),
                Arguments.of("SCATTER_ND",            "scatter_nd",       OP_TRAIT_SCATTER_ND),
                Arguments.of("SCATTER_ND_UPDATE",     "scatter_nd_update", OP_TRAIT_SCATTER_ND_UPDATE),
                Arguments.of("CAST",                  "cast",             OP_TRAIT_CAST)
        );
    }

    @ParameterizedTest(name = "{0} represented by {1}")
    @MethodSource("traitBitReps")
    @DisplayName("every OpTraits bit has at least one representative op carrying it")
    public void testEveryTraitBitHasRepresentative(String bitName, String opName, int bit) {
        int t = traits(opName);
        assertNotEquals(0, t, "representative op '" + opName + "' returned 0 traits — "
                + "either the op is unregistered or OpTraitTable is missing an entry");
        assertTrue(has(t, bit),
                "op '" + opName + "' should carry " + bitName + " (0x"
                        + Integer.toHexString(bit) + ") but returned 0x" + Integer.toHexString(t));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 2. Full OpTraitTable entries — every op listed in the C++ table
    //    Each entry asserts the declared shorthand-expanded trait subset.
    //    Auto-derived bits from the class hierarchy may add more — we only check
    //    that ALL declared bits are set.
    // ────────────────────────────────────────────────────────────────────────────

    /** Bundle: op name + the traits the C++ TABLE declares for it. */
    static Stream<Arguments> fullTraitTableEntries() {
        // Mirrors libnd4j/include/ops/impl/OpTraitTable.cpp TABLE map.
        // These constants mirror the static shorthands in that file.
        final int UNARY_EW          = OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
        final int UNARY_ACT         = UNARY_EW | OP_TRAIT_ACTIVATION;
        final int BINARY_EW         = OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
        final int BINARY_CMP        = BINARY_EW | OP_TRAIT_COMPARISON;
        final int BINARY_LOG        = BINARY_EW | OP_TRAIT_LOGICAL;
        final int TERNARY_EW        = OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
        final int REDUCE            = OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING;
        final int NORM              = OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING;
        final int MATMUL            = OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING;
        final int VIEW_SHAPE_DEP    = OP_TRAIT_VIEW_PRODUCING;
        final int VIEW              = OP_TRAIT_VIEW_PRODUCING | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        final int VALDEP            = OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        final int DATADEP           = OP_TRAIT_DATA_DEPENDENT;
        final int SHAPE_ONLY        = OP_TRAIT_SHAPE_ONLY_OUTPUT;
        final int IDENT             = OP_TRAIT_IDENTITY | OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
        final int DATA_MOVE         = OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING;
        final int DATA_MOVE_VALDEP  = DATA_MOVE | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        final int CONST_GEN         = OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING;
        final int CONST_GEN_VALDEP  = CONST_GEN | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
        final int ATTN              = OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING;
        final int GATHER            = DATA_MOVE | OP_TRAIT_GATHER;
        final int GATHER_ND         = DATA_MOVE | OP_TRAIT_GATHER_ND;
        final int CONCAT            = DATA_MOVE | OP_TRAIT_CONCAT;
        final int SPLIT             = DATA_MOVE | OP_TRAIT_SPLIT;
        final int SPLIT_V           = DATA_MOVE | OP_TRAIT_SPLIT_V;
        final int STACK             = DATA_MOVE | OP_TRAIT_STACK;
        final int SLICE             = DATA_MOVE_VALDEP | OP_TRAIT_SLICE;
        final int TILE              = DATA_MOVE_VALDEP | OP_TRAIT_TILE;
        final int SCATTER_PARTIAL   = OP_TRAIT_DATA_MOVEMENT;
        final int SCATTER_ND        = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND;
        // scatter_nd_update does output->assign(input) → full writer, not partial
        final int SCATTER_ND_UPDATE = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND_UPDATE | OP_TRAIT_FULLY_WRITING;

        return Stream.of(
                // ── Unary elementwise math ─────────────────────────────
                Arguments.of("abs", UNARY_EW),
                Arguments.of("neg", UNARY_EW),
                Arguments.of("exp", UNARY_EW),
                Arguments.of("log", UNARY_EW),
                Arguments.of("log1p", UNARY_EW),
                Arguments.of("sqrt", UNARY_EW),
                Arguments.of("rsqrt", UNARY_EW),
                Arguments.of("square", UNARY_EW),
                Arguments.of("reciprocal", UNARY_EW),
                Arguments.of("ceil", UNARY_EW),
                Arguments.of("floor", UNARY_EW),
                Arguments.of("round", UNARY_EW),
                Arguments.of("sign", UNARY_EW),
                Arguments.of("erf", UNARY_EW),
                Arguments.of("erfc", UNARY_EW),
                Arguments.of("sin", UNARY_EW),
                Arguments.of("cos", UNARY_EW),
                Arguments.of("asin", UNARY_EW),
                Arguments.of("acos", UNARY_EW),
                Arguments.of("atan", UNARY_EW),
                Arguments.of("sinh", UNARY_EW),
                Arguments.of("cosh", UNARY_EW),
                Arguments.of("asinh", UNARY_EW),
                Arguments.of("acosh", UNARY_EW),
                Arguments.of("atanh", UNARY_EW),

                // ── Activations ────────────────────────────────────────
                Arguments.of("relu", UNARY_ACT),
                Arguments.of("relu6", UNARY_ACT),
                Arguments.of("leakyrelu", UNARY_ACT),
                Arguments.of("elu", UNARY_ACT),
                Arguments.of("selu", UNARY_ACT),
                Arguments.of("gelu", UNARY_ACT),
                Arguments.of("sigmoid", UNARY_ACT),
                Arguments.of("tanh", UNARY_ACT),
                Arguments.of("softsign", UNARY_ACT),
                Arguments.of("softplus", UNARY_ACT),
                Arguments.of("swish", UNARY_ACT),
                Arguments.of("silu", UNARY_ACT),
                Arguments.of("mish", UNARY_ACT),
                Arguments.of("hard_sigmoid", UNARY_ACT),
                Arguments.of("hardtanh", UNARY_ACT),
                Arguments.of("fused_gelu", UNARY_ACT),

                // ── Logical unary ──────────────────────────────────────
                Arguments.of("boolean_not", UNARY_EW | OP_TRAIT_LOGICAL),
                Arguments.of("logical_not", UNARY_EW | OP_TRAIT_LOGICAL),

                // ── Identity/cast/clip ─────────────────────────────────
                Arguments.of("identity", IDENT),
                Arguments.of("assign", IDENT),
                Arguments.of("cast", UNARY_EW | OP_TRAIT_CAST),
                Arguments.of("clipbyvalue", UNARY_EW),

                // ── Binary elementwise ─────────────────────────────────
                Arguments.of("add", BINARY_EW),
                Arguments.of("subtract", BINARY_EW),
                Arguments.of("multiply", BINARY_EW),
                Arguments.of("divide", BINARY_EW),
                Arguments.of("floormod", BINARY_EW),
                Arguments.of("floordiv", BINARY_EW),
                Arguments.of("reversedivide", BINARY_EW),
                Arguments.of("reversesubtract", BINARY_EW),
                Arguments.of("squaredsubtract", BINARY_EW),
                Arguments.of("add_scalar", BINARY_EW),
                Arguments.of("subtract_scalar", BINARY_EW),
                Arguments.of("sub_scalar", BINARY_EW),
                Arguments.of("multiply_scalar", BINARY_EW),
                Arguments.of("mul_scalar", BINARY_EW),
                Arguments.of("divide_scalar", BINARY_EW),
                Arguments.of("div_scalar", BINARY_EW),
                Arguments.of("rsub_scalar", BINARY_EW),
                Arguments.of("rdiv_scalar", BINARY_EW),
                Arguments.of("pow", BINARY_EW),
                Arguments.of("min_pairwise", BINARY_EW),
                Arguments.of("max_pairwise", BINARY_EW),
                Arguments.of("atan2", BINARY_EW),
                Arguments.of("maximum", BINARY_EW),
                Arguments.of("minimum", BINARY_EW),
                Arguments.of("mod", BINARY_EW),
                Arguments.of("multiply_no_nan", BINARY_EW),
                Arguments.of("swish_mul", BINARY_EW),

                // ── Binary comparison ──────────────────────────────────
                Arguments.of("equals", BINARY_CMP),
                Arguments.of("not_equals", BINARY_CMP),
                Arguments.of("less", BINARY_CMP),
                Arguments.of("less_equal", BINARY_CMP),
                Arguments.of("greater", BINARY_CMP),
                Arguments.of("greater_equal", BINARY_CMP),

                // ── Binary logical ─────────────────────────────────────
                Arguments.of("boolean_and", BINARY_LOG),
                Arguments.of("boolean_or", BINARY_LOG),
                Arguments.of("boolean_xor", BINARY_LOG),
                Arguments.of("logical_and", BINARY_LOG),
                Arguments.of("logical_or", BINARY_LOG),

                // ── Ternary elementwise ────────────────────────────────
                Arguments.of("where", TERNARY_EW | OP_TRAIT_DATA_DEPENDENT),
                Arguments.of("select", TERNARY_EW),

                // ── Matrix ops ─────────────────────────────────────────
                Arguments.of("matmul", MATMUL),
                Arguments.of("mmul", MATMUL),
                Arguments.of("batched_gemm", MATMUL),
                Arguments.of("tensormmul", MATMUL),
                Arguments.of("fp8_matmul", MATMUL),
                Arguments.of("smooth_quant", MATMUL),
                Arguments.of("awq_matmul", MATMUL),
                Arguments.of("column_parallel_linear", MATMUL),
                Arguments.of("row_parallel_linear", MATMUL),
                Arguments.of("multi_lora_matmul", MATMUL),
                Arguments.of("fused_gemm_swiglu", MATMUL),

                // ── Reductions ─────────────────────────────────────────
                Arguments.of("reduce_sum", REDUCE),
                Arguments.of("reduce_mean", REDUCE),
                Arguments.of("reduce_max", REDUCE),
                Arguments.of("reduce_min", REDUCE),
                Arguments.of("reduce_prod", REDUCE),
                Arguments.of("reduce_norm1", REDUCE),
                Arguments.of("reduce_norm2", REDUCE),
                Arguments.of("reduce_logsumexp", REDUCE),
                Arguments.of("reduce_variance", REDUCE),
                Arguments.of("reduce_stdev", REDUCE),
                Arguments.of("sum", REDUCE),
                Arguments.of("mean", REDUCE),
                Arguments.of("max", REDUCE),
                Arguments.of("min", REDUCE),
                Arguments.of("prod", REDUCE),
                Arguments.of("norm1", REDUCE),
                Arguments.of("norm2", REDUCE),
                Arguments.of("normmax", REDUCE),
                Arguments.of("argmax", REDUCE),
                Arguments.of("argmin", REDUCE),

                // ── Normalization ──────────────────────────────────────
                Arguments.of("softmax", NORM),
                Arguments.of("log_softmax", NORM),
                Arguments.of("layer_norm", NORM),
                Arguments.of("fused_layer_norm", NORM),
                Arguments.of("batch_norm", NORM),
                Arguments.of("batchnorm", NORM),
                Arguments.of("rms_norm", NORM),
                Arguments.of("rms_norm_linear", NORM),
                Arguments.of("normalize_moments", NORM),
                Arguments.of("fused_rope", NORM),

                // ── Attention ──────────────────────────────────────────
                Arguments.of("onnx_multi_head_attention", ATTN),
                Arguments.of("dot_product_attention_v2", ATTN),
                Arguments.of("flash_attention", ATTN),
                Arguments.of("multi_head_dot_product_attention", ATTN),
                Arguments.of("multi_head_attention", ATTN),

                // ── Token sampling ─────────────────────────────────────
                Arguments.of("token_sample", OP_TRAIT_FULLY_WRITING),

                // ── View / view-shape-dep ──────────────────────────────
                Arguments.of("reshape", VIEW),
                Arguments.of("reshape_no_copy", VIEW),
                Arguments.of("strided_slice", VIEW | OP_TRAIT_SLICE),
                Arguments.of("expand_dims", VIEW_SHAPE_DEP),
                Arguments.of("squeeze", VIEW_SHAPE_DEP),
                Arguments.of("flatten", VIEW_SHAPE_DEP),
                Arguments.of("flatten_2d", VIEW_SHAPE_DEP),
                Arguments.of("permute", VIEW_SHAPE_DEP),

                // ── Shape-determined data movement ────────────────────
                Arguments.of("gather", GATHER),
                Arguments.of("gather_nd", GATHER_ND),
                Arguments.of("concat", CONCAT),
                Arguments.of("stack", STACK),
                Arguments.of("split", SPLIT),
                Arguments.of("split_v", SPLIT_V),
                Arguments.of("repeat", DATA_MOVE),

                // ── Value-dependent data movement ─────────────────────
                Arguments.of("slice", SLICE),
                Arguments.of("tile", TILE),
                Arguments.of("pad", DATA_MOVE_VALDEP),
                Arguments.of("fill", DATA_MOVE_VALDEP),
                Arguments.of("broadcast_to", DATA_MOVE_VALDEP),
                Arguments.of("scatter_nd", SCATTER_ND),
                Arguments.of("scatter_nd_update", SCATTER_ND_UPDATE),
                Arguments.of("range", CONST_GEN_VALDEP),
                Arguments.of("linspace", CONST_GEN_VALDEP),
                Arguments.of("create", CONST_GEN_VALDEP),

                // ── Data-dependent ops ─────────────────────────────────
                Arguments.of("unique", DATADEP),
                Arguments.of("non_max_suppression", DATADEP),
                Arguments.of("non_max_suppression_v3", DATADEP),

                // ── Shape-only output ─────────────────────────────────
                Arguments.of("shape_of", SHAPE_ONLY | CONST_GEN),
                Arguments.of("size_at", SHAPE_ONLY | CONST_GEN),
                Arguments.of("rank", SHAPE_ONLY | CONST_GEN),
                Arguments.of("zeros_like", SHAPE_ONLY | CONST_GEN),
                Arguments.of("zeros_as", SHAPE_ONLY | CONST_GEN),
                Arguments.of("zeroslike", SHAPE_ONLY | CONST_GEN),
                Arguments.of("ones_like", SHAPE_ONLY | CONST_GEN),
                Arguments.of("ones_as", SHAPE_ONLY | CONST_GEN),
                Arguments.of("oneslike", SHAPE_ONLY | CONST_GEN),

                // ── LLM attention (fwd + bp) ──────────────────────────
                Arguments.of("dot_product_attention", ATTN),
                Arguments.of("dot_product_attention_bp", ATTN),
                Arguments.of("dot_product_attention_v2_bp", ATTN),
                Arguments.of("multi_head_dot_product_attention_bp", ATTN),
                Arguments.of("flash_attention_bp", ATTN),
                Arguments.of("grouped_query_attention", ATTN),
                Arguments.of("grouped_query_attention_bp", ATTN),
                Arguments.of("sliding_window_attention", ATTN),
                Arguments.of("shared_kv_attention", ATTN),
                Arguments.of("windowed_attention", ATTN),
                Arguments.of("paged_attention_forward", ATTN),
                Arguments.of("turbo_quant_attention", ATTN),
                Arguments.of("two_way_cross_attention", ATTN),
                Arguments.of("two_way_cross_attention_bp", ATTN),
                Arguments.of("vlm_cross_attention", ATTN),
                Arguments.of("apply_alibi", ATTN),
                Arguments.of("relative_position_bias", ATTN),

                // ── KV cache ──────────────────────────────────────────
                Arguments.of("kv_cache_update", DATA_MOVE_VALDEP),
                Arguments.of("kv_cache_quantize", CONST_GEN_VALDEP),
                Arguments.of("kv_cache_dequantize", UNARY_EW),
                Arguments.of("paged_kv_append", DATA_MOVE_VALDEP),

                // ── Rotary / positional embedding ─────────────────────
                Arguments.of("rope", NORM),
                Arguments.of("rope_bp", NORM),
                Arguments.of("fused_rope_bp", NORM),
                Arguments.of("dual_rope", NORM),

                // ── Normalization backprop / fused variants ───────────
                Arguments.of("rms_norm_bp", NORM),
                Arguments.of("rms_norm_linear_bp", NORM),
                Arguments.of("fused_layer_norm_bp", NORM),
                Arguments.of("fused_rms_norm_swiglu", NORM),
                Arguments.of("fused_rms_norm_swiglu_bp", NORM),

                // ── Fused GEMM / SwiGLU ───────────────────────────────
                Arguments.of("fused_gemm_swiglu_bp", MATMUL),

                // ── Activation backprop + novel activations ───────────
                Arguments.of("silu_bp", UNARY_ACT),
                Arguments.of("fused_gelu_bp", UNARY_ACT),
                Arguments.of("squared_relu", UNARY_ACT),
                Arguments.of("squared_relu_bp", UNARY_ACT),
                Arguments.of("gated_delta_rule", UNARY_ACT),

                // ── Mamba / selective scan / SSM / causal conv ────────
                Arguments.of("gated_delta_net_block", REDUCE),
                Arguments.of("selective_scan", REDUCE),
                Arguments.of("mamba2_ssm", REDUCE),
                Arguments.of("causal_conv1d", UNARY_EW),

                // ── Fused training kernels ────────────────────────────
                Arguments.of("fused_bias_dropout_residual", UNARY_EW),
                Arguments.of("fused_elementwise_chain", UNARY_EW),
                Arguments.of("swish_mul_bp", BINARY_EW),
                Arguments.of("center_and_sharpen", UNARY_EW),
                Arguments.of("center_and_sharpen_bp", UNARY_EW),
                Arguments.of("ema_update", DATA_MOVE),
                Arguments.of("ema_update_bp", DATA_MOVE),

                // ── Quantization / adapter matmuls ────────────────────
                Arguments.of("quantized_matmul", MATMUL),
                Arguments.of("dora_matmul", MATMUL),
                Arguments.of("dora_matmul_bp", MATMUL),
                Arguments.of("lora_matmul", MATMUL),
                Arguments.of("lora_matmul_bp", MATMUL),
                Arguments.of("loha_matmul", MATMUL),
                Arguments.of("loha_matmul_bp", MATMUL),
                Arguments.of("lokr_matmul", MATMUL),
                Arguments.of("lokr_matmul_bp", MATMUL),

                // ── GGML / per-layer embedding / misc ─────────────────
                Arguments.of("ggml_dequantize", UNARY_EW),
                Arguments.of("per_layer_embedding", DATA_MOVE_VALDEP)
        );
    }

    @ParameterizedTest(name = "{0} must declare 0x{1}")
    @MethodSource("fullTraitTableEntries")
    @DisplayName("every OpTraitTable entry declares at least its expected trait subset")
    public void testDeclaredTraitSubset(String opName, int expected) {
        int actual = traits(opName);
        assertNotEquals(0, actual,
                "op '" + opName + "' is missing from the trait table (getOpTraits returned 0)");
        assertEquals(expected, actual & expected,
                () -> "op '" + opName + "' is missing declared trait bits. "
                        + "expected ⊆ actual — expected=0x" + Integer.toHexString(expected)
                        + " actual=0x" + Integer.toHexString(actual)
                        + " missing=0x" + Integer.toHexString(expected & ~actual));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 3. Regression — gather/gather_nd/repeat must NOT be VALUE_DEPENDENT_SHAPE
    //    (the SHAPES_FROZEN LIFECYCLE_ERROR regression we just fixed)
    // ────────────────────────────────────────────────────────────────────────────

    static List<String> nonValdepOps() {
        return Arrays.asList(
                "gather", "gather_nd", "repeat",
                "concat", "stack", "split", "split_v",
                "expand_dims", "squeeze",
                "flatten", "flatten_2d", "permute");
    }

    @ParameterizedTest(name = "{0} is NOT value-dependent")
    @MethodSource("nonValdepOps")
    @DisplayName("regression: shape-determined ops must NOT carry OP_TRAIT_VALUE_DEPENDENT_SHAPE")
    public void testShapeDeterminedOpsAreNotValdep(String opName) {
        int t = traits(opName);
        assertNotEquals(0, t, "op '" + opName + "' is missing from the trait table");
        assertFalse(has(t, OP_TRAIT_VALUE_DEPENDENT_SHAPE),
                "op '" + opName + "' MUST NOT carry OP_TRAIT_VALUE_DEPENDENT_SHAPE. "
                        + "Its output shape comes from input shapes + iArgs only; tagging it as "
                        + "value-dep causes SHAPES_FROZEN LIFECYCLE_ERROR false positives "
                        + "when input shapes change. actual=0x" + Integer.toHexString(t));
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 4. Positive: ops that SHOULD be VALUE_DEPENDENT_SHAPE
    // ────────────────────────────────────────────────────────────────────────────

    static List<String> valdepOps() {
        // NOTE: scatter_nd / scatter_nd_update are NOT in this list — the C++ table
        // defines them as SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND (no VALDEP). Their
        // output shape equals the `shape` input values ONLY when that arg is a
        // tensor, but the op's first positional input carries the output-shape
        // template, not tensor data. Keeping them value-dep would cause the
        // same SHAPES_FROZEN false positive we are guarding against.
        return Arrays.asList(
                "fill", "broadcast_to", "range", "linspace",
                "pad", "reshape", "reshape_no_copy", "strided_slice",
                "create",
                "slice", "tile",
                "kv_cache_update", "kv_cache_quantize", "paged_kv_append",
                "per_layer_embedding");
    }

    @ParameterizedTest(name = "{0} IS value-dependent")
    @MethodSource("valdepOps")
    @DisplayName("value-dependent ops MUST carry OP_TRAIT_VALUE_DEPENDENT_SHAPE")
    public void testValdepOpsCarryValdep(String opName) {
        int t = traits(opName);
        assertNotEquals(0, t, "op '" + opName + "' is missing from the trait table");
        assertTrue(has(t, OP_TRAIT_VALUE_DEPENDENT_SHAPE),
                "op '" + opName + "' MUST carry OP_TRAIT_VALUE_DEPENDENT_SHAPE. "
                        + "Its output shape depends on input TENSOR data. actual=0x" + Integer.toHexString(t));
    }

    @ParameterizedTest(name = "data-dependent op: {0}")
    @ValueSource(strings = {"unique", "non_max_suppression", "non_max_suppression_v3"})
    @DisplayName("unique / NMS must be DATA_DEPENDENT (variable-length output)")
    public void testDataDependentOps(String opName) {
        int t = traits(opName);
        assertTrue(has(t, OP_TRAIT_DATA_DEPENDENT),
                "op '" + opName + "' must carry OP_TRAIT_DATA_DEPENDENT");
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 5. Unknown op → 0, no throw
    // ────────────────────────────────────────────────────────────────────────────
    @Test
    @DisplayName("getOpTraits(unknown) returns 0 and does not throw")
    public void testUnknownOpReturnsZero() {
        int t = traits("this_op_does_not_exist_xyz_abc_def");
        assertEquals(0, t, "unknown op should return 0 traits (got 0x"
                + Integer.toHexString(t) + ")");
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 6. Trait combinators — pairs of bits that MUST co-occur
    // ────────────────────────────────────────────────────────────────────────────
    @Test
    @DisplayName("every OP_TRAIT_GATHER op also carries OP_TRAIT_DATA_MOVEMENT")
    public void testGatherImpliesDataMovement() {
        for (String op : new String[]{"gather"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_GATHER),    op + " should have GATHER");
            assertTrue(has(t, OP_TRAIT_DATA_MOVEMENT), op + " should have DATA_MOVEMENT");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING), op + " should have FULLY_WRITING");
        }
        for (String op : new String[]{"gather_nd"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_GATHER_ND), op + " should have GATHER_ND");
            assertTrue(has(t, OP_TRAIT_DATA_MOVEMENT), op + " should have DATA_MOVEMENT");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING), op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("every attention op carries OP_TRAIT_FULLY_WRITING")
    public void testAttentionImpliesFullyWriting() {
        String[] ops = {
                "onnx_multi_head_attention", "dot_product_attention_v2", "flash_attention",
                "multi_head_dot_product_attention", "multi_head_attention",
                "dot_product_attention", "dot_product_attention_bp", "dot_product_attention_v2_bp",
                "multi_head_dot_product_attention_bp", "flash_attention_bp",
                "grouped_query_attention", "grouped_query_attention_bp",
                "sliding_window_attention", "shared_kv_attention", "windowed_attention",
                "paged_attention_forward", "turbo_quant_attention",
                "two_way_cross_attention", "two_way_cross_attention_bp",
                "vlm_cross_attention", "apply_alibi", "relative_position_bias"
        };
        for (String op : ops) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_ATTENTION), op + " should have ATTENTION");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING), op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("every activation op carries OP_TRAIT_UNARY_ELEMENTWISE + FULLY_WRITING")
    public void testActivationImpliesUnary() {
        String[] ops = {
                "relu", "relu6", "leakyrelu", "elu", "selu", "gelu", "sigmoid", "tanh",
                "softsign", "softplus", "swish", "silu", "mish", "hard_sigmoid",
                "hardtanh", "fused_gelu", "silu_bp", "fused_gelu_bp", "squared_relu",
                "squared_relu_bp", "gated_delta_rule"
        };
        for (String op : ops) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_ACTIVATION), op + " should have ACTIVATION");
            assertTrue(has(t, OP_TRAIT_UNARY_ELEMENTWISE), op + " should have UNARY_ELEMENTWISE");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING), op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("every matmul op carries OP_TRAIT_FULLY_WRITING")
    public void testMatmulImpliesFullyWriting() {
        String[] ops = {
                "matmul", "mmul", "batched_gemm", "tensormmul", "fp8_matmul", "smooth_quant",
                "awq_matmul", "column_parallel_linear", "row_parallel_linear",
                "multi_lora_matmul", "fused_gemm_swiglu", "fused_gemm_swiglu_bp",
                "quantized_matmul", "dora_matmul", "dora_matmul_bp",
                "lora_matmul", "lora_matmul_bp", "loha_matmul", "loha_matmul_bp",
                "lokr_matmul", "lokr_matmul_bp"
        };
        for (String op : ops) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_MATMUL), op + " should have MATMUL");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING), op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("every scatter_nd* op carries OP_TRAIT_DATA_MOVEMENT")
    public void testScatterImpliesDataMovement() {
        int a = traits("scatter_nd");
        int b = traits("scatter_nd_update");
        assertTrue(has(a, OP_TRAIT_SCATTER_ND), "scatter_nd should have SCATTER_ND");
        assertTrue(has(a, OP_TRAIT_DATA_MOVEMENT), "scatter_nd should have DATA_MOVEMENT");
        assertTrue(has(b, OP_TRAIT_SCATTER_ND_UPDATE), "scatter_nd_update should have SCATTER_ND_UPDATE");
        assertTrue(has(b, OP_TRAIT_DATA_MOVEMENT), "scatter_nd_update should have DATA_MOVEMENT");
        // scatter_nd_update does output->assign(input) which fully writes the output
        // before scattering updates. This makes prezero redundant.
        assertTrue(has(b, OP_TRAIT_FULLY_WRITING),
                "scatter_nd_update should have FULLY_WRITING (assign copies entire input to output)");
        // scatter_nd zeroes output then scatters — it is a true partial writer, no FULLY_WRITING.
        assertFalse(has(a, OP_TRAIT_FULLY_WRITING),
                "scatter_nd should NOT have FULLY_WRITING (true partial writer)");
    }

    @Test
    @DisplayName("comparison binary ops carry BINARY_ELEMENTWISE + FULLY_WRITING")
    public void testComparisonImpliesBinary() {
        for (String op : new String[]{"equals", "not_equals", "less", "less_equal", "greater", "greater_equal"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_COMPARISON),         op + " should have COMPARISON");
            assertTrue(has(t, OP_TRAIT_BINARY_ELEMENTWISE), op + " should have BINARY_ELEMENTWISE");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING),      op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("identity ops carry IDENTITY + UNARY_ELEMENTWISE + FULLY_WRITING")
    public void testIdentityImpliesUnary() {
        for (String op : new String[]{"identity", "assign"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_IDENTITY),           op + " should have IDENTITY");
            assertTrue(has(t, OP_TRAIT_UNARY_ELEMENTWISE),  op + " should have UNARY_ELEMENTWISE");
            assertTrue(has(t, OP_TRAIT_FULLY_WRITING),      op + " should have FULLY_WRITING");
        }
    }

    @Test
    @DisplayName("shape-only ops carry CONSTANT_GENERATION")
    public void testShapeOnlyImpliesConstGen() {
        for (String op : new String[]{"shape_of", "size_at", "rank", "zeros_like", "ones_like"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_SHAPE_ONLY_OUTPUT),
                    op + " should have SHAPE_ONLY_OUTPUT");
            assertTrue(has(t, OP_TRAIT_CONSTANT_GENERATION),
                    op + " should have CONSTANT_GENERATION");
        }
    }

    @Test
    @DisplayName("VIEW ops (reshape/reshape_no_copy/strided_slice) carry VIEW_PRODUCING + VALUE_DEPENDENT_SHAPE")
    public void testViewOpsCarryBothBits() {
        for (String op : new String[]{"reshape", "reshape_no_copy", "strided_slice"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_VIEW_PRODUCING),
                    op + " should have VIEW_PRODUCING");
            assertTrue(has(t, OP_TRAIT_VALUE_DEPENDENT_SHAPE),
                    op + " should have VALUE_DEPENDENT_SHAPE");
        }
    }

    @Test
    @DisplayName("VIEW_SHAPE_DEP ops carry VIEW_PRODUCING but NOT VALUE_DEPENDENT_SHAPE")
    public void testShapeDepViewsDoNotCarryValdep() {
        for (String op : new String[]{"expand_dims", "squeeze", "flatten", "flatten_2d", "permute"}) {
            int t = traits(op);
            assertTrue(has(t, OP_TRAIT_VIEW_PRODUCING),
                    op + " should have VIEW_PRODUCING");
            assertFalse(has(t, OP_TRAIT_VALUE_DEPENDENT_SHAPE),
                    op + " must NOT have VALUE_DEPENDENT_SHAPE "
                            + "(shape is derived from input shapes + iArgs, no tensor data read)");
        }
    }
}
