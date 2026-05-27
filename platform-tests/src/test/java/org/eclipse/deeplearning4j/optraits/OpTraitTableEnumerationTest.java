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
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.Test;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * Exhaustive enumeration test over the libnd4j {@code OpTraitTable.cpp} TABLE map.
 *
 * <p>Every entry in the C++ trait table is mirrored here as {@code (opName →
 * expectedTraits)}. The {@code @TestFactory} generates one dynamic test per op
 * so a failure points at the specific offender. The assertion is a SUBSET check
 * ({@code (expected & actual) == expected}) — the C++ class hierarchy may add
 * extra bits beyond what the TABLE entry declares, but it must not drop any.
 *
 * <p>This is the broadest regression net. Any op added to the C++ table should
 * also be added here — that way a commit that introduces a trait-table entry
 * without Java-side coverage is visible in code review.
 */
@DisplayName("libnd4j OpTraitTable — full enumeration (one dynamic test per op)")
public class OpTraitTableEnumerationTest {

    // ── Trait bits — mirror OpDescriptor.h ─────────────────────────────────────
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

    // Shorthands (mirror OpTraitTable.cpp static constexpr shorthands)
    private static final int UNARY_EW          = OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
    private static final int UNARY_ACT         = UNARY_EW | OP_TRAIT_ACTIVATION;
    private static final int BINARY_EW         = OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
    private static final int BINARY_CMP        = BINARY_EW | OP_TRAIT_COMPARISON;
    private static final int BINARY_LOG        = BINARY_EW | OP_TRAIT_LOGICAL;
    private static final int TERNARY_EW        = OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
    private static final int REDUCE            = OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING;
    private static final int NORM              = OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING;
    private static final int MATMUL            = OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING;
    private static final int VIEW_SHAPE_DEP    = OP_TRAIT_VIEW_PRODUCING;
    private static final int VIEW              = OP_TRAIT_VIEW_PRODUCING | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
    private static final int DATADEP           = OP_TRAIT_DATA_DEPENDENT;
    private static final int SHAPE_ONLY        = OP_TRAIT_SHAPE_ONLY_OUTPUT;
    private static final int IDENT             = OP_TRAIT_IDENTITY | OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
    private static final int DATA_MOVE         = OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING;
    private static final int DATA_MOVE_VALDEP  = DATA_MOVE | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
    private static final int CONST_GEN         = OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING;
    private static final int CONST_GEN_VALDEP  = CONST_GEN | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
    private static final int ATTN              = OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING;
    private static final int GATHER            = DATA_MOVE | OP_TRAIT_GATHER;
    private static final int GATHER_ND         = DATA_MOVE | OP_TRAIT_GATHER_ND;
    private static final int CONCAT            = DATA_MOVE | OP_TRAIT_CONCAT;
    private static final int SPLIT             = DATA_MOVE | OP_TRAIT_SPLIT;
    private static final int SPLIT_V           = DATA_MOVE | OP_TRAIT_SPLIT_V;
    private static final int STACK             = DATA_MOVE | OP_TRAIT_STACK;
    private static final int SLICE             = DATA_MOVE_VALDEP | OP_TRAIT_SLICE;
    private static final int TILE              = DATA_MOVE_VALDEP | OP_TRAIT_TILE;
    private static final int SCATTER_PARTIAL   = OP_TRAIT_DATA_MOVEMENT;
    private static final int SCATTER_ND        = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND;
    // scatter_nd_update does output->assign(input) → full writer, not partial
    private static final int SCATTER_ND_UPDATE = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND_UPDATE | OP_TRAIT_FULLY_WRITING;

    /** Java-side mirror of the C++ TABLE map, in the exact order of the source. */
    private static Map<String, Integer> buildExpectedTraits() {
        Map<String, Integer> m = new LinkedHashMap<>();

        // Unary elementwise math
        for (String op : new String[]{
                "abs", "neg", "exp", "log", "log1p", "sqrt", "rsqrt", "square",
                "reciprocal", "ceil", "floor", "round", "sign", "erf", "erfc",
                "sin", "cos", "asin", "acos", "atan", "sinh", "cosh",
                "asinh", "acosh", "atanh"}) {
            m.put(op, UNARY_EW);
        }

        // Activations
        for (String op : new String[]{
                "relu", "relu6", "leakyrelu", "elu", "selu", "gelu", "sigmoid",
                "tanh", "softsign", "softplus", "swish", "silu", "mish",
                "hard_sigmoid", "hardtanh", "fused_gelu"}) {
            m.put(op, UNARY_ACT);
        }

        // Unary logical
        m.put("boolean_not", UNARY_EW | OP_TRAIT_LOGICAL);
        m.put("logical_not", UNARY_EW | OP_TRAIT_LOGICAL);

        // Identity / assign / cast / clip
        m.put("identity", IDENT);
        m.put("assign", IDENT);
        m.put("cast", UNARY_EW | OP_TRAIT_CAST);
        m.put("clipbyvalue", UNARY_EW);

        // Binary elementwise
        for (String op : new String[]{
                "add", "subtract", "multiply", "divide", "floormod", "floordiv",
                "reversedivide", "reversesubtract", "squaredsubtract",
                "add_scalar", "subtract_scalar", "sub_scalar",
                "multiply_scalar", "mul_scalar", "divide_scalar", "div_scalar",
                "rsub_scalar", "rdiv_scalar",
                "pow", "min_pairwise", "max_pairwise", "atan2", "maximum", "minimum",
                "mod", "multiply_no_nan", "swish_mul"}) {
            m.put(op, BINARY_EW);
        }

        // Binary comparison
        for (String op : new String[]{
                "equals", "not_equals", "less", "less_equal", "greater", "greater_equal"}) {
            m.put(op, BINARY_CMP);
        }

        // Binary logical
        for (String op : new String[]{
                "boolean_and", "boolean_or", "boolean_xor", "logical_and", "logical_or"}) {
            m.put(op, BINARY_LOG);
        }

        // Ternary
        m.put("where", TERNARY_EW | OP_TRAIT_DATA_DEPENDENT);
        m.put("select", TERNARY_EW);

        // Matrix ops
        for (String op : new String[]{
                "matmul", "mmul", "batched_gemm", "tensormmul", "fp8_matmul",
                "smooth_quant", "awq_matmul", "column_parallel_linear",
                "row_parallel_linear", "multi_lora_matmul", "fused_gemm_swiglu"}) {
            m.put(op, MATMUL);
        }

        // Reductions
        for (String op : new String[]{
                "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
                "reduce_norm1", "reduce_norm2", "reduce_logsumexp", "reduce_variance",
                "reduce_stdev", "sum", "mean", "max", "min", "prod",
                "norm1", "norm2", "normmax", "argmax", "argmin"}) {
            m.put(op, REDUCE);
        }

        // Normalization
        for (String op : new String[]{
                "softmax", "log_softmax", "layer_norm", "fused_layer_norm",
                "batch_norm", "batchnorm", "rms_norm", "rms_norm_linear",
                "normalize_moments", "fused_rope"}) {
            m.put(op, NORM);
        }

        // Attention (base)
        for (String op : new String[]{
                "onnx_multi_head_attention", "dot_product_attention_v2",
                "flash_attention", "multi_head_dot_product_attention", "multi_head_attention"}) {
            m.put(op, ATTN);
        }

        // Token sampling
        m.put("token_sample", OP_TRAIT_FULLY_WRITING);

        // View-producing
        m.put("reshape", VIEW);
        m.put("reshape_no_copy", VIEW);
        m.put("strided_slice", VIEW | OP_TRAIT_SLICE);
        m.put("expand_dims", VIEW_SHAPE_DEP);
        m.put("squeeze", VIEW_SHAPE_DEP);
        m.put("flatten", VIEW_SHAPE_DEP);
        m.put("flatten_2d", VIEW_SHAPE_DEP);
        m.put("permute", VIEW_SHAPE_DEP);

        // Shape-determined data movement
        m.put("gather", GATHER);
        m.put("gather_nd", GATHER_ND);
        m.put("concat", CONCAT);
        m.put("stack", STACK);
        m.put("split", SPLIT);
        m.put("split_v", SPLIT_V);
        m.put("repeat", DATA_MOVE);

        // Value-dependent data movement
        m.put("slice", SLICE);
        m.put("tile", TILE);
        m.put("pad", DATA_MOVE_VALDEP);
        m.put("fill", DATA_MOVE_VALDEP);
        m.put("broadcast_to", DATA_MOVE_VALDEP);
        m.put("scatter_nd", SCATTER_ND);
        m.put("scatter_nd_update", SCATTER_ND_UPDATE);
        m.put("range", CONST_GEN_VALDEP);
        m.put("linspace", CONST_GEN_VALDEP);
        m.put("create", CONST_GEN_VALDEP);

        // Data-dependent
        m.put("unique", DATADEP);
        m.put("non_max_suppression", DATADEP);
        m.put("non_max_suppression_v3", DATADEP);

        // Shape-only
        for (String op : new String[]{
                "shape_of", "size_at", "rank",
                "zeros_like", "zeros_as", "zeroslike",
                "ones_like", "ones_as", "oneslike"}) {
            m.put(op, SHAPE_ONLY | CONST_GEN);
        }

        // LLM attention forward + backprop
        for (String op : new String[]{
                "dot_product_attention", "dot_product_attention_bp",
                "dot_product_attention_v2_bp", "multi_head_dot_product_attention_bp",
                "flash_attention_bp", "grouped_query_attention", "grouped_query_attention_bp",
                "sliding_window_attention", "shared_kv_attention", "windowed_attention",
                "paged_attention_forward", "turbo_quant_attention",
                "two_way_cross_attention", "two_way_cross_attention_bp",
                "vlm_cross_attention", "apply_alibi", "relative_position_bias"}) {
            m.put(op, ATTN);
        }

        // KV cache
        m.put("kv_cache_update", DATA_MOVE_VALDEP);
        m.put("kv_cache_quantize", CONST_GEN_VALDEP);
        m.put("kv_cache_dequantize", UNARY_EW);
        m.put("paged_kv_append", DATA_MOVE_VALDEP);

        // Rotary / positional embedding
        m.put("rope", NORM);
        m.put("rope_bp", NORM);
        m.put("fused_rope_bp", NORM);
        m.put("dual_rope", NORM);

        // Normalization backprop / fused variants
        m.put("rms_norm_bp", NORM);
        m.put("rms_norm_linear_bp", NORM);
        m.put("fused_layer_norm_bp", NORM);
        m.put("fused_rms_norm_swiglu", NORM);
        m.put("fused_rms_norm_swiglu_bp", NORM);

        // Fused GEMM / SwiGLU backprop
        m.put("fused_gemm_swiglu_bp", MATMUL);

        // Activation backprop + novel activations
        m.put("silu_bp", UNARY_ACT);
        m.put("fused_gelu_bp", UNARY_ACT);
        m.put("squared_relu", UNARY_ACT);
        m.put("squared_relu_bp", UNARY_ACT);
        m.put("gated_delta_rule", UNARY_ACT);

        // Mamba / selective scan / SSM / causal conv
        m.put("gated_delta_net_block", REDUCE);
        m.put("selective_scan", REDUCE);
        m.put("mamba2_ssm", REDUCE);
        m.put("causal_conv1d", UNARY_EW);

        // Fused training kernels
        m.put("fused_bias_dropout_residual", UNARY_EW);
        m.put("fused_elementwise_chain", UNARY_EW);
        m.put("swish_mul_bp", BINARY_EW);
        m.put("center_and_sharpen", UNARY_EW);
        m.put("center_and_sharpen_bp", UNARY_EW);
        m.put("ema_update", DATA_MOVE);
        m.put("ema_update_bp", DATA_MOVE);

        // Quantization / adapter matmuls
        m.put("quantized_matmul", MATMUL);
        m.put("dora_matmul", MATMUL);
        m.put("dora_matmul_bp", MATMUL);
        m.put("lora_matmul", MATMUL);
        m.put("lora_matmul_bp", MATMUL);
        m.put("loha_matmul", MATMUL);
        m.put("loha_matmul_bp", MATMUL);
        m.put("lokr_matmul", MATMUL);
        m.put("lokr_matmul_bp", MATMUL);

        // GGML / per-layer embedding / misc
        m.put("ggml_dequantize", UNARY_EW);
        m.put("per_layer_embedding", DATA_MOVE_VALDEP);

        return m;
    }

    private static final Map<String, Integer> EXPECTED_TRAITS = buildExpectedTraits();

    private static int traits(String opName) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getOpTraits(opName);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Dynamic per-op tests so failures name the exact offender
    // ────────────────────────────────────────────────────────────────────────

    @TestFactory
    @DisplayName("every op in OpTraitTable.cpp declares its expected trait subset")
    List<DynamicTest> perOpTraitSubsetTests() {
        List<DynamicTest> tests = new ArrayList<>(EXPECTED_TRAITS.size());
        for (Map.Entry<String, Integer> e : EXPECTED_TRAITS.entrySet()) {
            final String op = e.getKey();
            final int expected = e.getValue();
            tests.add(DynamicTest.dynamicTest(
                    "op=" + op + " expected=0x" + Integer.toHexString(expected),
                    () -> {
                        int actual = traits(op);
                        assertNotEquals(0, actual,
                                "op '" + op + "' is missing from the trait table "
                                        + "(getOpTraits returned 0) — either OpTraitTable.cpp "
                                        + "lost the entry or the op is unregistered");
                        int missing = expected & ~actual;
                        assertEquals(expected, actual & expected,
                                () -> "op '" + op + "' missing declared trait bits: "
                                        + "expected=0x" + Integer.toHexString(expected)
                                        + " actual=0x" + Integer.toHexString(actual)
                                        + " missing=0x" + Integer.toHexString(missing));
                    }));
        }
        return tests;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Table-level invariants
    // ────────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("the mirrored table has entries for the canonical regression ops (gather/gather_nd/repeat)")
    void testCanonicalRegressionOpsArePresent() {
        for (String op : new String[]{"gather", "gather_nd", "repeat", "concat", "stack", "split", "split_v"}) {
            assertNotEquals(null, EXPECTED_TRAITS.get(op),
                    "op '" + op + "' must be mirrored in EXPECTED_TRAITS so future drift is caught");
        }
        // And they must NOT be tagged value-dep in the Java mirror either.
        for (String op : new String[]{
                "gather", "gather_nd", "repeat", "concat", "stack", "split", "split_v",
                "expand_dims", "squeeze", "flatten", "flatten_2d", "permute"}) {
            Integer e = EXPECTED_TRAITS.get(op);
            assertNotEquals(null, e, "expected-traits mirror must include " + op);
            assertEquals(0, e & OP_TRAIT_VALUE_DEPENDENT_SHAPE,
                    "Java mirror for '" + op + "' accidentally sets VALUE_DEPENDENT_SHAPE; "
                            + "the C++ table explicitly omits this bit for shape-determined ops");
        }
    }

    @Test
    @DisplayName("the expected-traits mirror contains at least the ~160 ops from OpTraitTable.cpp")
    void testMirrorSize() {
        assertEquals(true, EXPECTED_TRAITS.size() >= 150,
                "mirror size looks wrong: " + EXPECTED_TRAITS.size()
                        + " — if the C++ table grew, add the new entries here");
    }
}
