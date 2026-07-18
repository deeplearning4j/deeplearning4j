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
package org.eclipse.deeplearning4j.nd4j.backends;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.custom.LogSumExp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm1;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm;
import org.nd4j.linalg.api.ops.impl.scalar.PRelu;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FlashAttention;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ApplyAlibi;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedBiasDropoutResidual;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedMRoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GroupedQueryAttention;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RoPE;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SiluAndMul;
import org.nd4j.linalg.api.ops.impl.transforms.custom.VisionEmbeddingMerge;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.RealDivOp;
import org.nd4j.linalg.api.ops.impl.transforms.same.Cube;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Rint;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.nativeblas.MultiBackendNativeOpsHolder;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Strict, emitter-owned Vulkan coverage tests.
 *
 * <p>Every case is a one-op SameDiff graph. A passing nonconstant case proves
 * that the named descriptor was captured into one real Vulkan compute dispatch,
 * replayed with changed input data, and never executed through slot-by-slot or
 * fallback mode. Zero-input constants prove the same capture dispatch before
 * entering the framework's frozen-constant state. Numerical references are
 * deliberately pure Java and do not call another ND4J implementation of the op.</p>
 */
@Slf4j
@Tag(TagNames.VULKAN)
@DisplayName("Vulkan kernel emitters: strict real-pipeline capture/replay")
public class VulkanKernelEmitterStrictReplayTest {

    private static final String VULKAN_BINDINGS_CLASS =
            "org.nd4j.linalg.vulkan.bindings.Nd4jVulkan";
    private static final String OUTPUT = "out";
    private static final int REPLAY_STEPS = 24;
    private static final float ABS_TOL = 5.0e-4f;
    private static final float REL_TOL = 5.0e-4f;

    // Canonical FusedElemOp values accepted by the Vulkan descriptor validator.
    private static final int[] ACCEPTED_CHAIN_CODES = acceptedChainCodes();

    private static NativeOps nativeOps;
    private static int selectedDeviceId;
    private static String selectedDeviceName;
    private static boolean mlirEnabled;

    @BeforeAll
    static void setupVulkan() {
        try {
            assertNotNull(Nd4j.getBackend(),
                    "The Vulkan test profile did not initialize an ND4J backend");
            NativeOps activeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            assertNotNull(activeOps, "The Vulkan test profile did not initialize NativeOps");
            assertEquals(VULKAN_BINDINGS_CLASS, activeOps.getClass().getName(),
                    "The strict Vulkan suite must run through the tooling-selected Vulkan backend");

            MultiBackendNativeOpsHolder holder = MultiBackendNativeOpsHolder.getInstance();
            nativeOps = holder.getOpsForDeviceType(DeviceType.VULKAN_GPU);
            assertSame(activeOps, nativeOps,
                    "NativeOpsHolder and the multi-backend registry must share one Vulkan binding");

            Class<?> bindingsClass = nativeOps.getClass();
            int count = nativeOps.getAvailableDevices();
            assertTrue(count > 0, "Strict Vulkan emitter tests require an enumerated Vulkan device");

            String requestedRegex =
                    System.getProperty("nd4j.vulkan.test.deviceNameRegex", "").trim();
            Pattern requested = requestedRegex.isEmpty() ? null : Pattern.compile(requestedRegex);
            selectedDeviceId = -1;
            for (int deviceId = 0; deviceId < count; deviceId++) {
                String name = nativeOps.getDeviceName(deviceId);
                if (requested == null || requested.matcher(name).find()) {
                    selectedDeviceId = deviceId;
                    selectedDeviceName = name;
                    break;
                }
            }
            assertTrue(selectedDeviceId >= 0,
                    "No Vulkan device matched nd4j.vulkan.test.deviceNameRegex=" + requestedRegex);
            assertEquals(1, nativeOps.setDevice(selectedDeviceId),
                    "Selecting the Vulkan test device must succeed");

            mlirEnabled = probeMlir(bindingsClass);
            assertTrue(mlirEnabled,
                    "Strict Vulkan kernel-emitter tests require HAVE_MLIR=1");
            assertEquals("VULKAN",
                    System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE),
                    "Run strict emitter tests with -Dnd4j.dsp.graphExecutionMode=VULKAN");

            log.info("Strict Vulkan emitter tests selected device[{}]='{}'",
                    selectedDeviceId, selectedDeviceName);
        } catch (ReflectiveOperationException e) {
            fail("Vulkan bindings do not expose the strict replay diagnostics contract", e);
        }
    }

    @Test
    @DisplayName("fused_elementwise_chain: every accepted opcode is one real replayed pipeline")
    void fusedElementwiseAcceptedOpcodesCreateSinglePipelinesAndReplay() {
        assertEquals(47, ACCEPTED_CHAIN_CODES.length,
                "The test catalogue must track every accepted fused-chain opcode");

        for (int code : ACCEPTED_CHAIN_CODES) {
            final boolean binary = isBinaryChainCode(code);
            final List<InputSpec> inputs = new ArrayList<>();
            inputs.add(new InputSpec("x", new long[]{2, 3, 4},
                    step -> chainPrimary(code, step, 24)));
            if (binary) {
                inputs.add(new InputSpec("rhs", new long[]{2, 3, 4},
                        step -> chainSecondary(code, step, 24)));
            }

            runStrictSingleDispatch(
                    "fused_elementwise_chain opcode=" + code,
                    "fused_elementwise_chain",
                    inputs,
                    (sd, variables) -> {
                        SDVariable[] secondary = binary
                                ? new SDVariable[]{variables.get("rhs")}
                                : new SDVariable[0];
                        FusedElementwiseChain op = new FusedElementwiseChain(
                                sd, variables.get("x"), secondary, new int[]{code});
                        if (code == 30) {
                            op.addTArgument(-0.75, 0.625);
                        }
                        return op.outputVariable();
                    },
                    step -> {
                        float[][] values = binary
                                ? new float[][]{
                                        chainPrimary(code, step, 24),
                                        chainSecondary(code, step, 24)}
                                : new float[][]{chainPrimary(code, step, 24)};
                        return applyChain(values, new int[]{code}, -0.75f, 0.625f);
                    },
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("fused_elementwise_chain: an eight-step chain stays one dispatch")
    void fusedElementwiseMaximumEightStepChainCreatesOnePipeline() {
        final int[] codes = {0, 10, 2, 20, 18, 38, 39, 42};
        final List<InputSpec> inputs = List.of(
                new InputSpec("x", new long[]{2, 3, 4},
                        step -> generalValues(step, 24, 0.19f, -1.1f)),
                new InputSpec("addend", new long[]{2, 3, 4},
                        step -> generalValues(step + 3, 24, 0.07f, -0.2f)),
                new InputSpec("factor", new long[]{2, 3, 4},
                        step -> generalValues(step + 5, 24, 0.03f, 0.55f)));

        runStrictSingleDispatch(
                "fused_elementwise_chain eight-step maximum",
                "fused_elementwise_chain",
                inputs,
                (sd, variables) -> new FusedElementwiseChain(
                        sd,
                        variables.get("x"),
                        new SDVariable[]{variables.get("addend"), variables.get("factor")},
                        codes).outputVariable(),
                step -> applyChain(
                        new float[][]{
                                generalValues(step, 24, 0.19f, -1.1f),
                                generalValues(step + 3, 24, 0.07f, -0.2f),
                                generalValues(step + 5, 24, 0.03f, 0.55f)},
                        codes, 0.0f, 0.0f),
                8.0e-4f,
                8.0e-4f);
    }

    @Test
    @DisplayName("swiglu/geglu/reglu: standalone GLU emitters each replay one pipeline")
    void standaloneGluVariantsCreateSinglePipelinesAndReplay() {
        for (String opName : new String[]{"swiglu", "geglu", "reglu"}) {
            final int length = 2 * 3 * 8;
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec("x", new long[]{2, 3, 8},
                            step -> generalValues(step, length, 0.11f, -1.15f)));

            runStrictSingleDispatch(
                    opName + " rank-3",
                    opName,
                    inputs,
                    (sd, variables) -> new NamedDynamicOp(
                            opName, sd, variables.get("x")).outputVariable(),
                    step -> gluOracle(
                            generalValues(step, length, 0.11f, -1.15f),
                            2 * 3, 4, opName),
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("mean_square: keep-dims and squeezed last-axis reductions replay one pipeline")
    void meanSquareKeepDimsVariantsCreateSinglePipelinesAndReplay() {
        for (boolean keepDims : new boolean[]{true, false}) {
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec("x", new long[]{2, 3, 4},
                            step -> generalValues(step, 24, 0.17f, -1.2f)));

            runStrictSingleDispatch(
                    "mean_square keepDims=" + keepDims,
                    "mean_square",
                    inputs,
                    (sd, variables) ->
                            new MeanSquare(sd, variables.get("x"), keepDims).outputVariable(),
                    step -> meanSquareOracle(
                            generalValues(step, 24, 0.17f, -1.2f), 6, 4),
                    2.0e-5f,
                    2.0e-5f);
        }
    }

    @Test
    @DisplayName("matmul: rank-3 batched matrices replay one pipeline")
    void rankThreeBatchedMatmulCreatesSinglePipelineAndReplays() {
        final int batch = 2;
        final int rows = 3;
        final int inner = 4;
        final int columns = 5;
        final List<InputSpec> inputs = List.of(
                new InputSpec("a", new long[]{batch, rows, inner},
                        step -> generalValues(step, batch * rows * inner, 0.09f, -0.7f)),
                new InputSpec("b", new long[]{batch, inner, columns},
                        step -> generalValues(step + 7, batch * inner * columns, 0.05f, -0.35f)));

        runStrictSingleDispatch(
                "matmul rank-3 batched",
                "matmul",
                inputs,
                (sd, variables) -> sd.mmul(variables.get("a"), variables.get("b")),
                step -> batchedMatmulOracle(
                        generalValues(step, batch * rows * inner, 0.09f, -0.7f),
                        generalValues(step + 7, batch * inner * columns, 0.05f, -0.35f),
                        batch, rows, inner, columns),
                1.0e-4f,
                1.0e-4f);
    }

    @Test
    @DisplayName("fused_attention_projection: rank-3/rank-4 attention layouts replay one pipeline")
    void fusedAttentionProjectionRankVariantsCreateSinglePipelinesAndReplay() {
        runAttentionProjectionCase(false);
        runAttentionProjectionCase(true);
    }

    private void runAttentionProjectionCase(boolean rankFour) {
        final int batch = 2;
        final int sequence = 3;
        final int heads = 2;
        final int headSize = 2;
        final int hidden = heads * headSize;
        final int outputSize = 5;
        final long[] attentionShape = rankFour
                ? new long[]{batch, sequence, heads, headSize}
                : new long[]{batch, sequence, hidden};
        final boolean withBias = !rankFour;

        List<InputSpec> inputs = new ArrayList<>();
        inputs.add(new InputSpec("attention", attentionShape,
                step -> generalValues(step, batch * sequence * hidden, 0.08f, -0.55f)));
        inputs.add(new InputSpec("weight", new long[]{hidden, outputSize},
                step -> generalValues(step + 11, hidden * outputSize, 0.045f, -0.25f)));
        if (withBias) {
            inputs.add(new InputSpec("bias", new long[]{outputSize},
                    step -> generalValues(step + 2, outputSize, 0.025f, -0.04f)));
        }

        runStrictSingleDispatch(
                "fused_attention_projection " + (rankFour ? "rank-4" : "rank-3")
                        + (withBias ? " with bias" : " without bias"),
                "fused_attention_projection",
                inputs,
                (sd, variables) -> {
                    if (withBias) {
                        return new NamedDynamicOp(
                                "fused_attention_projection", sd,
                                variables.get("attention"),
                                variables.get("weight"),
                                variables.get("bias")).outputVariable();
                    }
                    return new NamedDynamicOp(
                            "fused_attention_projection", sd,
                            variables.get("attention"),
                            variables.get("weight")).outputVariable();
                },
                step -> attentionProjectionOracle(
                        generalValues(step, batch * sequence * hidden, 0.08f, -0.55f),
                        generalValues(step + 11, hidden * outputSize, 0.045f, -0.25f),
                        withBias
                                ? generalValues(step + 2, outputSize, 0.025f, -0.04f)
                                : null,
                        batch, sequence, hidden, outputSize),
                1.0e-4f,
                1.0e-4f);
    }

    @Test
    @DisplayName("log_softmax: rank-2 last-axis normalization replays one pipeline")
    void logSoftmaxCreatesSinglePipelineAndReplays() {
        final int rows = 4;
        final int width = 7;
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec("x", new long[]{rows, width},
                        step -> generalValues(step, rows * width, 1.15f, -7.5f)));

        runStrictSingleDispatch(
                "log_softmax rank-2 axis=1",
                "log_softmax",
                inputs,
                (sd, variables) ->
                        new LogSoftMax(sd, variables.get("x"), 1).outputVariable(),
                step -> logSoftmaxOracle(
                        generalValues(step, rows * width, 1.15f, -7.5f),
                        rows, width),
                1.0e-4f,
                1.0e-4f);
    }

    @Test
    @DisplayName("fused_bias_dropout_residual: deterministic inference replays one pipeline")
    void fusedBiasDropoutResidualDeterministicCreatesSinglePipelineAndReplays() {
        final int rows = 2 * 3;
        final int width = 5;
        final List<InputSpec> inputs = List.of(
                new InputSpec("x", new long[]{2, 3, width},
                        step -> generalValues(step, rows * width, 0.12f, -0.8f)),
                new InputSpec("bias", new long[]{width},
                        step -> generalValues(step + 5, width, 0.04f, -0.15f)),
                new InputSpec("residual", new long[]{2, 3, width},
                        step -> generalValues(step + 11, rows * width, 0.07f, -0.45f)));

        runStrictSingleDispatch(
                "fused_bias_dropout_residual p=0 training=false",
                "fused_bias_dropout_residual",
                inputs,
                (sd, variables) -> new FusedBiasDropoutResidual(
                        sd,
                        variables.get("x"),
                        variables.get("bias"),
                        variables.get("residual"),
                        0.0,
                        0x5EEDL,
                        false).outputVariable(),
                step -> biasDropoutResidualOracle(
                        generalValues(step, rows * width, 0.12f, -0.8f),
                        generalValues(step + 5, width, 0.04f, -0.15f),
                        generalValues(step + 11, rows * width, 0.07f, -0.45f),
                        rows, width),
                2.0e-5f,
                2.0e-5f);
    }

    @Test
    @DisplayName("softplus: stable extreme-value formula replays one pipeline")
    void softplusExtremeValuesCreateSinglePipelineAndReplayWithoutOverflow() {
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec("x", new long[]{3, 4},
                        step -> softplusExtremeValues(step)));

        runStrictSingleDispatch(
                "softplus stable extremes",
                "softplus",
                inputs,
                (sd, variables) ->
                        new SoftPlus(sd, variables.get("x")).outputVariable(),
                step -> softplusOracle(softplusExtremeValues(step)),
                2.0e-5f,
                2.0e-5f);
    }

    @Test
    @DisplayName("fused_mrope: contiguous and interleaved rank-4 layouts replay one pipeline")
    void fusedMRoPEVariantsCreateSinglePipelinesAndReplay() {
        for (boolean interleaved : new boolean[]{false, true}) {
            runMRoPECase(interleaved);
        }
    }

    private void runMRoPECase(boolean interleaved) {
        final int batch = 2;
        final int sequence = 3;
        final int heads = 2;
        final int headDimension = 12;
        final int sectionT = 4;
        final int sectionH = 4;
        final int sectionW = 4;
        final double frequencyBase = 10000.0;
        final int tensorLength = batch * sequence * heads * headDimension;
        final int positionsLength = batch * sequence;

        final List<InputSpec> inputs = List.of(
                new InputSpec("x", new long[]{batch, sequence, heads, headDimension},
                        step -> generalValues(step, tensorLength, 0.035f, -0.85f)),
                new InputSpec("position_t", new long[]{batch, sequence},
                        step -> positionValues(step, positionsLength, 0)),
                new InputSpec("position_h", new long[]{batch, sequence},
                        step -> positionValues(step, positionsLength, 2)),
                new InputSpec("position_w", new long[]{batch, sequence},
                        step -> positionValues(step, positionsLength, 4)));

        runStrictSingleDispatch(
                "fused_mrope " + (interleaved ? "interleaved" : "contiguous"),
                "fused_mrope",
                inputs,
                (sd, variables) -> new FusedMRoPE(
                        sd,
                        variables.get("x"),
                        variables.get("position_t"),
                        variables.get("position_h"),
                        variables.get("position_w"),
                        sectionT,
                        sectionH,
                        sectionW,
                        interleaved,
                        frequencyBase).outputVariable(),
                step -> mropeOracle(
                        generalValues(step, tensorLength, 0.035f, -0.85f),
                        positionValues(step, positionsLength, 0),
                        positionValues(step, positionsLength, 2),
                        positionValues(step, positionsLength, 4),
                        batch,
                        sequence,
                        heads,
                        headDimension,
                        sectionT,
                        sectionH,
                        sectionW,
                        interleaved,
                        frequencyBase),
                3.0e-4f,
                3.0e-4f);
    }

    @Test
    @DisplayName("catalogued unary emitters: canonical descriptors each replay one pipeline")
    void cataloguedUnaryEmittersCreateSinglePipelinesAndReplay() {
        final long[] shape = {2, 3, 4};
        final int length = 24;
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec("x", shape,
                        step -> generalValues(step, length, 0.17f, -1.65f)));

        runStrictSingleDispatch(
                "cube rank-3",
                "cube",
                inputs,
                (sd, variables) ->
                        new Cube(sd, variables.get("x")).outputVariable(),
                step -> mapValues(
                        generalValues(step, length, 0.17f, -1.65f),
                        value -> value * value * value),
                3.0e-5f,
                3.0e-5f);

        runStrictSingleDispatch(
                "rectifiedtanh canonical descriptor",
                "rectifiedtanh",
                inputs,
                (sd, variables) -> new NamedDynamicOp(
                        "rectifiedtanh", sd, variables.get("x")).outputVariable(),
                step -> mapValues(
                        generalValues(step, length, 0.17f, -1.65f),
                        value -> Math.max(0.0, Math.tanh(value))),
                ABS_TOL,
                REL_TOL);

        runStrictSingleDispatch(
                "rationaltanh canonical descriptor",
                "rationaltanh",
                inputs,
                (sd, variables) -> new NamedDynamicOp(
                        "rationaltanh", sd, variables.get("x")).outputVariable(),
                step -> mapValues(
                        generalValues(step, length, 0.17f, -1.65f),
                        VulkanKernelEmitterStrictReplayTest::rationalTanh),
                ABS_TOL,
                REL_TOL);

        runStrictSingleDispatch(
                "Floor canonical descriptor",
                "Floor",
                inputs,
                (sd, variables) -> new NamedDynamicOp(
                        "Floor", sd, variables.get("x")).outputVariable(),
                step -> mapValues(
                        generalValues(step, length, 0.17f, -1.65f),
                        Math::floor),
                0.0f,
                0.0f);

        final List<InputSpec> rintInputs = Collections.singletonList(
                new InputSpec("x", new long[]{3, 4},
                        VulkanKernelEmitterStrictReplayTest::roundingValues));
        runStrictSingleDispatch(
                "rint round-to-nearest-even",
                "rint",
                rintInputs,
                (sd, variables) ->
                        new Rint(sd, variables.get("x"), false).outputVariable(),
                step -> mapValues(roundingValues(step), Math::rint),
                0.0f,
                0.0f);

        final double scale = -1.75;
        runStrictSingleDispatch(
                "scale explicit scalar",
                "scale",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op =
                            new NamedDynamicOp("scale", sd, variables.get("x"));
                    op.addTArgument(scale);
                    return op.outputVariable();
                },
                step -> mapValues(
                        generalValues(step, length, 0.17f, -1.65f),
                        value -> value * scale),
                2.0e-5f,
                2.0e-5f);

        final double seluLambda = 1.0507009873554805;
        final double seluAlpha = 1.6732632423543772;
        final List<UnaryForwardSpec> forwardSpecs = List.of(
                new UnaryForwardSpec(
                        "silu", new double[0],
                        value -> value / (1.0 + Math.exp(-value))),
                new UnaryForwardSpec(
                        "fused_gelu", new double[0],
                        value -> value / (1.0 + Math.exp(-1.702 * value))),
                new UnaryForwardSpec(
                        "tanh", new double[0], Math::tanh),
                new UnaryForwardSpec(
                        "sigmoid", new double[0],
                        value -> 1.0 / (1.0 + Math.exp(-value))),
                new UnaryForwardSpec(
                        "relu", new double[0], value -> Math.max(0.0, value)),
                new UnaryForwardSpec(
                        "square", new double[0], value -> value * value),
                new UnaryForwardSpec(
                        "squared_relu", new double[0],
                        value -> Math.max(0.0, value) * Math.max(0.0, value)),
                new UnaryForwardSpec(
                        "hardsigmoid", new double[0],
                        value -> Math.min(1.0, Math.max(0.0, 0.2 * value + 0.5))),
                new UnaryForwardSpec(
                        "hardtanh", new double[0],
                        value -> Math.min(1.0, Math.max(-1.0, value))),
                new UnaryForwardSpec(
                        "relu6", new double[]{0.0},
                        value -> Math.min(6.0, Math.max(0.0, value))),
                new UnaryForwardSpec(
                        "lrelu", new double[]{0.2},
                        value -> value < 0.0 ? 0.2 * value : value),
                new UnaryForwardSpec(
                        "elu", new double[]{0.7},
                        value -> value >= 0.0 ? value : 0.7 * Math.expm1(value)),
                new UnaryForwardSpec(
                        "selu", new double[0],
                        value -> seluLambda * (value > 0.0
                                ? value : seluAlpha * Math.expm1(value))),
                new UnaryForwardSpec(
                        "softsign", new double[0],
                        value -> value / (1.0 + Math.abs(value))),
                new UnaryForwardSpec(
                        "clipbyvalue", new double[]{-0.6, 0.8},
                        value -> Math.min(0.8, Math.max(-0.6, value))),
                new UnaryForwardSpec(
                        "thresholdedrelu", new double[]{0.75},
                        value -> value > 0.75 ? value : 0.0));
        final List<InputSpec> forwardInputs = Collections.singletonList(
                new InputSpec("x", shape,
                        step -> activationBackwardValues(step, length)));
        for (UnaryForwardSpec spec : forwardSpecs) {
            runStrictSingleDispatch(
                    spec.opName + " canonical forward descriptor",
                    spec.opName,
                    forwardInputs,
                    (sd, variables) -> {
                        NamedDynamicOp op = new NamedDynamicOp(
                                spec.opName, sd, variables.get("x"));
                        if (spec.tArguments.length != 0) {
                            op.addTArgument(spec.tArguments);
                        }
                        return op.outputVariable();
                    },
                    step -> mapValues(
                            activationBackwardValues(step, length),
                            spec.function),
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("activation backward: every canonical gradient op replays one real pipeline")
    void activationBackwardEmittersCreateSinglePipelinesAndReplay() {
        final double seluLambda = 1.0507009873554805;
        final double seluAlpha = 1.6732632423543772;
        final List<ActivationBackwardSpec> specs = List.of(
                new ActivationBackwardSpec(
                        "relu_bp", new double[0],
                        value -> value > 0.0 ? 1.0 : 0.0),
                new ActivationBackwardSpec(
                        "relu6_bp", new double[0],
                        value -> value > 0.0 && value < 6.0 ? 1.0 : 0.0),
                new ActivationBackwardSpec(
                        "thresholdedrelu_bp", new double[]{0.75},
                        value -> value > 0.75 ? 1.0 : 0.0),
                new ActivationBackwardSpec(
                        "sigmoid_bp", new double[0],
                        value -> {
                            double sigmoid = 1.0 / (1.0 + Math.exp(-value));
                            return sigmoid * (1.0 - sigmoid);
                        }),
                new ActivationBackwardSpec(
                        "tanh_bp", new double[0],
                        value -> {
                            double tanh = Math.tanh(value);
                            return 1.0 - tanh * tanh;
                        }),
                new ActivationBackwardSpec(
                        "elu_bp", new double[]{0.7},
                        value -> value >= 0.0 ? 1.0 : 0.7 * Math.exp(value)),
                new ActivationBackwardSpec(
                        "selu_bp", new double[0],
                        value -> value > 0.0
                                ? seluLambda
                                : seluAlpha * seluLambda * Math.exp(value)),
                new ActivationBackwardSpec(
                        "lrelu_bp", new double[]{0.2},
                        value -> value >= 0.0 ? 1.0 : 0.2),
                new ActivationBackwardSpec(
                        "softplus_bp", new double[0],
                        value -> 1.0 / (1.0 + Math.exp(-value))),
                new ActivationBackwardSpec(
                        "softsign_bp", new double[0],
                        value -> {
                            double denominator = 1.0 + Math.abs(value);
                            return 1.0 / (denominator * denominator);
                        }),
                new ActivationBackwardSpec(
                        "hardsigmoid_bp", new double[0],
                        value -> value >= -2.5 && value <= 2.5 ? 0.2 : 0.0),
                new ActivationBackwardSpec(
                        "hardtanh_bp", new double[0],
                        value -> value >= -1.0 && value <= 1.0 ? 1.0 : 0.0),
                new ActivationBackwardSpec(
                        "silu_bp", new double[0],
                        value -> {
                            double sigmoid = 1.0 / (1.0 + Math.exp(-value));
                            return sigmoid + value * sigmoid * (1.0 - sigmoid);
                        }),
                new ActivationBackwardSpec(
                        "fused_gelu_bp", new double[0],
                        value -> {
                            double sigmoid = 1.0 / (1.0 + Math.exp(-1.702 * value));
                            return sigmoid
                                    + 1.702 * value * sigmoid * (1.0 - sigmoid);
                        }),
                new ActivationBackwardSpec(
                        "squared_relu_bp", new double[0],
                        value -> value > 0.0 ? 2.0 * value : 0.0),
                new ActivationBackwardSpec(
                        "rectifiedtanh_bp", new double[0],
                        value -> {
                            double tanh = Math.tanh(value);
                            return value > 0.0 ? 1.0 - tanh * tanh : 0.0;
                        }));

        final long[] shape = {2, 3, 4};
        final int length = 24;
        for (ActivationBackwardSpec spec : specs) {
            final List<InputSpec> inputs = List.of(
                    new InputSpec("x", shape,
                            step -> activationBackwardValues(step, length)),
                    new InputSpec("gradient", shape,
                            step -> generalValues(step + 17, length, 0.07f, -0.35f)));

            runStrictSingleDispatch(
                    spec.opName + " rank-3 boundary coverage",
                    spec.opName,
                    inputs,
                    (sd, variables) -> {
                        NamedDynamicOp op = new NamedDynamicOp(
                                spec.opName,
                                sd,
                                variables.get("x"),
                                variables.get("gradient"));
                        if (spec.tArguments.length != 0) {
                            op.addTArgument(spec.tArguments);
                        }
                        return op.outputVariable();
                    },
                    step -> activationBackwardOracle(
                            activationBackwardValues(step, length),
                            generalValues(step + 17, length, 0.07f, -0.35f),
                            spec.derivative),
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("swish_mul_bp: both gradients come from one fused replayed pipeline")
    void swishMulBackwardCreatesSinglePipelineAndReplaysBothOutputs() {
        final long[] shape = {2, 3, 4};
        final int length = 24;
        final List<InputSpec> inputs = List.of(
                new InputSpec("x", shape,
                        step -> activationBackwardValues(step, length)),
                new InputSpec("y", shape,
                        step -> generalValues(step + 7, length, 0.09f, -0.65f)),
                new InputSpec("gradient", shape,
                        step -> generalValues(step + 13, length, 0.06f, -0.42f)));

        for (int outputIndex = 0; outputIndex < 2; outputIndex++) {
            final int observedOutput = outputIndex;
            runStrictSingleDispatch(
                    "swish_mul_bp observe "
                            + (observedOutput == 0 ? "gradX" : "gradY"),
                    "swish_mul_bp",
                    inputs,
                    (sd, variables) -> new NamedFixedOutputOp(
                            "swish_mul_bp",
                            2,
                            sd,
                            variables.get("x"),
                            variables.get("y"),
                            variables.get("gradient"))
                            .outputVariables()[observedOutput],
                    step -> swishMulBackwardOracle(
                            activationBackwardValues(step, length),
                            generalValues(step + 7, length, 0.09f, -0.65f),
                            generalValues(step + 13, length, 0.06f, -0.42f),
                            observedOutput),
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("rms_norm_bp: rank-3 strided views replay one fused row kernel")
    void rmsNormBackwardStridedViewsCreateSinglePipelineAndReplay() {
        final int batch = 2;
        final int sequence = 3;
        final int features = 5;
        final int length = batch * sequence * features;
        final double epsilon = 2.0e-4;
        final long[] shape = {batch, sequence, features};
        final List<InputSpec> inputs = List.of(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        shape,
                        step -> featureStridedInput(
                                step, batch, sequence, features),
                        true),
                new InputSpec(
                        "gradient",
                        DataType.FLOAT,
                        shape,
                        step -> featureStridedInput(
                                step + 7, batch, sequence, features),
                        true));

        runStrictSingleDispatch(
                "rms_norm_bp gamma-free rank-3 strided views",
                "rms_norm_bp",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "rms_norm_bp",
                            sd,
                            variables.get("x"),
                            variables.get("gradient"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> rmsNormBackwardOracle(
                        generalValues(step, length, 0.0375f, -0.65f),
                        generalValues(step + 7, length, 0.0375f, -0.65f),
                        batch * sequence,
                        features,
                        epsilon),
                DataType.FLOAT,
                shape,
                3.0e-4f,
                3.0e-4f);
    }

    @Test
    @DisplayName("fused_layer_norm_bp: rank-1 strided inputs replay one two-output pipeline")
    void fusedLayerNormBackwardRankOneCreatesSinglePipelineAndReplaysBothOutputs() {
        final int features = 9;
        final double epsilon = 3.0e-4;
        final long[] shape = {features};
        final List<InputSpec> inputs = List.of(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        shape,
                        step -> rankOneStridedInput(
                                generalValues(step, features, 0.075f, -0.55f)),
                        true),
                new InputSpec(
                        "gain",
                        DataType.FLOAT,
                        shape,
                        step -> rankOneStridedInput(
                                generalValues(step + 5, features, 0.04f, 0.72f)),
                        true),
                new InputSpec(
                        "gradient",
                        DataType.FLOAT,
                        shape,
                        step -> rankOneStridedInput(
                                generalValues(step + 11, features, 0.055f, -0.31f)),
                        true));

        for (int outputIndex = 0; outputIndex < 2; outputIndex++) {
            final int observedOutput = outputIndex;
            runStrictSingleDispatch(
                    "fused_layer_norm_bp rank-1 observe "
                            + (observedOutput == 0 ? "dX" : "dGamma"),
                    "fused_layer_norm_bp",
                    inputs,
                    (sd, variables) -> {
                        NamedFixedOutputOp op = new NamedFixedOutputOp(
                                "fused_layer_norm_bp",
                                2,
                                sd,
                                variables.get("x"),
                                variables.get("gain"),
                                variables.get("gradient"));
                        op.addTArgument(epsilon);
                        return op.outputVariables()[observedOutput];
                    },
                    step -> fusedLayerNormBackwardOracle(
                            generalValues(step, features, 0.075f, -0.55f),
                            generalValues(step + 5, features, 0.04f, 0.72f),
                            generalValues(step + 11, features, 0.055f, -0.31f),
                            epsilon,
                            observedOutput),
                    DataType.FLOAT,
                    shape,
                    4.0e-4f,
                    4.0e-4f);
        }
    }

    @Test
    @DisplayName("Log1p: canonical native descriptor maps INT32 input to FLOAT output")
    void log1pInt32InputCreatesFloatingSinglePipelineAndReplays() {
        final long[] shape = {2, 3, 4};
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.INT32,
                        shape,
                        step -> nonNegativeIntegerValues(step, 24)));

        // The legacy Java Log1p transform is strict-FP only. The canonical
        // native Log1p descriptor itself accepts integer input and selects a
        // floating output, which is the contract exercised here.
        runStrictSingleDispatch(
                "Log1p INT32 to FLOAT",
                "Log1p",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "Log1p", DataType.FLOAT, sd, variables.get("x"));
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> mapValues(
                        nonNegativeIntegerValues(step, 24),
                        Math::log1p),
                DataType.FLOAT,
                shape,
                2.0e-5f,
                2.0e-5f);
    }

    @Test
    @DisplayName("realdiv/add_inplace/silu_and_mul: binary emitters replay one pipeline")
    void cataloguedBinaryEmittersCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 4;
        final long[] outputShape = {batch, rows, width};

        final List<InputSpec> realDivInputs = List.of(
                new InputSpec(
                        "numerator",
                        DataType.INT32,
                        outputShape,
                        step -> signedIntegerValues(step, batch * rows * width)),
                new InputSpec(
                        "denominator",
                        new long[]{1, rows, 1},
                        step -> divisorValues(step, rows)));
        runStrictSingleDispatch(
                "realdiv mixed INT32/FLOAT broadcast",
                "realdiv",
                realDivInputs,
                (sd, variables) -> new RealDivOp(
                        sd,
                        new SDVariable[]{
                                variables.get("numerator"),
                                variables.get("denominator")},
                        false).outputVariable(),
                step -> realDivisionOracle(
                        signedIntegerValues(step, batch * rows * width),
                        divisorValues(step, rows),
                        batch,
                        rows,
                        width),
                DataType.FLOAT,
                outputShape,
                2.0e-5f,
                2.0e-5f);

        final List<InputSpec> addInputs = List.of(
                new InputSpec("accumulator", outputShape,
                        step -> generalValues(step, batch * rows * width, 0.11f, -0.8f)),
                new InputSpec("addend", outputShape,
                        step -> generalValues(step + 7, batch * rows * width, 0.06f, -0.3f)));
        runStrictSingleDispatch(
                "add_inplace same-shape rank-3",
                "add_inplace",
                addInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "add_inplace",
                        sd,
                        variables.get("accumulator"),
                        variables.get("addend")).outputVariable(),
                step -> pairwiseOracle(
                        generalValues(step, batch * rows * width, 0.11f, -0.8f),
                        generalValues(step + 7, batch * rows * width, 0.06f, -0.3f),
                        (left, right) -> left + right),
                2.0e-5f,
                2.0e-5f);

        final List<InputSpec> siluInputs = List.of(
                new InputSpec("gate", outputShape,
                        step -> generalValues(step, batch * rows * width, 0.14f, -1.05f)),
                new InputSpec("up", outputShape,
                        step -> generalValues(step + 11, batch * rows * width, 0.08f, -0.45f)));
        runStrictSingleDispatch(
                "silu_and_mul rank-3",
                "silu_and_mul",
                siluInputs,
                (sd, variables) -> new SiluAndMul(
                        sd,
                        variables.get("gate"),
                        variables.get("up")).outputVariable(),
                step -> siluAndMulOracle(
                        generalValues(step, batch * rows * width, 0.14f, -1.05f),
                        generalValues(step + 11, batch * rows * width, 0.08f, -0.45f)),
                ABS_TOL,
                REL_TOL);

        final int length = batch * rows * width;
        final IntFunction<float[]> leftValues = step -> {
            float[] values = new float[length];
            for (int i = 0; i < length; i++) {
                values[i] = ((i + step) % 9) - 3.75f;
            }
            return values;
        };
        final IntFunction<float[]> rightValues = step -> {
            float[] values = new float[length];
            for (int i = 0; i < length; i++) {
                values[i] = 1.0f + ((i * 3 + step) % 4);
            }
            return values;
        };
        final List<InputSpec> binaryInputs = List.of(
                new InputSpec("left", outputShape, leftValues),
                new InputSpec("right", outputShape, rightValues));
        final List<BinaryForwardSpec> binarySpecs = List.of(
                new BinaryForwardSpec("add", (left, right) -> left + right),
                new BinaryForwardSpec("subtract", (left, right) -> left - right),
                new BinaryForwardSpec("multiply", (left, right) -> left * right),
                new BinaryForwardSpec("divide", (left, right) -> left / right),
                new BinaryForwardSpec("minimum", Math::min),
                new BinaryForwardSpec("maximum", Math::max),
                new BinaryForwardSpec(
                        "floormod",
                        (left, right) -> left - Math.floor(left / right) * right),
                new BinaryForwardSpec("mod", (left, right) -> left % right),
                new BinaryForwardSpec("floordiv", (left, right) -> Math.floor(left / right)),
                new BinaryForwardSpec("reversedivide", (left, right) -> right / left),
                new BinaryForwardSpec("reversesubtract", (left, right) -> right - left),
                new BinaryForwardSpec(
                        "squaredsubtract",
                        (left, right) -> (left - right) * (left - right)),
                new BinaryForwardSpec("Pow", Math::pow),
                new BinaryForwardSpec("tf_atan2", Math::atan2),
                new BinaryForwardSpec(
                        "swish_mul",
                        (left, right) -> left / (1.0 + Math.exp(-left)) * right),
                new BinaryForwardSpec("assign", (left, right) -> right));
        for (BinaryForwardSpec spec : binarySpecs) {
            runStrictSingleDispatch(
                    spec.opName + " canonical binary descriptor",
                    spec.opName,
                    binaryInputs,
                    (sd, variables) -> new NamedDynamicOp(
                            spec.opName,
                            sd,
                            variables.get("left"),
                            variables.get("right")).outputVariable(),
                    step -> pairwiseOracle(
                            leftValues.apply(step),
                            rightValues.apply(step),
                            spec.function),
                    ABS_TOL,
                    REL_TOL);
        }
    }

    @Test
    @DisplayName("remaining fused forward descriptors each replay one real compute pipeline")
    void remainingFusedForwardDescriptorsCreateSinglePipelinesAndReplay() {
        final int rows = 2;
        final int hidden = 4;
        final int projected = 3;
        final double epsilon = 1.0e-4;
        final long[] rowShape = {rows, hidden};
        final List<InputSpec> normalizationInputs = List.of(
                new InputSpec(
                        "x",
                        rowShape,
                        step -> generalValues(step, rows * hidden, 0.16f, -0.7f)),
                new InputSpec(
                        "gamma",
                        new long[]{hidden},
                        step -> generalValues(step + 3, hidden, 0.07f, 0.8f)));
        runStrictSingleDispatch(
                "rms_norm rank-2 with gain",
                "rms_norm",
                normalizationInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "rms_norm",
                            sd,
                            variables.get("x"),
                            variables.get("gamma"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> rmsNormOracle(
                        generalValues(step, rows * hidden, 0.16f, -0.7f),
                        generalValues(step + 3, hidden, 0.07f, 0.8f),
                        rows,
                        hidden,
                        epsilon),
                ABS_TOL,
                REL_TOL);

        final List<InputSpec> skipInputs = List.of(
                new InputSpec(
                        "x",
                        rowShape,
                        step -> generalValues(step, rows * hidden, 0.11f, -0.45f)),
                new InputSpec(
                        "skip",
                        rowShape,
                        step -> generalValues(step + 7, rows * hidden, 0.09f, 0.2f)),
                new InputSpec(
                        "gamma",
                        new long[]{hidden},
                        step -> generalValues(step + 3, hidden, 0.07f, 0.8f)));
        runStrictSingleDispatch(
                "skip_rms_norm deterministic one-output form",
                "skip_rms_norm",
                skipInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "skip_rms_norm",
                            sd,
                            variables.get("x"),
                            variables.get("skip"),
                            variables.get("gamma"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> skipRmsNormOracle(
                        generalValues(step, rows * hidden, 0.11f, -0.45f),
                        generalValues(step + 7, rows * hidden, 0.09f, 0.2f),
                        generalValues(step + 3, hidden, 0.07f, 0.8f),
                        rows,
                        hidden,
                        epsilon),
                ABS_TOL,
                REL_TOL);

        final List<InputSpec> linearInputs = List.of(
                normalizationInputs.get(0),
                normalizationInputs.get(1),
                new InputSpec(
                        "weight",
                        new long[]{hidden, projected},
                        step -> generalValues(
                                step + 11, hidden * projected, 0.045f, -0.18f)));
        runStrictSingleDispatch(
                "rms_norm_linear fused normalization and projection",
                "rms_norm_linear",
                linearInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "rms_norm_linear",
                            sd,
                            variables.get("x"),
                            variables.get("gamma"),
                            variables.get("weight"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> rmsNormLinearOracle(
                        generalValues(step, rows * hidden, 0.16f, -0.7f),
                        generalValues(step + 3, hidden, 0.07f, 0.8f),
                        generalValues(
                                step + 11, hidden * projected, 0.045f, -0.18f),
                        rows,
                        hidden,
                        projected,
                        epsilon),
                DataType.FLOAT,
                new long[]{rows, projected},
                ABS_TOL,
                REL_TOL);

        final List<InputSpec> gemmSwiGluInputs = List.of(
                normalizationInputs.get(0),
                new InputSpec(
                        "gateWeight",
                        new long[]{hidden, projected},
                        step -> generalValues(
                                step + 13, hidden * projected, 0.04f, -0.16f)),
                new InputSpec(
                        "upWeight",
                        new long[]{hidden, projected},
                        step -> generalValues(
                                step + 17, hidden * projected, 0.035f, 0.09f)));
        runStrictSingleDispatch(
                "fused_gemm_swiglu two projections",
                "fused_gemm_swiglu",
                gemmSwiGluInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "fused_gemm_swiglu",
                        sd,
                        variables.get("x"),
                        variables.get("gateWeight"),
                        variables.get("upWeight")).outputVariable(),
                step -> fusedGemmSwiGluOracle(
                        generalValues(step, rows * hidden, 0.16f, -0.7f),
                        generalValues(
                                step + 13, hidden * projected, 0.04f, -0.16f),
                        generalValues(
                                step + 17, hidden * projected, 0.035f, 0.09f),
                        rows,
                        hidden,
                        projected),
                DataType.FLOAT,
                new long[]{rows, projected},
                ABS_TOL,
                REL_TOL);

        final List<InputSpec> fusedNormSwiGluInputs = List.of(
                new InputSpec(
                        "x",
                        new long[]{1, rows, hidden},
                        step -> generalValues(step, rows * hidden, 0.16f, -0.7f)),
                normalizationInputs.get(1),
                gemmSwiGluInputs.get(1),
                gemmSwiGluInputs.get(2));
        runStrictSingleDispatch(
                "fused_rms_norm_swiglu rank-3",
                "fused_rms_norm_swiglu",
                fusedNormSwiGluInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "fused_rms_norm_swiglu",
                            sd,
                            variables.get("x"),
                            variables.get("gamma"),
                            variables.get("gateWeight"),
                            variables.get("upWeight"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> fusedRmsNormSwiGluOracle(
                        generalValues(step, rows * hidden, 0.16f, -0.7f),
                        generalValues(step + 3, hidden, 0.07f, 0.8f),
                        generalValues(
                                step + 13, hidden * projected, 0.04f, -0.16f),
                        generalValues(
                                step + 17, hidden * projected, 0.035f, 0.09f),
                        rows,
                        hidden,
                        projected,
                        epsilon),
                DataType.FLOAT,
                new long[]{1, rows, projected},
                ABS_TOL,
                REL_TOL);

        final int sequence = 2;
        final int heads = 2;
        final int headDimension = 4;
        final int cacheLength = sequence * (headDimension / 2);
        final List<InputSpec> cachedRopeInputs = List.of(
                new InputSpec(
                        "x",
                        new long[]{1, sequence, heads, headDimension},
                        step -> generalValues(
                                step,
                                sequence * heads * headDimension,
                                0.08f,
                                -0.65f)),
                new InputSpec(
                        "cos",
                        new long[]{sequence, headDimension / 2},
                        step -> rotaryCacheValues(step, cacheLength, true)),
                new InputSpec(
                        "sin",
                        new long[]{sequence, headDimension / 2},
                        step -> rotaryCacheValues(step, cacheLength, false)));
        runStrictSingleDispatch(
                "fused_rope cached adjacent-pair form",
                "fused_rope",
                cachedRopeInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "fused_rope",
                            sd,
                            variables.get("x"),
                            variables.get("cos"),
                            variables.get("sin"));
                    op.addIArgument(1, 0, 0);
                    return op.outputVariable();
                },
                step -> cachedRopeOracle(
                        generalValues(
                                step,
                                sequence * heads * headDimension,
                                0.08f,
                                -0.65f),
                        rotaryCacheValues(step, cacheLength, true),
                        rotaryCacheValues(step, cacheLength, false),
                        1,
                        sequence,
                        heads,
                        headDimension),
                ABS_TOL,
                REL_TOL);

        final int queryFeatures = 3;
        final int querySteps = 2;
        final int keySteps = 3;
        final int valueFeatures = 2;
        final List<InputSpec> attentionInputs = List.of(
                new InputSpec(
                        "query",
                        new long[]{1, queryFeatures, querySteps},
                        step -> generalValues(
                                step, queryFeatures * querySteps, 0.09f, -0.3f)),
                new InputSpec(
                        "key",
                        new long[]{1, queryFeatures, keySteps},
                        step -> generalValues(
                                step + 5, queryFeatures * keySteps, 0.07f, -0.25f)),
                new InputSpec(
                        "value",
                        new long[]{1, valueFeatures, keySteps},
                        step -> generalValues(
                                step + 9, valueFeatures * keySteps, 0.06f, 0.15f)));
        runStrictSingleDispatch(
                "dot_product_attention normalized rank-3",
                "dot_product_attention",
                attentionInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "dot_product_attention",
                            sd,
                            variables.get("query"),
                            variables.get("key"),
                            variables.get("value"));
                    op.addIArgument(1, 0);
                    return op.outputVariable();
                },
                step -> dotProductAttentionOracle(
                        generalValues(
                                step, queryFeatures * querySteps, 0.09f, -0.3f),
                        generalValues(
                                step + 5, queryFeatures * keySteps, 0.07f, -0.25f),
                        generalValues(
                                step + 9, valueFeatures * keySteps, 0.06f, 0.15f),
                        1,
                        queryFeatures,
                        querySteps,
                        keySteps,
                        valueFeatures,
                        true),
                DataType.FLOAT,
                new long[]{1, valueFeatures, querySteps},
                ABS_TOL,
                REL_TOL);
    }

    @Test
    @DisplayName("cast/normalization/movement: compute descriptors dispatch and views replay as aliases")
    void remainingNormalizationAndMovementDescriptorsReplayWithExpectedDispatches() {
        final long[] castShape = {2, 3};
        final List<InputSpec> castInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.INT32,
                        castShape,
                        step -> signedIntegerValues(step, 6)));
        runStrictSingleDispatch(
                "cast INT32 to FLOAT",
                "cast",
                castInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "cast", DataType.FLOAT, sd, variables.get("x"));
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> signedIntegerValues(step, 6),
                DataType.FLOAT,
                castShape,
                0.0f,
                0.0f);

        final int rows = 3;
        final int width = 4;
        final long[] normalizedShape = {rows, width};
        final List<InputSpec> softmaxInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        normalizedShape,
                        step -> generalValues(step, rows * width, 0.41f, -1.3f)));
        runStrictSingleDispatch(
                "softmax rank-2 axis=1",
                "softmax",
                softmaxInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "softmax", sd, variables.get("x"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> mapValues(
                        logSoftmaxOracle(
                                generalValues(step, rows * width, 0.41f, -1.3f),
                                rows,
                                width),
                        Math::exp),
                ABS_TOL,
                REL_TOL);

        final double epsilon = 1.0e-4;
        final List<InputSpec> layerNormInputs = List.of(
                new InputSpec(
                        "x",
                        normalizedShape,
                        step -> generalValues(step, rows * width, 0.19f, -0.8f)),
                new InputSpec(
                        "gain",
                        new long[]{width},
                        step -> generalValues(step + 5, width, 0.08f, 0.75f)),
                new InputSpec(
                        "bias",
                        new long[]{width},
                        step -> generalValues(step + 9, width, 0.05f, -0.12f)));
        runStrictSingleDispatch(
                "layer_norm explicit last axis",
                "layer_norm",
                layerNormInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "layer_norm",
                            sd,
                            variables.get("x"),
                            variables.get("gain"),
                            variables.get("bias"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> layerNormOracle(
                        generalValues(step, rows * width, 0.19f, -0.8f),
                        generalValues(step + 5, width, 0.08f, 0.75f),
                        generalValues(step + 9, width, 0.05f, -0.12f),
                        rows,
                        width,
                        1.0e-5),
                ABS_TOL,
                REL_TOL);
        runStrictSingleDispatch(
                "fused_layer_norm implicit last axis",
                "fused_layer_norm",
                layerNormInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "fused_layer_norm",
                            sd,
                            variables.get("x"),
                            variables.get("gain"),
                            variables.get("bias"));
                    op.addTArgument(epsilon);
                    return op.outputVariable();
                },
                step -> layerNormOracle(
                        generalValues(step, rows * width, 0.19f, -0.8f),
                        generalValues(step + 5, width, 0.08f, 0.75f),
                        generalValues(step + 9, width, 0.05f, -0.12f),
                        rows,
                        width,
                        epsilon),
                ABS_TOL,
                REL_TOL);

        final int gatherRows = 4;
        final int gatherWidth = 3;
        final long[] gatherOutputShape = {2, gatherWidth};
        final List<InputSpec> gatherInputs = List.of(
                new InputSpec(
                        "x",
                        new long[]{gatherRows, gatherWidth},
                        step -> generalValues(
                                step, gatherRows * gatherWidth, 0.13f, -0.55f)),
                new InputSpec(
                        "indices",
                        DataType.INT32,
                        new long[]{2},
                        step -> new float[]{3.0f, 1.0f}));
        runStrictSingleDispatch(
                "gather static axis zero",
                "gather",
                gatherInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "gather",
                            sd,
                            variables.get("x"),
                            variables.get("indices"));
                    op.addIArgument(0);
                    return op.outputVariable();
                },
                step -> gatherAxisZeroOracle(
                        generalValues(
                                step, gatherRows * gatherWidth, 0.13f, -0.55f),
                        new int[]{3, 1},
                        gatherRows,
                        gatherWidth),
                DataType.FLOAT,
                gatherOutputShape,
                0.0f,
                0.0f);

        final List<InputSpec> concatInputs = List.of(
                new InputSpec(
                        "left",
                        new long[]{2, 2},
                        step -> generalValues(step, 4, 0.17f, -0.4f)),
                new InputSpec(
                        "right",
                        new long[]{2, 1},
                        step -> generalValues(step + 3, 2, 0.09f, 0.6f)));
        runStrictSingleDispatch(
                "concat static axis one",
                "concat",
                concatInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "concat",
                            sd,
                            variables.get("left"),
                            variables.get("right"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> concatAxisOneOracle(
                        generalValues(step, 4, 0.17f, -0.4f),
                        generalValues(step + 3, 2, 0.09f, 0.6f),
                        2,
                        2,
                        1),
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final long[] permutationInputShape = {2, 3, 4};
        final List<InputSpec> permutationInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        permutationInputShape,
                        step -> generalValues(step, 24, 0.07f, -0.9f)));
        runStrictAliasReplay(
                "transpose default reverse axes",
                "transpose",
                permutationInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "transpose", sd, variables.get("x")).outputVariable(),
                step -> permuteOracle(
                        generalValues(step, 24, 0.07f, -0.9f),
                        permutationInputShape,
                        new int[]{2, 1, 0}),
                DataType.FLOAT,
                new long[]{4, 3, 2},
                0.0f,
                0.0f);
        runStrictAliasReplay(
                "permute explicit axes",
                "permute",
                permutationInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "permute", sd, variables.get("x"));
                    op.addIArgument(1, 2, 0);
                    return op.outputVariable();
                },
                step -> permuteOracle(
                        generalValues(step, 24, 0.07f, -0.9f),
                        permutationInputShape,
                        new int[]{1, 2, 0}),
                DataType.FLOAT,
                new long[]{3, 4, 2},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("reduce_norm1/reduce_norm2: INT32 multi-axis reductions produce FLOAT")
    void normReductionsInt32AxesAndKeepDimsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 4;
        final long[] inputShape = {batch, rows, width};
        final long[] axes = {0, 2};
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.INT32,
                        inputShape,
                        step -> signedIntegerValues(step, batch * rows * width)));

        runStrictSingleDispatch(
                "reduce_norm1 INT32 axes=[0,2] keepDims=true",
                "reduce_norm1",
                inputs,
                (sd, variables) -> new Norm1(
                        sd, variables.get("x"), true, axes).outputVariable(),
                step -> rankThreeNormOracle(
                        signedIntegerValues(step, batch * rows * width),
                        batch,
                        rows,
                        width,
                        false),
                DataType.FLOAT,
                new long[]{1, rows, 1},
                0.0f,
                0.0f);

        runStrictSingleDispatch(
                "reduce_norm2 INT32 axes=[0,2] keepDims=false",
                "reduce_norm2",
                inputs,
                (sd, variables) -> new Norm2(
                        sd, variables.get("x"), false, axes).outputVariable(),
                step -> rankThreeNormOracle(
                        signedIntegerValues(step, batch * rows * width),
                        batch,
                        rows,
                        width,
                        true),
                DataType.FLOAT,
                new long[]{rows},
                2.0e-5f,
                2.0e-5f);
    }

    @Test
    @DisplayName("flash/GQA: rank-4 explicit-scale causal attention replays one pipeline")
    void flashAndGroupedQueryAttentionCreateSinglePipelinesAndReplay() {
        runRankFourAttentionCase(false);
        runRankFourAttentionCase(true);
    }

    private void runRankFourAttentionCase(boolean grouped) {
        final int batch = 2;
        final int querySteps = 4;
        final int keySteps = 4;
        final int queryHeads = 4;
        final int keyValueHeads = grouped ? 2 : queryHeads;
        final int headDimension = 3;
        final double scale = 0.375;
        final boolean causal = true;
        final long[] queryShape = {
                batch, querySteps, queryHeads, headDimension};
        final long[] keyValueShape = {
                batch, keySteps, keyValueHeads, headDimension};
        final int queryLength =
                batch * querySteps * queryHeads * headDimension;
        final int keyValueLength =
                batch * keySteps * keyValueHeads * headDimension;
        final String opName =
                grouped ? "grouped_query_attention" : "flash_attention";

        final List<InputSpec> inputs = List.of(
                new InputSpec("query", queryShape,
                        step -> generalValues(step, queryLength, 0.055f, -0.62f)),
                new InputSpec("key", keyValueShape,
                        step -> generalValues(step + 5, keyValueLength, 0.045f, -0.48f)),
                new InputSpec("value", keyValueShape,
                        step -> generalValues(step + 13, keyValueLength, 0.075f, -0.91f)));

        runStrictSingleDispatch(
                opName + " rank-4 qHeads=" + queryHeads
                        + " kvHeads=" + keyValueHeads
                        + " scale=" + scale + " causal=true",
                opName,
                inputs,
                (sd, variables) -> grouped
                        ? new GroupedQueryAttention(
                                sd,
                                variables.get("query"),
                                variables.get("key"),
                                variables.get("value"),
                                scale,
                                causal,
                                queryHeads,
                                keyValueHeads).outputVariable()
                        : new FlashAttention(
                                sd,
                                variables.get("query"),
                                variables.get("key"),
                                variables.get("value"),
                                scale,
                                causal,
                                queryHeads,
                                keyValueHeads).outputVariable(),
                step -> groupedAttentionOracle(
                        generalValues(step, queryLength, 0.055f, -0.62f),
                        generalValues(step + 5, keyValueLength, 0.045f, -0.48f),
                        generalValues(step + 13, keyValueLength, 0.075f, -0.91f),
                        batch,
                        querySteps,
                        keySteps,
                        queryHeads,
                        keyValueHeads,
                        headDimension,
                        scale,
                        causal),
                DataType.FLOAT,
                queryShape,
                8.0e-4f,
                8.0e-4f);
    }

    @Test
    @DisplayName("rope: standard/NeoX partial rotation preserves a feature-strided rank-3 view")
    void ropeModesTailAndFeatureStridesCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int sequence = 3;
        final int headDimension = 8;
        final int rotaryDimensions = 6;
        final int positionOffset = 5;
        final double frequencyBase = 1000.0;
        final double frequencyScale = 0.75;
        final long[] shape = {batch, sequence, headDimension};
        final int length = batch * sequence * headDimension;

        for (int mode : new int[]{0, 1}) {
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec(
                            "x",
                            DataType.FLOAT,
                            shape,
                            step -> featureStridedInput(
                                    step, batch, sequence, headDimension),
                            true));
            runStrictSingleDispatch(
                    "rope mode=" + mode + " rank-3 feature-stride=2 partial-tail",
                    "rope",
                    inputs,
                    (sd, variables) -> new RoPE(
                            sd,
                            variables.get("x"),
                            mode,
                            positionOffset,
                            rotaryDimensions,
                            frequencyBase,
                            frequencyScale).outputVariable(),
                    step -> ropeOracle(
                            generalValues(step, length, 0.0375f, -0.65f),
                            batch,
                            sequence,
                            1,
                            headDimension,
                            mode,
                            positionOffset,
                            rotaryDimensions,
                            frequencyBase,
                            frequencyScale),
                    DataType.FLOAT,
                    shape,
                    4.0e-4f,
                    4.0e-4f);
        }

        final List<InputSpec> identityInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3, 1},
                        step -> generalValues(step, 6, 0.2f, -0.5f)));
        runStrictSingleDispatch(
                "rope width-1 identity",
                "rope",
                identityInputs,
                (sd, variables) -> new RoPE(
                        sd, variables.get("x"), 0, 3, 0,
                        frequencyBase, frequencyScale).outputVariable(),
                step -> generalValues(step, 6, 0.2f, -0.5f),
                DataType.FLOAT,
                new long[]{2, 3, 1},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("rope_bp/fused_rope_bp: inverse rotation replays one strided rank-3 pipeline")
    void ropeBackwardVariantsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int sequence = 3;
        final int headDimension = 8;
        final int rotaryDimensions = 6;
        final int positionOffset = 4;
        final double frequencyBase = 2000.0;
        final double frequencyScale = 0.625;
        final long[] shape = {batch, sequence, headDimension};
        final int length = batch * sequence * headDimension;

        for (String opName : new String[]{"rope_bp", "fused_rope_bp"}) {
            for (int mode : new int[]{0, 1}) {
                final List<InputSpec> inputs = List.of(
                        new InputSpec(
                                "x",
                                DataType.FLOAT,
                                shape,
                                step -> featureStridedInput(
                                        step, batch, sequence, headDimension),
                                true),
                        new InputSpec(
                                "gradient",
                                DataType.FLOAT,
                                shape,
                                step -> featureStridedInput(
                                        step + 9,
                                        batch,
                                        sequence,
                                        headDimension),
                                true));
                runStrictSingleDispatch(
                        opName + " mode=" + mode
                                + " rank-3 feature-stride=2 partial-tail",
                        opName,
                        inputs,
                        (sd, variables) -> {
                            NamedDynamicOp op = new NamedDynamicOp(
                                    opName,
                                    sd,
                                    variables.get("x"),
                                    variables.get("gradient"));
                            op.addIArgument(
                                    mode, positionOffset, rotaryDimensions);
                            op.addTArgument(frequencyBase, frequencyScale);
                            return op.outputVariable();
                        },
                        step -> ropeBackwardOracle(
                                generalValues(
                                        step + 9,
                                        length,
                                        0.0375f,
                                        -0.65f),
                                batch,
                                sequence,
                                1,
                                headDimension,
                                mode,
                                positionOffset,
                                rotaryDimensions,
                                frequencyBase,
                                frequencyScale),
                        DataType.FLOAT,
                        shape,
                        4.0e-4f,
                        4.0e-4f);
            }
        }
    }

    @Test
    @DisplayName("apply_alibi: exact per-head slopes replay one rank-4 pipeline")
    void applyAlibiCreatesSinglePipelineAndReplay() {
        final int batch = 2;
        final int heads = 3;
        final int queries = 4;
        final int keys = 5;
        final int length = batch * heads * queries * keys;
        final long[] shape = {batch, heads, queries, keys};
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec("scores", shape,
                        step -> generalValues(step, length, 0.04f, -0.35f)));

        runStrictSingleDispatch(
                "apply_alibi heads=3 rank-4",
                "apply_alibi",
                inputs,
                (sd, variables) -> new ApplyAlibi(
                        sd, variables.get("scores"), heads).outputVariable(),
                step -> alibiOracle(
                        generalValues(step, length, 0.04f, -0.35f),
                        batch, heads, queries, keys),
                DataType.FLOAT,
                shape,
                3.0e-5f,
                3.0e-5f);
    }

    @Test
    @DisplayName("vision_embedding_merge: INT32 tokens select per-batch vision prefixes in one pipeline")
    void visionEmbeddingMergeCreatesSinglePipelineAndReplay() {
        final int batch = 2;
        final int sequence = 5;
        final int hidden = 4;
        final int visionTokens = 3;
        final int targetTokenId = 99;
        final long[] outputShape = {batch, sequence, hidden};
        final List<InputSpec> inputs = List.of(
                new InputSpec("text", outputShape,
                        step -> generalValues(step, batch * sequence * hidden,
                                0.055f, -0.75f)),
                new InputSpec("vision", new long[]{batch, visionTokens, hidden},
                        step -> generalValues(step + 7, batch * visionTokens * hidden,
                                0.075f, 0.35f)),
                new InputSpec("tokens", DataType.INT32,
                        new long[]{batch, sequence},
                        step -> visionTokenIds(step, batch, sequence, targetTokenId)));

        runStrictSingleDispatch(
                "vision_embedding_merge target=99",
                "vision_embedding_merge",
                inputs,
                (sd, variables) -> new VisionEmbeddingMerge(
                        sd,
                        variables.get("text"),
                        variables.get("vision"),
                        variables.get("tokens"),
                        targetTokenId).outputVariable(),
                step -> visionMergeOracle(
                        generalValues(step, batch * sequence * hidden,
                                0.055f, -0.75f),
                        generalValues(step + 7, batch * visionTokens * hidden,
                                0.075f, 0.35f),
                        visionTokenIds(step, batch, sequence, targetTokenId),
                        batch, sequence, hidden, visionTokens, targetTokenId),
                DataType.FLOAT,
                outputShape,
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("logsumexp/variance/stdev: rank-N reductions replay one AccT pipeline")
    void statisticalReductionsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int middle = 3;
        final int width = 4;
        final int length = batch * middle * width;
        final long[] inputShape = {batch, middle, width};

        final List<InputSpec> logSumExpInputs = Collections.singletonList(
                new InputSpec("x", inputShape,
                        step -> generalValues(step, length, 0.11f, -0.85f)));
        runStrictSingleDispatch(
                "reduce_logsumexp axes=[0,2] keepDims=true",
                "reduce_logsumexp",
                logSumExpInputs,
                (sd, variables) -> new LogSumExp(
                        sd, variables.get("x"), true, new long[]{0, 2})
                        .outputVariable(),
                step -> logSumExpAxesZeroTwoOracle(
                        generalValues(step, length, 0.11f, -0.85f),
                        batch, middle, width),
                DataType.FLOAT,
                new long[]{1, middle, 1},
                5.0e-5f,
                5.0e-5f);

        final List<InputSpec> varianceInputs = Collections.singletonList(
                new InputSpec("x", DataType.INT32, inputShape,
                        step -> integerValues(step, length)));
        runStrictSingleDispatch(
                "reduce_variance INT32 axes=[0,2] biasCorrected=true",
                "reduce_variance",
                varianceInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "reduce_variance", DataType.FLOAT,
                            sd, variables.get("x"));
                    op.addIArgument(0, 2);
                    op.addBArgument(false, true);
                    return op.outputVariable();
                },
                step -> statisticalAxesZeroTwoOracle(
                        integerValues(step, length), batch, middle, width,
                        true, false),
                DataType.FLOAT,
                new long[]{middle},
                2.0e-5f,
                2.0e-5f);

        final List<InputSpec> stdevInputs = Collections.singletonList(
                new InputSpec("x", DataType.INT32, inputShape,
                        step -> integerValues(step + 5, length)));
        runStrictSingleDispatch(
                "reduce_stdev INT32 axes=[0,2] keepDims=true",
                "reduce_stdev",
                stdevInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "reduce_stdev", DataType.FLOAT,
                            sd, variables.get("x"));
                    op.addIArgument(0, 2);
                    op.addBArgument(true, false);
                    return op.outputVariable();
                },
                step -> statisticalAxesZeroTwoOracle(
                        integerValues(step + 5, length), batch, middle, width,
                        false, true),
                DataType.FLOAT,
                new long[]{1, middle, 1},
                2.0e-5f,
                2.0e-5f);
    }

    @Test
    @DisplayName("sum/mean/max/min/prod: shared reduction traits cover integer, strided, F-order, and full axes")
    void baseReductionsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 5;
        final int length = batch * rows * width;
        final long[] inputShape = {batch, rows, width};
        final long[] outputShape = {batch, rows};

        for (String opName : new String[]{
                "reduce_sum", "reduce_mean", "reduce_max",
                "reduce_min", "reduce_prod"}) {
            final DataType dataType = opName.equals("reduce_mean")
                    ? DataType.FLOAT
                    : DataType.INT32;
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec(
                            "x",
                            dataType,
                            inputShape,
                            step -> transformedReductionStridedInput(
                                    dataType, step, batch, rows, width),
                            true));
            runStrictSingleDispatch(
                    opName + " " + dataType
                            + " feature-stride=2 axis=-1",
                    opName,
                    inputs,
                    (sd, variables) -> {
                        NamedDynamicOp op = new NamedDynamicOp(
                                opName, dataType, sd, variables.get("x"));
                        op.addIArgument(-1);
                        op.addBArgument(false);
                        return op.outputVariable();
                    },
                    step -> baseLastAxisReductionOracle(
                            transformedReductionValues(dataType, step, length),
                            batch * rows,
                            width,
                            opName),
                    dataType,
                    outputShape,
                    dataType == DataType.FLOAT ? 1.0e-5f : 0.0f,
                    dataType == DataType.FLOAT ? 1.0e-5f : 0.0f);
        }

        final List<InputSpec> fullReductionInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        inputShape,
                        step -> denseFortranInput(
                                transformedReductionValues(
                                        DataType.FLOAT, step + 9, length),
                                inputShape),
                        true));
        runStrictSingleDispatch(
                "reduce_sum dense-F full-axis scalar output",
                "reduce_sum",
                fullReductionInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "reduce_sum", DataType.FLOAT,
                            sd, variables.get("x"));
                    op.addBArgument(false);
                    return op.outputVariable();
                },
                step -> baseLastAxisReductionOracle(
                        transformedReductionValues(
                                DataType.FLOAT, step + 9, length),
                        1,
                        length,
                        "reduce_sum"),
                DataType.FLOAT,
                new long[0],
                1.0e-5f,
                1.0e-5f);
    }

    @Test
    @DisplayName("argmax/argmin: trait-selected index reductions preserve strided views and first ties")
    void indexReductionsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 5;
        final long[] inputShape = {batch, rows, width};
        final long[] outputShape = {batch, rows};

        for (String opName : new String[]{"argmax", "argmin"}) {
            final boolean maximum = opName.equals("argmax");
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec(
                            "x",
                            DataType.FLOAT,
                            inputShape,
                            step -> argReductionStridedInput(
                                    step, batch, rows, width),
                            true));
            runStrictSingleDispatch(
                    opName + " FLOAT feature-stride=2 axis=-1 INT32 output",
                    opName,
                    inputs,
                    (sd, variables) -> {
                        NamedDynamicOp op = new NamedDynamicOp(
                                opName, DataType.INT32,
                                sd, variables.get("x"));
                        op.addIArgument(-1);
                        op.addBArgument(false);
                        op.addDArgument(DataType.INT32);
                        return op.outputVariable();
                    },
                    step -> argReductionOracle(
                            argReductionValues(step, batch, rows, width),
                            batch * rows, width, maximum),
                    DataType.INT32,
                    outputShape,
                    0.0f,
                    0.0f);
        }
    }

    @Test
    @DisplayName("trait-composed square/absolute reductions replay one strided pipeline")
    void transformedReductionsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 5;
        final int length = batch * rows * width;
        final long[] inputShape = {batch, rows, width};
        final long[] outputShape = {batch, rows};

        for (DataType inputDataType : new DataType[]{DataType.FLOAT, DataType.INT32}) {
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec(
                            "x",
                            inputDataType,
                            inputShape,
                            step -> transformedReductionStridedInput(
                                    inputDataType, step, batch, rows, width),
                            true));
            for (String opName : new String[]{"reduce_sqnorm", "reduce_norm_max"}) {
                final boolean squareAndSum = opName.equals("reduce_sqnorm");
                runStrictSingleDispatch(
                        opName + " " + inputDataType
                                + " feature-stride=2 axis=-1 FLOAT output",
                        opName,
                        inputs,
                        (sd, variables) -> {
                            NamedDynamicOp op = new NamedDynamicOp(
                                    opName, DataType.FLOAT,
                                    sd, variables.get("x"));
                            op.addIArgument(-1);
                            op.addBArgument(false);
                            if (squareAndSum) {
                                op.addDArgument(DataType.FLOAT);
                            }
                            return op.outputVariable();
                        },
                        step -> transformedLastAxisReductionOracle(
                                transformedReductionValues(
                                        inputDataType, step, length),
                                batch * rows,
                                width,
                                squareAndSum),
                        DataType.FLOAT,
                        outputShape,
                        1.0e-4f,
                        1.0e-4f);
            }
        }
    }

    @Test
    @DisplayName("argamax/argamin: absolute-input trait covers FLOAT and signed INT32 views")
    void absoluteIndexReductionsCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int rows = 3;
        final int width = 7;
        final long[] inputShape = {batch, rows, width};
        final long[] outputShape = {batch, rows};

        for (DataType inputDataType : new DataType[]{DataType.FLOAT, DataType.INT32}) {
            final List<InputSpec> inputs = Collections.singletonList(
                    new InputSpec(
                            "x",
                            inputDataType,
                            inputShape,
                            step -> absoluteArgReductionStridedInput(
                                    inputDataType, step, batch, rows, width),
                            true));
            for (String opName : new String[]{"argamax", "argamin"}) {
                final boolean maximum = opName.equals("argamax");
                runStrictSingleDispatch(
                        opName + " " + inputDataType
                                + " feature-stride=2 axis=-1 INT32 output",
                        opName,
                        inputs,
                        (sd, variables) -> {
                            NamedDynamicOp op = new NamedDynamicOp(
                                    opName, DataType.INT32,
                                    sd, variables.get("x"));
                            op.addIArgument(-1);
                            op.addBArgument(false);
                            op.addDArgument(DataType.INT32);
                            return op.outputVariable();
                        },
                        step -> absoluteArgReductionOracle(
                                absoluteArgReductionValues(
                                        inputDataType, step, batch, rows, width),
                                batch * rows,
                                width,
                                maximum),
                        DataType.INT32,
                        outputShape,
                        0.0f,
                        0.0f);
            }
        }
    }

    @Test
    @DisplayName("identity aliases; stop_gradient/identity_bp copy their selected input")
    void sameShapeCopySchedulesCreateSinglePipelinesAndReplay() {
        final long[] shape = {2, 3, 4};
        final int length = 24;
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        shape,
                        step -> transformedReductionStridedInput(
                                DataType.FLOAT, step, 2, 3, 4),
                        true));

        runStrictAliasReplay(
                "identity rank-3 feature-stride=2",
                "identity",
                inputs,
                (sd, variables) -> new NamedDynamicOp(
                        "identity", sd, variables.get("x")).outputVariable(),
                step -> transformedReductionValues(
                        DataType.FLOAT, step, length),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        runStrictSingleDispatch(
                "stop_gradient rank-3 feature-stride=2",
                "stop_gradient",
                inputs,
                (sd, variables) -> new NamedDynamicOp(
                        "stop_gradient", sd, variables.get("x")).outputVariable(),
                step -> transformedReductionValues(
                        DataType.FLOAT, step, length),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final List<InputSpec> backwardInputs = List.of(
                new InputSpec("activation", shape,
                        step -> generalValues(step + 31, length, 0.13f, -2.0f)),
                new InputSpec(
                        "gradient",
                        DataType.FLOAT,
                        shape,
                        step -> transformedReductionStridedInput(
                                DataType.FLOAT, step + 7, 2, 3, 4),
                        true));
        runStrictSingleDispatch(
                "identity_bp follows trailing gradient payload",
                "identity_bp",
                backwardInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "identity_bp",
                        sd,
                        variables.get("activation"),
                        variables.get("gradient")).outputVariable(),
                step -> transformedReductionValues(
                        DataType.FLOAT, step + 7, length),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("identity_n: heterogeneous shapes copy every destination in one dispatch")
    void identityNMultiDestinationCopyCreatesSinglePipelineAndReplays() {
        final long[] leftShape = {2, 3};
        final long[] rightShape = {2, 2, 2};
        final List<InputSpec> inputs = List.of(
                new InputSpec(
                        "left",
                        leftShape,
                        step -> generalValues(step, 6, 0.17f, -0.9f)),
                new InputSpec(
                        "right",
                        DataType.FLOAT,
                        rightShape,
                        step -> transformedReductionStridedInput(
                                DataType.FLOAT, step + 11, 2, 2, 2),
                        true));

        runStrictMultiOutputSingleDispatch(
                "identity_n two differently-shaped destinations",
                "identity_n",
                inputs,
                (sd, variables) -> new NamedFixedOutputOp(
                        "identity_n",
                        2,
                        sd,
                        variables.get("left"),
                        variables.get("right")).outputVariables(),
                List.of(
                        new OutputSpec(
                                DataType.FLOAT,
                                leftShape,
                                step -> generalValues(
                                        step, 6, 0.17f, -0.9f),
                                0.0f,
                                0.0f),
                        new OutputSpec(
                                DataType.FLOAT,
                                rightShape,
                                step -> transformedReductionValues(
                                        DataType.FLOAT, step + 11, 8),
                                0.0f,
                                0.0f)));
    }

    @Test
    @DisplayName("linear_copy: structural shape input drives dense C/F raw-copy pipelines")
    void structuralLinearCopyCreatesSinglePipelinesAndReplay() {
        final long[] inputShape = {2, 3, 4};
        final int inputLength = 24;

        final List<InputSpec> truncatingInputs = List.of(
                new InputSpec(
                        "x",
                        inputShape,
                        step -> generalValues(step, inputLength, 0.11f, -0.8f)),
                new InputSpec(
                        "shape",
                        DataType.INT32,
                        new long[]{2},
                        step -> new float[]{3.0f, 4.0f}));
        runStrictSingleDispatch(
                "linear_copy dense-C rank-change truncation",
                "linear_copy",
                truncatingInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "linear_copy",
                        sd,
                        variables.get("x"),
                        variables.get("shape")).outputVariable(),
                step -> {
                    float[] source =
                            generalValues(step, inputLength, 0.11f, -0.8f);
                    float[] output = new float[12];
                    System.arraycopy(source, 0, output, 0, output.length);
                    return output;
                },
                DataType.FLOAT,
                new long[]{3, 4},
                0.0f,
                0.0f);

        final long[] fortranOutputShape = {4, 6};
        final List<InputSpec> fortranInputs = List.of(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        inputShape,
                        step -> denseFortranInput(
                                transformedReductionValues(
                                        DataType.FLOAT, step + 5, inputLength),
                                inputShape),
                        true),
                new InputSpec(
                        "shape",
                        DataType.INT32,
                        new long[]{2},
                        step -> new float[]{4.0f, 6.0f}));
        runStrictSingleDispatch(
                "linear_copy dense-F rank-change full copy",
                "linear_copy",
                fortranInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "linear_copy",
                        sd,
                        variables.get("x"),
                        variables.get("shape")).outputVariable(),
                step -> linearCopyFortranOracle(
                        transformedReductionValues(
                                DataType.FLOAT, step + 5, inputLength),
                        inputShape,
                        fortranOutputShape),
                DataType.FLOAT,
                fortranOutputShape,
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("biasadd/prelu/batchnorm: broadcast normalization replays one pipeline")
    void broadcastNormalizationCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int channels = 3;
        final int width = 4;
        final int length = batch * channels * width;
        final long[] shape = {batch, channels, width};

        final List<InputSpec> biasAddInputs = List.of(
                new InputSpec("x", DataType.INT32, shape,
                        step -> integerValues(step, length)),
                new InputSpec("bias", new long[]{channels},
                        step -> generalValues(step + 3, channels, 0.17f, -0.3f)));
        runStrictSingleDispatch(
                "biasadd INT32 input FLOAT bias NCHW",
                "biasadd",
                biasAddInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "biasadd", DataType.FLOAT, sd,
                            variables.get("x"), variables.get("bias"));
                    op.addBArgument(true);
                    return op.outputVariable();
                },
                step -> biasAddNchwOracle(
                        integerValues(step, length),
                        generalValues(step + 3, channels, 0.17f, -0.3f),
                        batch, channels, width),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final List<InputSpec> preluInputs = List.of(
                new InputSpec("x", shape,
                        step -> generalValues(step, length, 0.19f, -1.25f)),
                new InputSpec("alpha", new long[]{channels, width},
                        step -> generalValues(step + 11, channels * width,
                                0.025f, 0.05f)));
        runStrictSingleDispatch(
                "prelu unshared rank-3 alpha=[3,4]",
                "prelu",
                preluInputs,
                (sd, variables) -> new PRelu(
                        sd, variables.get("x"), variables.get("alpha"))
                        .outputVariable(),
                step -> preluOracle(
                        generalValues(step, length, 0.19f, -1.25f),
                        generalValues(step + 11, channels * width,
                                0.025f, 0.05f),
                        batch, channels, width),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final double epsilon = 1.0e-4;
        final List<InputSpec> batchNormInputs = List.of(
                new InputSpec("x", shape,
                        step -> generalValues(step, length, 0.08f, -0.6f)),
                new InputSpec("mean", new long[]{width},
                        step -> generalValues(step + 2, width, 0.04f, -0.1f)),
                new InputSpec("variance", new long[]{width},
                        step -> positiveVarianceValues(step, width)),
                new InputSpec("gamma", new long[]{width},
                        step -> generalValues(step + 5, width, 0.03f, 0.8f)),
                new InputSpec("beta", new long[]{width},
                        step -> generalValues(step + 7, width, 0.025f, -0.2f)));
        runStrictSingleDispatch(
                "batchnorm axis=2 scale+offset",
                "batchnorm",
                batchNormInputs,
                (sd, variables) -> new BatchNorm(
                        sd,
                        variables.get("x"),
                        variables.get("mean"),
                        variables.get("variance"),
                        variables.get("gamma"),
                        variables.get("beta"),
                        epsilon,
                        new int[]{2}).outputVariable(),
                step -> batchNormOracle(
                        generalValues(step, length, 0.08f, -0.6f),
                        generalValues(step + 2, width, 0.04f, -0.1f),
                        positiveVarianceValues(step, width),
                        generalValues(step + 5, width, 0.03f, 0.8f),
                        generalValues(step + 7, width, 0.025f, -0.2f),
                        batch, channels, width, epsilon),
                DataType.FLOAT,
                shape,
                2.0e-4f,
                2.0e-4f);
    }

    @Test
    @DisplayName("fill_as: frozen scalar fills strided and integer tensors in one pipeline")
    void fillAsCreatesSinglePipelineAndReplay() {
        final long[] shape = {2, 3, 4};
        final int length = 24;
        final double floatingValue = 2.75;
        final List<InputSpec> stridedFloatInput = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        shape,
                        step -> featureStridedInput(step, 2, 3, 4),
                        true));
        runStrictSingleDispatch(
                "fill_as FLOAT feature-strided donor",
                "fill_as",
                stridedFloatInput,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "fill_as", sd, variables.get("x"));
                    op.addTArgument(floatingValue);
                    return op.outputVariable();
                },
                step -> constantValues(length, (float) floatingValue),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final double integerValue = -7.0;
        final List<InputSpec> integerInput = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.INT32,
                        shape,
                        step -> integerValues(step, length)));
        runStrictSingleDispatch(
                "fill_as INT32 donor with TArg conversion",
                "fill_as",
                integerInput,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "fill_as", sd, variables.get("x"));
                    op.addTArgument(integerValue);
                    return op.outputVariable();
                },
                step -> constantValues(length, (float) integerValue),
                DataType.INT32,
                shape,
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("zeros_as/ones_as/triu/triu_bp: generated and triangular tensors replay real pipelines")
    void generatedAndTriangularTensorsCreateSinglePipelinesAndReplay() {
        final long[] shape = {2, 3, 4};
        final int length = 24;
        final List<InputSpec> integerInput = Collections.singletonList(
                new InputSpec("x", DataType.INT32, shape,
                        step -> integerValues(step, length)));

        runStrictSingleDispatch(
                "zeros_as INT32 rank-3",
                "zeros_as",
                integerInput,
                (sd, variables) -> new NamedDynamicOp(
                        "zeros_as", sd, variables.get("x"))
                        .outputVariable(),
                step -> constantValues(length, 0.0f),
                DataType.INT32,
                shape,
                0.0f,
                0.0f);

        runStrictSingleDispatch(
                "ones_as INT32 shape donor with FLOAT output",
                "ones_as",
                integerInput,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "ones_as", DataType.FLOAT, sd, variables.get("x"));
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> constantValues(length, 1.0f),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final List<InputSpec> triangularInput = Collections.singletonList(
                new InputSpec("x", shape,
                        step -> generalValues(step, length, 0.12f, -0.8f)));
        runStrictSingleDispatch(
                "triu batched rectangular matrices diagonal=1",
                "triu",
                triangularInput,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "triu", sd, variables.get("x"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> triuOracle(
                        generalValues(step, length, 0.12f, -0.8f),
                        2, 3, 4, 1),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);

        final List<InputSpec> triangularGradientInputs = List.of(
                new InputSpec("x", shape,
                        step -> generalValues(step, length, 0.07f, -0.35f)),
                new InputSpec("grad", shape,
                        step -> generalValues(step + 4, length, 0.16f, -1.1f)));
        runStrictSingleDispatch(
                "triu_bp masks upstream gradient diagonal=-1",
                "triu_bp",
                triangularGradientInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "triu_bp", sd,
                            variables.get("x"), variables.get("grad"));
                    op.addIArgument(-1);
                    return op.outputVariable();
                },
                step -> triuOracle(
                        generalValues(step + 4, length, 0.16f, -1.1f),
                        2, 3, 4, -1),
                DataType.FLOAT,
                shape,
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("shape_of/onehot: static metadata generates tensors in one real pipeline")
    void shapeAndOneHotGenerationCreateSinglePipelinesAndReplay() {
        final long[] donorShape = {2, 3, 4};
        final List<InputSpec> shapeInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        donorShape,
                        step -> featureStridedInput(step, 2, 3, 4),
                        true));
        runStrictFrozenConstantDispatch(
                "shape_of INT32 from rank-3 feature-strided view",
                "shape_of",
                shapeInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "shape_of", DataType.INT32,
                            sd, variables.get("x"));
                    op.addIArgument(DataType.INT32.toInt());
                    return op.outputVariable();
                },
                step -> new float[]{2.0f, 3.0f, 4.0f},
                DataType.INT32,
                new long[]{3},
                0.0f,
                0.0f);

        final int rows = 2;
        final int columns = 3;
        final int depth = 4;
        final int axis = 1;
        final double on = 2.5;
        final double off = -0.25;
        final List<InputSpec> oneHotInputs = Collections.singletonList(
                new InputSpec("indices", new long[]{rows, columns},
                        VulkanKernelEmitterStrictReplayTest::oneHotIndexValues));
        runStrictSingleDispatch(
                "onehot FLOAT indices axis=1 with static on/off values",
                "onehot",
                oneHotInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "onehot", DataType.FLOAT,
                            sd, variables.get("indices"));
                    op.addIArgument(axis, depth);
                    op.addTArgument(on, off);
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> oneHotOracle(
                        oneHotIndexValues(step), rows, columns,
                        axis, depth, (float) on, (float) off),
                DataType.FLOAT,
                new long[]{rows, depth, columns},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("eye/lin_space/range: zero-input constants execute one output-only pipeline then freeze")
    void zeroInputGenerationCreatesSinglePipelinesAndFreezes() {
        final int batches = 2;
        final int rows = 3;
        final int columns = 4;
        runStrictFrozenConstantDispatch(
                "eye F-order batched rectangular matrices",
                "eye",
                Collections.emptyList(),
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "eye", DataType.FLOAT, sd);
                    op.addIArgument(-102, rows, columns, batches);
                    return op.outputVariable();
                },
                step -> eyeOracle(batches, rows, columns),
                DataType.FLOAT,
                new long[]{batches, rows, columns},
                0.0f,
                0.0f);

        final int steps = 7;
        final double start = -1.5;
        final double end = 1.5;
        runStrictFrozenConstantDispatch(
                "lin_space static endpoint form",
                "lin_space",
                Collections.emptyList(),
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "lin_space", DataType.FLOAT, sd);
                    op.addIArgument(steps);
                    op.addTArgument(start, end);
                    op.addBArgument(true);
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> linSpaceOracle(start, end, steps),
                DataType.FLOAT,
                new long[]{steps},
                0.0f,
                0.0f);

        final double rangeStart = 2.25;
        final double rangeLimit = -1.75;
        final double rangeDelta = -0.5;
        final int rangeSteps = 8;
        runStrictFrozenConstantDispatch(
                "range static TArgs FLOAT descending",
                "range",
                Collections.emptyList(),
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "range", DataType.FLOAT, sd);
                    op.addTArgument(rangeStart, rangeLimit, rangeDelta);
                    op.addDArgument(DataType.FLOAT);
                    return op.outputVariable();
                },
                step -> rangeOracle(rangeStart, rangeDelta, rangeSteps),
                DataType.FLOAT,
                new long[]{rangeSteps},
                0.0f,
                0.0f);

        final int integerRangeStart = -5;
        final int integerRangeLimit = 8;
        final int integerRangeDelta = 3;
        final int integerRangeSteps = 5;
        runStrictFrozenConstantDispatch(
                "range static IArgs INT32 ascending",
                "range",
                Collections.emptyList(),
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "range", DataType.INT32, sd);
                    op.addIArgument(
                            integerRangeStart,
                            integerRangeLimit,
                            integerRangeDelta);
                    op.addDArgument(DataType.INT32);
                    return op.outputVariable();
                },
                step -> rangeOracle(
                        integerRangeStart,
                        integerRangeDelta,
                        integerRangeSteps),
                DataType.INT32,
                new long[]{integerRangeSteps},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("rank/size/size_at/min_max_datatype: scalar metadata executes once then freezes")
    void scalarMetadataGenerationCreatesSinglePipelinesThenFreezes() {
        final long[] shape = {2, 3, 4};
        final List<InputSpec> inputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        shape,
                        step -> featureStridedInput(step, 2, 3, 4),
                        true));

        runStrictFrozenConstantDispatch(
                "rank with explicit INT32 output override",
                "rank",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "rank", DataType.INT32, sd, variables.get("x"));
                    op.addDArgument(DataType.INT32);
                    return op.outputVariable();
                },
                step -> new float[]{3.0f},
                DataType.INT32,
                new long[0],
                0.0f,
                0.0f);

        runStrictFrozenConstantDispatch(
                "size_at axis=-2 from feature-strided view",
                "size_at",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "size_at", DataType.INT32, sd, variables.get("x"));
                    op.addIArgument(-2);
                    op.addDArgument(DataType.INT32);
                    return op.outputVariable();
                },
                step -> new float[]{3.0f},
                DataType.INT32,
                new long[0],
                0.0f,
                0.0f);

        runStrictFrozenConstantDispatch(
                "size with explicit INT32 output override",
                "size",
                inputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "size", DataType.INT32, sd, variables.get("x"));
                    op.addDArgument(DataType.INT32);
                    return op.outputVariable();
                },
                step -> new float[]{24.0f},
                DataType.INT32,
                new long[0],
                0.0f,
                0.0f);

        runStrictFrozenConstantDispatch(
                "min_max_datatype FLOAT minimum finite value",
                "min_max_datatype",
                Collections.emptyList(),
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "min_max_datatype", DataType.FLOAT, sd);
                    op.addIArgument(DataType.FLOAT.toInt(), 0);
                    return op.outputVariable();
                },
                step -> new float[]{1.175494e-38f},
                DataType.FLOAT,
                new long[0],
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("split/unstack: multi-output movement replays one compute pipeline")
    void multiOutputMovementCreatesSinglePipelineAndReplay() {
        final List<InputSpec> splitInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 6},
                        step -> generalValues(step, 12, 0.14f, -0.7f)));
        runStrictSingleDispatch(
                "split three equal axis-1 shards; observe shard 1",
                "split",
                splitInputs,
                (sd, variables) -> {
                    NamedFixedOutputOp op = new NamedFixedOutputOp(
                            "split", 3, sd, variables.get("x"));
                    op.addIArgument(3, 1);
                    return op.outputVariables()[1];
                },
                step -> splitShardOracle(
                        generalValues(step, 12, 0.14f, -0.7f),
                        2, 6, 3, 1),
                DataType.FLOAT,
                new long[]{2, 2},
                0.0f,
                0.0f);

        final List<InputSpec> unstackInputs = Collections.singletonList(
                new InputSpec("x", DataType.INT32, new long[]{2, 3, 4},
                        step -> integerValues(step, 24)));
        runStrictSingleDispatch(
                "unstack axis=1; observe slice 2",
                "unstack",
                unstackInputs,
                (sd, variables) -> {
                    NamedFixedOutputOp op = new NamedFixedOutputOp(
                            "unstack", 3, sd, variables.get("x"));
                    op.addIArgument(1);
                    return op.outputVariables()[2];
                },
                step -> unstackSliceOracle(
                        integerValues(step, 24), 2, 3, 4, 2),
                DataType.INT32,
                new long[]{2, 4},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("gather/embedding/tile/repeat/reverse/roll/slice/stack: rank-N movement replays pipelines")
    void staticRankNMovementEmittersCreateSinglePipelinesAndReplay() {
        final List<InputSpec> gatherInputs = List.of(
                new InputSpec("x", new long[]{3, 4},
                        step -> generalValues(step, 12, 0.13f, -0.8f)),
                new InputSpec("indices", DataType.INT32, new long[]{4, 1},
                        VulkanKernelEmitterStrictReplayTest::gatherNdIndices));
        runStrictSingleDispatch(
                "gather_nd rank-2 with native clamp semantics",
                "gather_nd",
                gatherInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "gather_nd", sd,
                        variables.get("x"), variables.get("indices"))
                        .outputVariable(),
                step -> gatherNdOracle(
                        generalValues(step, 12, 0.13f, -0.8f),
                        gatherNdIndices(step), 3, 4),
                DataType.FLOAT,
                new long[]{4, 4},
                0.0f,
                0.0f);

        final List<InputSpec> embeddingInputs = List.of(
                new InputSpec("table", new long[]{5, 2, 3},
                        step -> generalValues(step, 30, 0.065f, -0.5f)),
                new InputSpec("indices", DataType.UINT32, new long[]{4},
                        VulkanKernelEmitterStrictReplayTest::embeddingIndices));
        runStrictSingleDispatch(
                "embedding_lookup rank-3 table with UINT32 indices",
                "embedding_lookup",
                embeddingInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "embedding_lookup", sd,
                            variables.get("table"), variables.get("indices"));
                    op.addIArgument(0);
                    return op.outputVariable();
                },
                step -> embeddingLookupOracle(
                        generalValues(step, 30, 0.065f, -0.5f),
                        embeddingIndices(step), 5, 2, 3),
                DataType.FLOAT,
                new long[]{4, 2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> tileInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.21f, -0.6f)));
        runStrictSingleDispatch(
                "tile static repetitions [2,3]",
                "tile",
                tileInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "tile", sd, variables.get("x"));
                    op.addIArgument(2, 3);
                    return op.outputVariable();
                },
                step -> tileOracle(
                        generalValues(step, 6, 0.21f, -0.6f),
                        2, 3, 2, 3),
                DataType.FLOAT,
                new long[]{4, 9},
                0.0f,
                0.0f);

        final List<InputSpec> repeatInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.16f, -0.45f)));
        runStrictSingleDispatch(
                "repeat axis=1 prefix counts [1,2,1]",
                "repeat",
                repeatInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "repeat", sd, variables.get("x"));
                    op.addIArgument(1, 2, 1, 1);
                    return op.outputVariable();
                },
                step -> repeatOracle(
                        generalValues(step, 6, 0.16f, -0.45f),
                        2, 3, new int[]{1, 2, 1}),
                DataType.FLOAT,
                new long[]{2, 4},
                0.0f,
                0.0f);

        final List<InputSpec> reverseInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3, 4},
                        step -> generalValues(step, 24, 0.09f, -0.7f)));
        runStrictSingleDispatch(
                "reverse rank-3 axes [0,-1]",
                "reverse",
                reverseInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "reverse", sd, variables.get("x"));
                    op.addIArgument(0, -1);
                    return op.outputVariable();
                },
                step -> reverseRankThreeOracle(
                        generalValues(step, 24, 0.09f, -0.7f), 2, 3, 4),
                DataType.FLOAT,
                new long[]{2, 3, 4},
                0.0f,
                0.0f);

        final List<InputSpec> rollLinearInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.14f, -0.35f)));
        runStrictSingleDispatch(
                "roll frozen linear shift -2",
                "roll",
                rollLinearInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "roll", sd, variables.get("x"));
                    op.addIArgument(-2);
                    return op.outputVariable();
                },
                step -> rollLinearOracle(
                        generalValues(step, 6, 0.14f, -0.35f), -2),
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> rollAxesInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3, 4},
                        step -> generalValues(step, 24, 0.055f, -0.2f)));
        runStrictSingleDispatch(
                "roll frozen shift 1 over axes [0,-1]",
                "roll",
                rollAxesInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "roll", sd, variables.get("x"));
                    op.addIArgument(1, 0, -1);
                    return op.outputVariable();
                },
                step -> rollRankThreeOracle(
                        generalValues(step, 24, 0.055f, -0.2f),
                        2, 3, 4, new float[]{1.0f, 1.0f},
                        new float[]{0.0f, -1.0f}),
                DataType.FLOAT,
                new long[]{2, 3, 4},
                0.0f,
                0.0f);

        final List<InputSpec> rollTensorLinearInputs = List.of(
                new InputSpec("x", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.11f, -0.4f)),
                new InputSpec("shift", DataType.INT32, new long[]{1},
                        VulkanKernelEmitterStrictReplayTest::rollLinearShift));
        runStrictSingleDispatch(
                "roll live tensor linear shift",
                "roll",
                rollTensorLinearInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "roll", sd, variables.get("x"),
                        variables.get("shift")).outputVariable(),
                step -> rollLinearOracle(
                        generalValues(step, 6, 0.11f, -0.4f),
                        (int) rollLinearShift(step)[0]),
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> rollTensorAxesInputs = List.of(
                new InputSpec("x", new long[]{2, 3, 4},
                        step -> generalValues(step, 24, 0.07f, -0.55f)),
                new InputSpec("shifts", DataType.INT32, new long[]{3},
                        VulkanKernelEmitterStrictReplayTest::rollAxisShifts),
                new InputSpec("axes", DataType.INT32, new long[]{3},
                        VulkanKernelEmitterStrictReplayTest::rollAxes));
        runStrictSingleDispatch(
                "roll live tensor shifts and duplicate axes",
                "roll",
                rollTensorAxesInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "roll", sd, variables.get("x"),
                        variables.get("shifts"),
                        variables.get("axes")).outputVariable(),
                step -> rollRankThreeOracle(
                        generalValues(step, 24, 0.07f, -0.55f),
                        2, 3, 4, rollAxisShifts(step), rollAxes(step)),
                DataType.FLOAT,
                new long[]{2, 3, 4},
                0.0f,
                0.0f);

        final List<InputSpec> sliceInputs = Collections.singletonList(
                new InputSpec("x", new long[]{4, 5},
                        step -> generalValues(step, 20, 0.12f, -0.9f)));
        runStrictSingleDispatch(
                "slice static begin=[1,1] size=[2,3]",
                "slice",
                sliceInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "slice", sd, variables.get("x"));
                    op.addIArgument(1, 1, 2, 3);
                    return op.outputVariable();
                },
                step -> sliceOracle(
                        generalValues(step, 20, 0.12f, -0.9f),
                        4, 5, 1, 1, 2, 3),
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> stridedSliceInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 5, 4},
                        step -> generalValues(step, 40, 0.075f, -0.65f)));
        runStrictSingleDispatch(
                "strided_slice static positive strides rank-3",
                "strided_slice",
                stridedSliceInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "strided_slice", sd, variables.get("x"));
                    op.addIArgument(
                            0, 0, 0, 0, 0,
                            0, 1, 0,
                            2, 5, 4,
                            1, 2, 2);
                    return op.outputVariable();
                },
                step -> stridedSliceOracle(
                        generalValues(step, 40, 0.075f, -0.65f),
                        2, 5, 4),
                DataType.FLOAT,
                new long[]{2, 2, 2},
                0.0f,
                0.0f);

        final List<InputSpec> stackInputs = List.of(
                new InputSpec("a", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.08f, -0.4f)),
                new InputSpec("b", new long[]{2, 3},
                        step -> generalValues(step + 5, 6, 0.11f, -0.2f)),
                new InputSpec("c", new long[]{2, 3},
                        step -> generalValues(step + 9, 6, 0.06f, 0.1f)));
        runStrictSingleDispatch(
                "stack three rank-2 inputs axis=1",
                "stack",
                stackInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "stack", sd, variables.get("a"),
                            variables.get("b"), variables.get("c"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> stackAxisOneOracle(
                        new float[][]{
                                generalValues(step, 6, 0.08f, -0.4f),
                                generalValues(step + 5, 6, 0.11f, -0.2f),
                                generalValues(step + 9, 6, 0.06f, 0.1f)},
                        2, 3),
                DataType.FLOAT,
                new long[]{2, 3, 3},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("win_part/win_unpart: exact rank-4 window mappings replay one pipeline")
    void windowPartitionAndUnpartitionCreateSinglePipelinesAndReplay() {
        final int batch = 2;
        final int height = 4;
        final int width = 4;
        final int channels = 3;
        final int window = 2;
        final int length = batch * height * width * channels;

        final List<InputSpec> partitionInputs = Collections.singletonList(
                new InputSpec("x", new long[]{batch, height, width, channels},
                        step -> generalValues(step, length, 0.045f, -0.55f)));
        runStrictSingleDispatch(
                "win_part rank-4 window=2",
                "win_part",
                partitionInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "win_part", sd, variables.get("x"));
                    op.addIArgument(window);
                    return op.outputVariable();
                },
                step -> windowPartitionOracle(
                        generalValues(step, length, 0.045f, -0.55f),
                        batch, height, width, channels, window),
                DataType.FLOAT,
                new long[]{8, window, window, channels},
                0.0f,
                0.0f);

        final List<InputSpec> unpartitionInputs = Collections.singletonList(
                new InputSpec("windows", new long[]{8, window, window, channels},
                        step -> generalValues(step + 3, length, 0.055f, -0.35f)));
        runStrictSingleDispatch(
                "win_unpart rank-4 window=2",
                "win_unpart",
                unpartitionInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "win_unpart", sd, variables.get("windows"));
                    op.addIArgument(window, height, width);
                    return op.outputVariable();
                },
                step -> windowUnpartitionOracle(
                        generalValues(step + 3, length, 0.055f, -0.35f),
                        batch, height, width, channels, window),
                DataType.FLOAT,
                new long[]{batch, height, width, channels},
                0.0f,
                0.0f);
    }

    @Test
    @DisplayName("contract movement and normalize_moments create one real replayable pipeline")
    void contractMovementAndNormalizeMomentsCreateSinglePipelinesAndReplay() {
        final List<InputSpec> squeezeInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.INT32,
                        new long[]{2, 1, 3},
                        step -> new float[]{
                                step + 1, step + 2, step + 3,
                                step + 4, step + 5, step + 6}));
        runStrictSingleDispatch(
                "squeeze INT32 explicit singleton axis",
                "squeeze",
                squeezeInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "squeeze", sd, variables.get("x"));
                    op.addIArgument(1);
                    return op.outputVariable();
                },
                step -> new float[]{
                        step + 1, step + 2, step + 3,
                        step + 4, step + 5, step + 6},
                DataType.INT32,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> expandInputs = Collections.singletonList(
                new InputSpec("x", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.125f, -0.5f)));
        runStrictSingleDispatch(
                "expand_dims negative axis",
                "expand_dims",
                expandInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "expand_dims", sd, variables.get("x"));
                    op.addIArgument(-1);
                    return op.outputVariable();
                },
                step -> generalValues(step, 6, 0.125f, -0.5f),
                DataType.FLOAT,
                new long[]{2, 3, 1},
                0.0f,
                0.0f);

        final List<InputSpec> reshapeInputs = Collections.singletonList(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        new long[]{2, 3, 4},
                        step -> featureStridedInput(step, 2, 3, 4),
                        true));
        runStrictSingleDispatch(
                "reshape C-order from a feature-strided view",
                "reshape",
                reshapeInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "reshape", sd, variables.get("x"));
                    op.addIArgument(-'c', 6, 4);
                    return op.outputVariable();
                },
                step -> generalValues(step, 24, 0.0375f, -0.65f),
                DataType.FLOAT,
                new long[]{6, 4},
                0.0f,
                0.0f);

        final List<InputSpec> structuralReshapeInputs = List.of(
                new InputSpec(
                        "x",
                        DataType.FLOAT,
                        new long[]{2, 3, 4},
                        step -> featureStridedInput(step, 2, 3, 4),
                        true),
                new InputSpec(
                        "shape",
                        DataType.LONG,
                        new long[]{2},
                        step -> new float[]{4, 6}));
        runStrictSingleDispatch(
                "reshape structural shape tensor remains device-side metadata",
                "reshape",
                structuralReshapeInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "reshape", sd,
                        variables.get("x"), variables.get("shape"))
                        .outputVariable(),
                step -> generalValues(step, 24, 0.0375f, -0.65f),
                DataType.FLOAT,
                new long[]{4, 6},
                0.0f,
                0.0f);

        final List<InputSpec> flattenInputs = List.of(
                new InputSpec("a", new long[]{2, 2},
                        step -> generalValues(step, 4, 0.10f, -0.4f)),
                new InputSpec("b", new long[]{2, 3},
                        step -> generalValues(step + 4, 6, 0.075f, 0.2f)));
        runStrictSingleDispatch(
                "flatten two inputs in Fortran traversal order",
                "flatten",
                flattenInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "flatten", sd,
                            variables.get("a"), variables.get("b"));
                    op.addIArgument('f');
                    return op.outputVariable();
                },
                step -> {
                    float[] a = generalValues(step, 4, 0.10f, -0.4f);
                    float[] b = generalValues(step + 4, 6, 0.075f, 0.2f);
                    return new float[]{
                            a[0], a[2], a[1], a[3],
                            b[0], b[3], b[1], b[4], b[2], b[5]};
                },
                DataType.FLOAT,
                new long[]{10},
                0.0f,
                0.0f);

        final List<InputSpec> broadcastInputs = List.of(
                new InputSpec("x", new long[]{2, 1},
                        step -> generalValues(step, 2, 0.25f, -0.75f)),
                new InputSpec("shape", DataType.LONG, new long[]{2},
                        step -> new float[]{2, 3}));
        runStrictSingleDispatch(
                "broadcast_to ignores structural shape payload in the shader",
                "broadcast_to",
                broadcastInputs,
                (sd, variables) -> new NamedDynamicOp(
                        "broadcast_to", sd,
                        variables.get("x"), variables.get("shape"))
                        .outputVariable(),
                step -> {
                    float[] input = generalValues(step, 2, 0.25f, -0.75f);
                    return new float[]{
                            input[0], input[0], input[0],
                            input[1], input[1], input[1]};
                },
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> splitInputs = List.of(
                new InputSpec("x", new long[]{2, 5},
                        step -> generalValues(step, 10, 0.09f, -0.6f)),
                new InputSpec("sizes", DataType.LONG, new long[]{2},
                        step -> new float[]{2, 3}));
        for (int observedOutput = 0; observedOutput < 2; observedOutput++) {
            final int outputIndex = observedOutput;
            runStrictSingleDispatch(
                    "split_v unequal axis partitions output=" + outputIndex,
                    "split_v",
                    splitInputs,
                    (sd, variables) -> {
                        NamedFixedOutputOp op = new NamedFixedOutputOp(
                                "split_v", 2, sd,
                                variables.get("x"), variables.get("sizes"));
                        op.addIArgument(1);
                        return op.outputVariables()[outputIndex];
                    },
                    step -> {
                        float[] input = generalValues(step, 10, 0.09f, -0.6f);
                        return outputIndex == 0
                                ? new float[]{input[0], input[1], input[5], input[6]}
                                : new float[]{
                                        input[2], input[3], input[4],
                                        input[7], input[8], input[9]};
                    },
                    DataType.FLOAT,
                    outputIndex == 0 ? new long[]{2, 2} : new long[]{2, 3},
                    0.0f,
                    0.0f);
        }

        final List<InputSpec> createInputs = Collections.singletonList(
                new InputSpec("shape", DataType.LONG, new long[]{2},
                        step -> new float[]{2, 3}));
        runStrictSingleDispatch(
                "create initialized INT32 tensor",
                "create",
                createInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "create", DataType.INT32,
                            sd, variables.get("shape"));
                    op.addDArgument(DataType.INT32);
                    op.addIArgument('c', DataType.INT32.toInt());
                    op.addBArgument(true);
                    return op.outputVariable();
                },
                step -> new float[6],
                DataType.INT32,
                new long[]{2, 3},
                0.0f,
                0.0f);
        runStrictSingleDispatch(
                "create initialized FLOAT tensor",
                "create",
                createInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "create", DataType.FLOAT,
                            sd, variables.get("shape"));
                    op.addDArgument(DataType.FLOAT);
                    op.addIArgument('c', DataType.FLOAT.toInt());
                    op.addBArgument(true);
                    return op.outputVariable();
                },
                step -> new float[6],
                DataType.FLOAT,
                new long[]{2, 3},
                0.0f,
                0.0f);

        final List<InputSpec> normalizeInputs = List.of(
                new InputSpec("counts", new long[]{1},
                        step -> new float[]{2.0f + step * 0.25f}),
                new InputSpec("summedMeans", new long[]{2, 3},
                        step -> generalValues(step, 6, 0.16f, -0.3f)),
                new InputSpec("summedVariances", new long[]{2, 3},
                        step -> generalValues(step + 7, 6, 0.23f, 2.5f)));
        final double shift = 0.25;
        for (int observedOutput = 0; observedOutput < 2; observedOutput++) {
            final int outputIndex = observedOutput;
            runStrictSingleDispatch(
                    "normalize_moments output=" + outputIndex,
                    "normalize_moments",
                    normalizeInputs,
                    (sd, variables) -> {
                        NamedFixedOutputOp op = new NamedFixedOutputOp(
                                "normalize_moments", 2, sd,
                                variables.get("counts"),
                                variables.get("summedMeans"),
                                variables.get("summedVariances"));
                        op.addTArgument(shift);
                        return op.outputVariables()[outputIndex];
                    },
                    step -> {
                        float count = 2.0f + step * 0.25f;
                        float[] summedMeans =
                                generalValues(step, 6, 0.16f, -0.3f);
                        float[] summedVariances =
                                generalValues(step + 7, 6, 0.23f, 2.5f);
                        float[] output = new float[6];
                        for (int i = 0; i < output.length; i++) {
                            float mean = summedMeans[i] / count;
                            output[i] = outputIndex == 0
                                    ? (float) (mean + shift)
                                    : summedVariances[i] / count - mean * mean;
                        }
                        return output;
                    },
                    DataType.FLOAT,
                    new long[]{2, 3},
                    1.0e-6f,
                    1.0e-6f);
        }
    }

    @Test
    @DisplayName("batched matrix lists and duplicate-index scatter replay one real pipeline")
    void batchedMatrixListAndIndexedAccumulationCreateSinglePipelinesAndReplay() {
        final List<InputSpec> batchedInputs = List.of(
                new InputSpec("alpha", new long[]{2},
                        step -> new float[]{
                                1.0f + step * 0.05f,
                                -0.75f + step * 0.025f}),
                new InputSpec("beta", new long[]{1},
                        step -> new float[]{0.0f}),
                new InputSpec("a0", new long[]{3, 2},
                        step -> generalValues(step, 6, 0.09f, -0.4f)),
                new InputSpec("a1", new long[]{3, 2},
                        step -> generalValues(step + 3, 6, 0.08f, 0.15f)),
                new InputSpec("b0", new long[]{2, 3},
                        step -> generalValues(step + 4, 6, 0.07f, 0.2f)),
                new InputSpec("b1", new long[]{2, 3},
                        step -> generalValues(step + 8, 6, 0.06f, -0.3f)));
        for (int observedOutput = 0; observedOutput < 2; observedOutput++) {
            final int outputIndex = observedOutput;
            runStrictSingleDispatch(
                    "batched_gemm transposed pair output=" + outputIndex,
                    "batched_gemm",
                    batchedInputs,
                    (sd, variables) -> {
                        NamedFixedOutputOp op = new NamedFixedOutputOp(
                                "batched_gemm", 2, sd,
                                variables.get("alpha"), variables.get("beta"),
                                variables.get("a0"), variables.get("a1"),
                                variables.get("b0"), variables.get("b1"));
                        op.addIArgument(1, 1);
                        return op.outputVariables()[outputIndex];
                    },
                    step -> {
                        float alpha = outputIndex == 0
                                ? 1.0f + step * 0.05f
                                : -0.75f + step * 0.025f;
                        float[] a = outputIndex == 0
                                ? generalValues(step, 6, 0.09f, -0.4f)
                                : generalValues(step + 3, 6, 0.08f, 0.15f);
                        float[] b = outputIndex == 0
                                ? generalValues(step + 4, 6, 0.07f, 0.2f)
                                : generalValues(step + 8, 6, 0.06f, -0.3f);
                        float[] output = new float[4];
                        for (int row = 0; row < 2; row++) {
                            for (int column = 0; column < 2; column++) {
                                float sum = 0.0f;
                                for (int k = 0; k < 3; k++) {
                                    sum += a[k * 2 + row] * b[column * 3 + k];
                                }
                                output[row * 2 + column] = alpha * sum;
                            }
                        }
                        return output;
                    },
                    DataType.FLOAT,
                    new long[]{2, 2},
                    1.0e-5f,
                    1.0e-5f);
        }

        for (int[] transposePair : new int[][]{
                {0, 0}, {1, 0}, {0, 1}}) {
            final boolean transposeA = transposePair[0] != 0;
            final boolean transposeB = transposePair[1] != 0;
            final long[] aShape = transposeA
                    ? new long[]{3, 2}
                    : new long[]{2, 3};
            final long[] bShape = transposeB
                    ? new long[]{4, 3}
                    : new long[]{3, 4};
            final List<InputSpec> singlePairInputs = List.of(
                    new InputSpec("alpha", new long[]{1},
                            step -> new float[]{0.5f + step * 0.02f}),
                    new InputSpec("beta", new long[]{1},
                            step -> new float[]{1.25f - step * 0.01f}),
                    new InputSpec("a", aShape,
                            step -> generalValues(step + 2, 6, 0.08f, -0.3f)),
                    new InputSpec("b", bShape,
                            step -> generalValues(step + 6, 12, 0.055f, 0.2f)));
            runStrictSingleDispatch(
                    "batched_gemm single pair transposeA=" + transposeA
                            + " transposeB=" + transposeB,
                    "batched_gemm",
                    singlePairInputs,
                    (sd, variables) -> {
                        NamedFixedOutputOp op = new NamedFixedOutputOp(
                                "batched_gemm", 1, sd,
                                variables.get("alpha"), variables.get("beta"),
                                variables.get("a"), variables.get("b"));
                        op.addIArgument(transposePair[0], transposePair[1]);
                        return op.outputVariables()[0];
                    },
                    step -> batchedMatrixProductOracle(
                            generalValues(step + 2, 6, 0.08f, -0.3f),
                            generalValues(step + 6, 12, 0.055f, 0.2f),
                            2,
                            3,
                            4,
                            transposeA,
                            transposeB,
                            0.5f + step * 0.02f),
                    DataType.FLOAT,
                    new long[]{2, 4},
                    1.0e-5f,
                    1.0e-5f);
        }

        final List<InputSpec> scatterInputs = List.of(
                new InputSpec("indices", DataType.INT32, new long[]{3, 1},
                        step -> new float[]{1, 1, 3}),
                new InputSpec("updates", new long[]{3},
                        step -> generalValues(step, 3, 0.2f, -0.4f)),
                new InputSpec("shape", DataType.LONG, new long[]{1},
                        step -> new float[]{5}));
        runStrictSingleDispatch(
                "scatter_nd duplicate indices use serial indexed accumulation",
                "scatter_nd",
                scatterInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "scatter_nd", DataType.FLOAT, sd,
                            variables.get("indices"),
                            variables.get("updates"),
                            variables.get("shape"));
                    op.addBArgument(false, false);
                    return op.outputVariable();
                },
                step -> {
                    float[] updates = generalValues(step, 3, 0.2f, -0.4f);
                    return new float[]{
                            0.0f, updates[0] + updates[1],
                            0.0f, updates[2], 0.0f};
                },
                DataType.FLOAT,
                new long[]{5},
                0.0f,
                0.0f);

        final List<InputSpec> sliceScatterInputs = List.of(
                new InputSpec(
                        "indices",
                        DataType.UINT32,
                        new long[]{2, 2, 1},
                        step -> new float[]{0, 2, 1, 2}),
                new InputSpec(
                        "updates",
                        new long[]{2, 2, 4},
                        step -> generalValues(step + 11, 16, 0.075f, -0.5f)),
                new InputSpec(
                        "shape",
                        DataType.LONG,
                        new long[]{2},
                        step -> new float[]{3, 4}));
        runStrictSingleDispatch(
                "scatter_nd UINT32 rank-two prefix slice accumulation lock=true",
                "scatter_nd",
                sliceScatterInputs,
                (sd, variables) -> {
                    NamedDynamicOp op = new NamedDynamicOp(
                            "scatter_nd", DataType.FLOAT, sd,
                            variables.get("indices"),
                            variables.get("updates"),
                            variables.get("shape"));
                    op.addBArgument(true, false);
                    return op.outputVariable();
                },
                step -> {
                    float[] updates = generalValues(
                            step + 11, 16, 0.075f, -0.5f);
                    int[] rows = {0, 2, 1, 2};
                    float[] output = new float[12];
                    for (int tuple = 0; tuple < rows.length; tuple++) {
                        for (int column = 0; column < 4; column++) {
                            output[rows[tuple] * 4 + column] +=
                                    updates[tuple * 4 + column];
                        }
                    }
                    return output;
                },
                DataType.FLOAT,
                new long[]{3, 4},
                0.0f,
                0.0f);
    }

    private void runStrictMultiOutputSingleDispatch(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            MultiOutputGraphBuilder graphBuilder,
            List<OutputSpec> outputSpecs) {

        assertTrue(outputSpecs.size() > 1,
                label + ": multi-output helper requires at least two outputs");
        activateSelectedDevice();
        invokeNative("dspDiagClear", new Class<?>[0]);
        invokeNative("dspDiagSetCategories", new Class<?>[]{int.class}, -1);
        invokeNative("dspDiagSetLevel", new Class<?>[]{int.class}, 2);

        SameDiff sameDiff = SameDiff.create();
        String diagnosticJson;
        try {
            Map<String, SDVariable> variables = new LinkedHashMap<>();
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            for (InputSpec spec : inputSpecs) {
                variables.put(spec.name,
                        sameDiff.placeHolder(spec.name, spec.dataType, spec.shape));
                inputs.put(spec.name, spec.arrayAt(0));
            }

            SDVariable[] outputs = graphBuilder.build(sameDiff, variables);
            assertNotNull(outputs, label + ": graph builder returned no outputs");
            assertEquals(outputSpecs.size(), outputs.length,
                    label + ": graph output count");
            String[] outputNames = new String[outputs.length];
            for (int outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
                outputNames[outputIndex] = OUTPUT + outputIndex;
                sameDiff.updateVariableNameAndReference(
                        outputs[outputIndex], outputNames[outputIndex]);
            }
            sameDiff.getSessions().clear();
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);
            sameDiff.setDspFallbackToAutoIfTritonUnavailable(false);

            float[][] actual = new float[outputs.length][];
            for (int step = 0; step < REPLAY_STEPS; step++) {
                for (InputSpec spec : inputSpecs) {
                    inputs.get(spec.name).assign(spec.arrayAt(step));
                }
                Map<String, INDArray> results =
                        sameDiff.output(inputs, outputNames);
                for (int outputIndex = 0;
                        outputIndex < outputNames.length;
                        outputIndex++) {
                    OutputSpec expected = outputSpecs.get(outputIndex);
                    INDArray result = results.get(outputNames[outputIndex]);
                    assertNotNull(
                            result,
                            label + ": SameDiff returned no output["
                                    + outputIndex + "] at step " + step);
                    assertEquals(
                            expected.dataType,
                            result.dataType(),
                            label + ": output[" + outputIndex
                                    + "] dtype at step " + step);
                    assertArrayEquals(
                            expected.shape,
                            result.shape(),
                            label + ": output[" + outputIndex
                                    + "] shape at step " + step);
                    actual[outputIndex] = result.dup('c').toFloatVector();
                    assertClose(
                            expected.oracle.apply(step),
                            actual[outputIndex],
                            expected.absoluteTolerance,
                            expected.relativeTolerance,
                            label + " output[" + outputIndex
                                    + "] numerical result at step " + step);
                }
            }

            assertEquals(1, sameDiff.dsp().numSegments(),
                    label + ": a one-op graph must compile as exactly one segment");
            DspPlanAssertions.assertOpCompiled(
                    sameDiff, canonicalOpName, label);
            DspPlanAssertions.assertFullyReplaying(sameDiff, label);
            assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                    label + ": no graph replay was recorded");
            DspPlanAssertions.assertAllSegmentsCompiledWith(
                    sameDiff, "vulkan-native", label);
            DspPlanAssertions.assertNoSlotBySlotFallback(sameDiff, label);
            DspPlanAssertions.assertNoFallbacks(sameDiff, label);

            diagnosticJson = (String) invokeNative(
                    "dspDiagGetJsonReport", new Class<?>[0]);
        } finally {
            sameDiff.close();
        }

        assertNotNull(diagnosticJson,
                label + ": Vulkan diagnostic report is null");
        assertTrue(diagnosticJson.contains("vulkan_backend CAPTURE_DONE"),
                () -> label + ": no real Vulkan capture event was emitted:\n"
                        + diagnosticJson);
        assertTrue(diagnosticJson.contains("vulkan_backend REPLAY_DONE"),
                () -> label + ": no real Vulkan replay event was emitted:\n"
                        + diagnosticJson);
        assertEquals(1L, jsonLongMetric(diagnosticJson, "num_dispatches"),
                () -> label
                        + ": all destinations must share one compute dispatch:\n"
                        + diagnosticJson);
        assertTrue(jsonLongMetric(diagnosticJson, "replay_count") > 0,
                () -> label + ": replay_count must be positive:\n"
                        + diagnosticJson);
        assertEquals(
                selectedDeviceName,
                jsonStringMetric(diagnosticJson, "device_name"),
                label + ": replay ran on a different Vulkan device");

        if (Boolean.getBoolean("nd4j.vulkan.test.requireHardware")) {
            String lower = selectedDeviceName.toLowerCase(Locale.ROOT);
            assertFalse(lower.contains("lavapipe")
                            || lower.contains("llvmpipe"),
                    label + ": strict hardware mode selected a software device");
        }
    }

    private void runStrictSingleDispatch(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            GraphBuilder graphBuilder,
            IntFunction<float[]> oracle,
            float absoluteTolerance,
            float relativeTolerance) {
        runStrictSingleDispatch(
                label,
                canonicalOpName,
                inputSpecs,
                graphBuilder,
                oracle,
                DataType.FLOAT,
                null,
                absoluteTolerance,
                relativeTolerance);
    }

    private void runStrictAliasReplay(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            GraphBuilder graphBuilder,
            IntFunction<float[]> oracle,
            DataType expectedOutputDataType,
            long[] expectedOutputShape,
            float absoluteTolerance,
            float relativeTolerance) {
        runStrictSingleDispatch(
                label,
                canonicalOpName,
                inputSpecs,
                graphBuilder,
                oracle,
                expectedOutputDataType,
                expectedOutputShape,
                absoluteTolerance,
                relativeTolerance,
                false,
                0L);
    }

    private void runStrictFrozenConstantDispatch(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            GraphBuilder graphBuilder,
            IntFunction<float[]> oracle,
            DataType expectedOutputDataType,
            long[] expectedOutputShape,
            float absoluteTolerance,
            float relativeTolerance) {
        runStrictSingleDispatch(
                label,
                canonicalOpName,
                inputSpecs,
                graphBuilder,
                oracle,
                expectedOutputDataType,
                expectedOutputShape,
                absoluteTolerance,
                relativeTolerance,
                true,
                1L);
    }

    private void runStrictSingleDispatch(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            GraphBuilder graphBuilder,
            IntFunction<float[]> oracle,
            DataType expectedOutputDataType,
            long[] expectedOutputShape,
            float absoluteTolerance,
            float relativeTolerance) {
        runStrictSingleDispatch(
                label,
                canonicalOpName,
                inputSpecs,
                graphBuilder,
                oracle,
                expectedOutputDataType,
                expectedOutputShape,
                absoluteTolerance,
                relativeTolerance,
                false,
                1L);
    }

    private void runStrictSingleDispatch(
            String label,
            String canonicalOpName,
            List<InputSpec> inputSpecs,
            GraphBuilder graphBuilder,
            IntFunction<float[]> oracle,
            DataType expectedOutputDataType,
            long[] expectedOutputShape,
            float absoluteTolerance,
            float relativeTolerance,
            boolean expectFrozenConstant,
            long expectedDispatches) {

        activateSelectedDevice();
        invokeNative("dspDiagClear", new Class<?>[0]);
        // Enable every category so assertNoFallbacks observes, rather than hides,
        // any compiler/backend fallback event. GRAPH_REPLAY is bit 16.
        invokeNative("dspDiagSetCategories", new Class<?>[]{int.class}, -1);
        invokeNative("dspDiagSetLevel", new Class<?>[]{int.class}, 2);

        SameDiff sameDiff = SameDiff.create();
        String diagnosticJson;
        try {
            Map<String, SDVariable> variables = new LinkedHashMap<>();
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            for (InputSpec spec : inputSpecs) {
                variables.put(spec.name,
                        sameDiff.placeHolder(spec.name, spec.dataType, spec.shape));
                inputs.put(spec.name, spec.arrayAt(0));
            }

            SDVariable output = graphBuilder.build(sameDiff, variables);
            sameDiff.updateVariableNameAndReference(output, OUTPUT);
            sameDiff.getSessions().clear();
            sameDiff.setDspAutoCompileEnabled(true);
            sameDiff.setDspNativeAutoCompileEnabled(true);
            sameDiff.setDspFallbackToAutoIfTritonUnavailable(false);

            float[] actual = null;
            for (int step = 0; step < REPLAY_STEPS; step++) {
                for (InputSpec spec : inputSpecs) {
                    inputs.get(spec.name).assign(spec.arrayAt(step));
                }
                INDArray result = sameDiff.output(inputs, OUTPUT).get(OUTPUT);
                assertNotNull(result, label + ": SameDiff returned no output at step " + step);
                assertEquals(
                        expectedOutputDataType,
                        result.dataType(),
                        label + ": output dtype at step " + step);
                if (expectedOutputShape != null) {
                    assertArrayEquals(
                            expectedOutputShape,
                            result.shape(),
                            label + ": output shape at step " + step);
                }
                if (expectedDispatches == 0L) {
                    assertFalse(
                            result.isView(),
                            label + ": requested alias output was not materialized at the plan boundary at step "
                                    + step);
                }
                actual = result.dup('c').toFloatVector();
                assertClose(
                        oracle.apply(step),
                        actual,
                        absoluteTolerance,
                        relativeTolerance,
                        label + " numerical result at step " + step);
            }

            assertEquals(1, sameDiff.dsp().numSegments(),
                    label + ": a one-op graph must compile as exactly one segment");
            DspPlanAssertions.assertOpCompiled(sameDiff, canonicalOpName, label);
            if (expectFrozenConstant) {
                // CUDA's established DSP contract freezes descriptor-static
                // constant segments after capture replay.  That includes both
                // zero-input generators and shape-only outputs whose placeholder
                // shapes are frozen; no later execution is needed.
                // Freezing releases the live segment handle (and therefore its
                // backend-name field), so the Vulkan backend is proven below by
                // the captured diagnostic event and selected hardware device.
                DspPlanAssertions.assertPhaseReached(
                        sameDiff, PlanPhase.SHAPES_FROZEN, label);
                DspPlanAssertions.assertSlotIsFrozenConstant(sameDiff, 0, label);
                DspPlanAssertions.assertFrozenExecCountAtLeast(sameDiff, 1, label);
            } else {
                DspPlanAssertions.assertFullyReplaying(sameDiff, label);
                assertTrue(DspPlanAssertions.getTotalGraphReplays(sameDiff) > 0,
                        label + ": no graph replay was recorded");
                DspPlanAssertions.assertAllSegmentsCompiledWith(
                        sameDiff, "vulkan-native", label);
            }
            DspPlanAssertions.assertNoSlotBySlotFallback(sameDiff, label);
            DspPlanAssertions.assertNoFallbacks(sameDiff, label);

            diagnosticJson = (String) invokeNative(
                    "dspDiagGetJsonReport", new Class<?>[0]);
        } finally {
            sameDiff.close();
        }

        assertNotNull(diagnosticJson, label + ": Vulkan diagnostic report is null");
        assertTrue(diagnosticJson.contains("vulkan_backend CAPTURE_DONE"),
                () -> label + ": no real Vulkan capture event was emitted:\n" + diagnosticJson);
        assertTrue(diagnosticJson.contains("vulkan_backend REPLAY_DONE"),
                () -> label + ": no real Vulkan replay event was emitted:\n" + diagnosticJson);
        if (expectedDispatches == 0L) {
            assertTrue(hasReplayEventWithDispatchCount(diagnosticJson, 0L),
                    () -> label + ": the view-only plan replayed a compute dispatch:\n"
                            + diagnosticJson);
        } else {
            assertEquals(expectedDispatches, jsonLongMetric(diagnosticJson, "num_dispatches"),
                    () -> label + ": unexpected captured compute-dispatch count:\n"
                            + diagnosticJson);
        }
        assertTrue(jsonLongMetric(diagnosticJson, "replay_count") > 0,
                () -> label + ": replay_count must be positive:\n" + diagnosticJson);
        assertEquals(selectedDeviceName, jsonStringMetric(diagnosticJson, "device_name"),
                label + ": replay ran on a different Vulkan device");

        if (Boolean.getBoolean("nd4j.vulkan.test.requireHardware")) {
            String lower = selectedDeviceName.toLowerCase(Locale.ROOT);
            assertFalse(lower.contains("lavapipe") || lower.contains("llvmpipe"),
                    label + ": strict hardware mode selected a software device");
        }
    }

    private static boolean probeMlir(Class<?> bindingsClass)
            throws ReflectiveOperationException {
        try {
            Object value = bindingsClass.getField("HAVE_MLIR").get(null);
            return value instanceof Number && ((Number) value).intValue() == 1;
        } catch (NoSuchFieldException missingGeneratedConstant) {
            try {
                Method method = bindingsClass.getMethod("isMlirEnabled");
                return (Boolean) method.invoke(nativeOps);
            } catch (NoSuchMethodException missingCapabilityMethod) {
                Object value = bindingsClass.getMethod(
                        "getConfigIntValue", String.class).invoke(nativeOps, "HAVE_MLIR");
                return value instanceof Number && ((Number) value).intValue() == 1;
            }
        }
    }

    private static void activateSelectedDevice() {
        assertNotNull(nativeOps, "Vulkan NativeOps must be initialized");
        Object result = invokeNative(
                "setDevice", new Class<?>[]{int.class}, selectedDeviceId);
        assertEquals(1, ((Number) result).intValue(),
                "setDevice(" + selectedDeviceId + ") must succeed");
    }

    private static Object invokeNative(
            String methodName, Class<?>[] parameterTypes, Object... arguments) {
        try {
            return nativeOps.getClass().getMethod(methodName, parameterTypes)
                    .invoke(nativeOps, arguments);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(
                    "Required Vulkan NativeOps method unavailable: " + methodName, e);
        }
    }

    private static long jsonLongMetric(String json, String key) {
        Matcher matcher = Pattern.compile(
                "\"" + Pattern.quote(key) + "\"\\s*:\\s*([0-9]+)")
                .matcher(json);
        assertTrue(matcher.find(),
                () -> "Missing Vulkan diagnostic metric '" + key + "':\n" + json);
        return Long.parseLong(matcher.group(1));
    }

    private static boolean hasReplayEventWithDispatchCount(
            String json, long expectedDispatches) {
        return Pattern.compile(
                        "vulkan_backend REPLAY_DONE[^\\r\\n}]*replay_count=([1-9][0-9]*)"
                                + "[^\\r\\n}]*dispatches=" + expectedDispatches
                                + "(?:[^0-9]|$)")
                .matcher(json)
                .find();
    }

    private static String jsonStringMetric(String json, String key) {
        Matcher matcher = Pattern.compile(
                "\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]+)\"")
                .matcher(json);
        assertTrue(matcher.find(),
                () -> "Missing Vulkan diagnostic metric '" + key + "':\n" + json);
        return matcher.group(1);
    }

    private static void assertClose(
            float[] expected,
            float[] actual,
            float absoluteTolerance,
            float relativeTolerance,
            String context) {
        assertNotNull(actual, context + ": actual output is null");
        assertEquals(expected.length, actual.length,
                context + ": output length mismatch");
        for (int i = 0; i < expected.length; i++) {
            final float wanted = expected[i];
            final float got = actual[i];
            if (Float.isNaN(wanted)) {
                assertTrue(Float.isNaN(got),
                        context + ": element[" + i + "] expected NaN but got " + got);
                continue;
            }
            if (Float.isInfinite(wanted)) {
                assertEquals(wanted, got,
                        context + ": element[" + i + "] infinity mismatch");
                continue;
            }
            float difference = Math.abs(wanted - got);
            float allowed = absoluteTolerance
                    + relativeTolerance * Math.abs(wanted);
            assertTrue(difference <= allowed,
                    context + ": element[" + i + "] expected=" + wanted
                            + " actual=" + got + " difference=" + difference
                            + " allowed=" + allowed);
        }
    }

    private static int[] acceptedChainCodes() {
        int[] result = new int[47];
        int index = 0;
        for (int code = 0; code <= 3; code++) {
            result[index++] = code;
        }
        for (int code = 10; code <= 42; code++) {
            result[index++] = code;
        }
        for (int code = 50; code <= 59; code++) {
            result[index++] = code;
        }
        assert index == result.length;
        return result;
    }

    private static boolean isBinaryChainCode(int code) {
        return (code >= 0 && code <= 3)
                || code == 31
                || (code >= 50 && code <= 59);
    }

    private static float[] chainPrimary(int code, int step, int length) {
        if (code == 15 || code == 23 || code == 24 || code == 59) {
            return generalValues(step, length, 0.09f, 0.25f);
        }
        if (code == 28) {
            return generalValues(step, length, 0.08f, -0.70f);
        }
        if (code == 14 || code == 22 || code == 26 || code == 27) {
            return generalValues(step, length, 0.10f, -1.0f);
        }
        if (code == 55) {
            return generalValues(step, length, 0.08f, 0.30f);
        }
        return generalValues(step, length, 0.17f, -1.35f);
    }

    private static float[] chainSecondary(int code, int step, int length) {
        float[] values;
        if (code == 31) {
            values = new float[length];
            for (int i = 0; i < length; i++) {
                values[i] = 0.05f + (i % 4) * 0.05f;
            }
            return values;
        }
        if (code == 59) {
            values = new float[length];
            for (int i = 0; i < length; i++) {
                values[i] = 0.5f + (i % 4) * 0.5f;
            }
            return values;
        }

        values = generalValues(step + 13, length, 0.11f, 0.35f);
        if (code == 3 || code == 58) {
            for (int i = 0; i < length; i += 5) {
                values[i] = 0.0f;
            }
        } else if (code == 52 || code == 54) {
            for (int i = 0; i < length; i++) {
                if (Math.abs(values[i]) < 0.15f) {
                    values[i] = 0.75f;
                }
            }
        }
        return values;
    }

    private static float[] generalValues(
            int step, int length, float scale, float offset) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            int bucket = Math.floorMod(i * 7 + step * 3, 17);
            values[i] = offset + bucket * scale;
        }
        return values;
    }

    private static float[] activationBackwardValues(int step, int length) {
        final float[] boundaryValues = {
                -6.0f, -4.0f, -2.5f, -1.0f, -0.5f, 0.0f,
                0.75f, 1.0f, 2.5f, 4.0f, 6.0f, 7.0f
        };
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = boundaryValues[
                    Math.floorMod(i * 5 + step, boundaryValues.length)];
        }
        return values;
    }

    private static float[] activationBackwardOracle(
            float[] input,
            float[] gradient,
            DoubleUnaryOperator derivative) {
        assertEquals(input.length, gradient.length,
                "Activation backward input/gradient length");
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) (gradient[i]
                    * derivative.applyAsDouble(input[i]));
        }
        return output;
    }

    private static float[] swishMulBackwardOracle(
            float[] x,
            float[] y,
            float[] gradient,
            int outputIndex) {
        assertEquals(x.length, y.length, "swish_mul_bp x/y length");
        assertEquals(x.length, gradient.length, "swish_mul_bp gradient length");
        assertTrue(outputIndex == 0 || outputIndex == 1,
                "swish_mul_bp output index must select gradX or gradY");
        float[] output = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-x[i]));
            output[i] = outputIndex == 0
                    ? (float) (gradient[i] * y[i]
                            * (sigmoid + x[i] * sigmoid * (1.0 - sigmoid)))
                    : (float) (gradient[i] * x[i] * sigmoid);
        }
        return output;
    }

    private static float[] rmsNormBackwardOracle(
            float[] input,
            float[] gradient,
            int rows,
            int features,
            double epsilon) {
        assertEquals(rows * features, input.length,
                "rms_norm_bp input shape");
        assertEquals(input.length, gradient.length,
                "rms_norm_bp gradient shape");
        float[] output = new float[input.length];
        for (int row = 0; row < rows; row++) {
            int offset = row * features;
            double squareSum = 0.0;
            double dot = 0.0;
            for (int feature = 0; feature < features; feature++) {
                double value = input[offset + feature];
                squareSum += value * value;
                dot += value * gradient[offset + feature];
            }
            double inverseRms = 1.0
                    / Math.sqrt(squareSum / features + epsilon);
            double correction = dot / features
                    * inverseRms * inverseRms * inverseRms;
            for (int feature = 0; feature < features; feature++) {
                int index = offset + feature;
                output[index] = (float) (gradient[index] * inverseRms
                        - input[index] * correction);
            }
        }
        return output;
    }

    private static float[] fusedLayerNormBackwardOracle(
            float[] input,
            float[] gain,
            float[] gradient,
            double epsilon,
            int outputIndex) {
        assertEquals(input.length, gain.length,
                "fused_layer_norm_bp input/gain shape");
        assertEquals(input.length, gradient.length,
                "fused_layer_norm_bp input/gradient shape");
        assertTrue(outputIndex == 0 || outputIndex == 1,
                "fused_layer_norm_bp output index must select dX or dGamma");
        final int features = input.length;
        double mean = 0.0;
        for (float value : input) mean += value;
        mean /= features;
        double variance = 0.0;
        for (float value : input) {
            double centered = value - mean;
            variance += centered * centered;
        }
        variance /= features;
        double inverseStd = 1.0 / Math.sqrt(variance + epsilon);
        double meanDyGain = 0.0;
        double meanDyGainXHat = 0.0;
        for (int feature = 0; feature < features; feature++) {
            double xHat = (input[feature] - mean) * inverseStd;
            double dyGain = gradient[feature] * gain[feature];
            meanDyGain += dyGain;
            meanDyGainXHat += dyGain * xHat;
        }
        meanDyGain /= features;
        meanDyGainXHat /= features;
        float[] output = new float[features];
        for (int feature = 0; feature < features; feature++) {
            double xHat = (input[feature] - mean) * inverseStd;
            output[feature] = outputIndex == 0
                    ? (float) (inverseStd
                            * (gradient[feature] * gain[feature]
                            - meanDyGain
                            - xHat * meanDyGainXHat))
                    : (float) (gradient[feature] * xHat);
        }
        return output;
    }

    private static float[] integerValues(int step, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = Math.floorMod(i * 5 + step * 3, 13) - 6;
        }
        return values;
    }

    private static float[] applyChain(
            float[][] inputs, int[] codes, float clipMin, float clipMax) {
        float[] output = new float[inputs[0].length];
        for (int i = 0; i < output.length; i++) {
            float value = inputs[0][i];
            int secondary = 1;
            for (int code : codes) {
                float rhs = isBinaryChainCode(code)
                        ? inputs[secondary++][i]
                        : 0.0f;
                value = applyChainStep(code, value, rhs, clipMin, clipMax);
            }
            assertEquals(inputs.length, secondary,
                    "Pure-Java chain oracle did not consume every secondary input");
            output[i] = value;
        }
        return output;
    }

    private static float applyChainStep(
            int code, float value, float rhs, float clipMin, float clipMax) {
        switch (code) {
            case 0:
                return value + rhs;
            case 1:
                return value - rhs;
            case 2:
                return value * rhs;
            case 3:
                return rhs == 0.0f ? 0.0f : value / rhs;
            case 10:
                return Math.max(value, 0.0f);
            case 11:
                return sigmoid(value);
            case 12:
                return (float) Math.tanh(value);
            case 13:
                return gelu(value);
            case 14:
                return (float) Math.exp(value);
            case 15:
                return value > 0.0f ? (float) Math.log(value) : -1.0e38f;
            case 16:
                return Math.abs(value);
            case 17:
                return -value;
            case 18:
                return value * value;
            case 19:
                return value >= 0.0f ? (float) Math.sqrt(value) : 0.0f;
            case 20:
            case 21:
                return value * sigmoid(value);
            case 22:
                return value * (float) Math.tanh(Math.log1p(Math.exp(value)));
            case 23:
                return 1.0f / (float) Math.sqrt(value);
            case 24:
                return 1.0f / value;
            case 25:
                return Math.signum(value);
            case 26:
                return erfApprox(value);
            case 27:
                return 1.0f - erfApprox(value);
            case 28:
                return (float) Math.log1p(value);
            case 29:
                return (float) Math.ceil(value);
            case 30:
                return Math.min(Math.max(value, clipMin), clipMax);
            case 31:
                return value >= 0.0f ? value : value * rhs;
            case 32:
                return (float) Math.floor(value);
            case 33:
                return (float) Math.rint(value);
            case 34:
                return (float) Math.sin(value);
            case 35:
                return (float) Math.cos(value);
            case 36:
                return value >= 0.0f ? value : (float) Math.expm1(value);
            case 37:
                return 1.0507009873554805f * (value >= 0.0f
                        ? value
                        : 1.6732632423543772f * (float) Math.expm1(value));
            case 38:
                return (float) Math.log1p(Math.exp(value));
            case 39:
                return value / (1.0f + Math.abs(value));
            case 40:
                return Math.min(1.0f, Math.max(0.0f, value / 6.0f + 0.5f));
            case 41:
                return Math.min(1.0f, Math.max(-1.0f, value));
            case 42:
                return Math.min(6.0f, Math.max(0.0f, value));
            case 50:
                return Math.min(value, rhs);
            case 51:
                return Math.max(value, rhs);
            case 52:
                return value % rhs;
            case 53:
                return (float) Math.atan2(value, rhs);
            case 54:
                return (float) Math.floor(value / rhs);
            case 55:
                return rhs / value;
            case 56:
                return rhs - value;
            case 57:
                float difference = value - rhs;
                return difference * difference;
            case 58:
                return rhs == 0.0f ? 0.0f : value * rhs;
            case 59:
                return (float) Math.pow(value, rhs);
            default:
                throw new AssertionError("Uncatalogued fused-chain opcode " + code);
        }
    }

    private static float sigmoid(float value) {
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private static float gelu(float value) {
        double cube = value * value * value;
        return (float) (0.5 * value
                * (1.0 + Math.tanh(0.7978845608 * (value + 0.044715 * cube))));
    }

    private static float erfApprox(float value) {
        float absolute = Math.abs(value);
        float t = 1.0f / (1.0f + 0.3275911f * absolute);
        float polynomial = 1.061405429f;
        for (float coefficient :
                new float[]{-1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f}) {
            polynomial = coefficient + polynomial * t;
        }
        polynomial *= t;
        float magnitude = 1.0f
                - polynomial * (float) Math.exp(-absolute * absolute);
        return value < 0.0f ? -magnitude : magnitude;
    }

    private static float[] gluOracle(
            float[] input, int rows, int halfWidth, String opName) {
        float[] output = new float[rows * halfWidth];
        for (int row = 0; row < rows; row++) {
            int inputOffset = row * halfWidth * 2;
            int outputOffset = row * halfWidth;
            for (int column = 0; column < halfWidth; column++) {
                float gate = input[inputOffset + column];
                float up = input[inputOffset + halfWidth + column];
                float activated;
                switch (opName) {
                    case "swiglu":
                        activated = gate * sigmoid(gate);
                        break;
                    case "geglu":
                        activated = gelu(gate);
                        break;
                    case "reglu":
                        activated = Math.max(gate, 0.0f);
                        break;
                    default:
                        throw new AssertionError("Unknown GLU variant " + opName);
                }
                output[outputOffset + column] = activated * up;
            }
        }
        return output;
    }

    private static float[] meanSquareOracle(
            float[] input, int rows, int width) {
        float[] output = new float[rows];
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            for (int column = 0; column < width; column++) {
                float value = input[row * width + column];
                sum += value * value;
            }
            output[row] = sum / width;
        }
        return output;
    }

    private static float[] logSumExpAxesZeroTwoOracle(
            float[] input, int batch, int middle, int width) {
        float[] output = new float[middle];
        for (int m = 0; m < middle; m++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int b = 0; b < batch; b++) {
                for (int w = 0; w < width; w++) {
                    maximum = Math.max(
                            maximum, input[(b * middle + m) * width + w]);
                }
            }
            double exponentialSum = 0.0;
            for (int b = 0; b < batch; b++) {
                for (int w = 0; w < width; w++) {
                    exponentialSum += Math.exp(
                            input[(b * middle + m) * width + w] - maximum);
                }
            }
            output[m] = (float) (maximum + Math.log(exponentialSum));
        }
        return output;
    }

    private static float[] statisticalAxesZeroTwoOracle(
            float[] input,
            int batch,
            int middle,
            int width,
            boolean biasCorrected,
            boolean squareRoot) {
        float[] output = new float[middle];
        final int count = batch * width;
        final int denominator = count - (biasCorrected ? 1 : 0);
        assertTrue(denominator > 0,
                "Statistical reduction oracle requires a positive denominator");
        for (int m = 0; m < middle; m++) {
            double mean = 0.0;
            for (int b = 0; b < batch; b++) {
                for (int w = 0; w < width; w++) {
                    mean += input[(b * middle + m) * width + w];
                }
            }
            mean /= count;
            double sumSquares = 0.0;
            for (int b = 0; b < batch; b++) {
                for (int w = 0; w < width; w++) {
                    double delta = input[(b * middle + m) * width + w] - mean;
                    sumSquares += delta * delta;
                }
            }
            double variance = sumSquares / denominator;
            output[m] = (float) (squareRoot ? Math.sqrt(variance) : variance);
        }
        return output;
    }

    private static float[] argReductionValues(
            int step, int batch, int rows, int width) {
        float[] values = new float[batch * rows * width];
        for (int outer = 0; outer < batch * rows; outer++) {
            int base = outer * width;
            for (int column = 0; column < width; column++) {
                values[base + column] =
                        (float) Math.sin((step + 1) * 0.17
                                + outer * 0.41 + column * 0.73);
            }
            int maximumIndex = Math.floorMod(step + outer, width);
            int minimumIndex = Math.floorMod(step + outer + 2, width);
            values[base + maximumIndex] = 100.0f + step + outer;
            values[base + minimumIndex] = -100.0f - step - outer;
            if (outer == 0) {
                // Exercise first-index tie behavior without making every replay
                // choose the same coordinate.
                int tiedMaximum = (maximumIndex + 1) % width;
                if (tiedMaximum != minimumIndex) {
                    values[base + tiedMaximum] = values[base + maximumIndex];
                }
            }
        }
        return values;
    }

    private static INDArray argReductionStridedInput(
            int step, int batch, int rows, int width) {
        INDArray storage = Nd4j.zeros(
                DataType.FLOAT, batch, rows, width * 2L);
        INDArray view = storage.get(
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, 2, width * 2L));
        view.assign(Nd4j.create(
                argReductionValues(step, batch, rows, width),
                new long[]{batch, rows, width}));
        assertTrue(view.isView(), "Index-reduction input must remain a view");
        assertEquals(2L, view.stride(2),
                "Index-reduction feature dimension must have stride two");
        return view;
    }

    private static float[] argReductionOracle(
            float[] input, int outerSize, int width, boolean maximum) {
        float[] output = new float[outerSize];
        for (int outer = 0; outer < outerSize; outer++) {
            int base = outer * width;
            int bestIndex = 0;
            float best = input[base];
            for (int column = 1; column < width; column++) {
                float candidate = input[base + column];
                if (maximum ? candidate > best : candidate < best) {
                    best = candidate;
                    bestIndex = column;
                }
            }
            output[outer] = bestIndex;
        }
        return output;
    }

    private static float[] transformedReductionValues(
            DataType dataType, int step, int length) {
        float[] values = new float[length];
        for (int index = 0; index < length; index++) {
            int integral =
                    Math.floorMod(index * 5 + step * 3, 13) - 6;
            values[index] = dataType == DataType.INT32
                    ? integral
                    : integral + (index % 3) * 0.125f;
        }
        return values;
    }

    private static INDArray transformedReductionStridedInput(
            DataType dataType,
            int step,
            int batch,
            int rows,
            int width) {
        INDArray storage = Nd4j.zeros(
                dataType, batch, rows, width * 2L);
        INDArray view = storage.get(
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, 2, width * 2L));
        view.assign(Nd4j.create(
                transformedReductionValues(
                        dataType, step, batch * rows * width),
                new long[]{batch, rows, width},
                dataType));
        assertTrue(view.isView(),
                "Transformed-reduction input must remain a view");
        assertEquals(2L, view.stride(2),
                "Transformed-reduction feature dimension must have stride two");
        return view;
    }

    private static float[] transformedLastAxisReductionOracle(
            float[] input,
            int outerSize,
            int width,
            boolean squareAndSum) {
        float[] output = new float[outerSize];
        for (int outer = 0; outer < outerSize; outer++) {
            int base = outer * width;
            double accumulator = 0.0;
            for (int column = 0; column < width; column++) {
                double value = input[base + column];
                if (squareAndSum) {
                    accumulator += value * value;
                } else {
                    accumulator = Math.max(accumulator, Math.abs(value));
                }
            }
            output[outer] = (float) accumulator;
        }
        return output;
    }

    private static float[] baseLastAxisReductionOracle(
            float[] input,
            int outerSize,
            int width,
            String opName) {
        assertEquals(outerSize * width, input.length,
                "Base-reduction oracle input length");
        float[] output = new float[outerSize];
        for (int outer = 0; outer < outerSize; outer++) {
            double accumulator;
            if (opName.equals("reduce_prod")) {
                accumulator = 1.0;
            } else if (opName.equals("reduce_max")) {
                accumulator = -Double.MAX_VALUE;
            } else if (opName.equals("reduce_min")) {
                accumulator = Double.MAX_VALUE;
            } else {
                accumulator = 0.0;
            }
            for (int column = 0; column < width; column++) {
                double value = input[outer * width + column];
                if (opName.equals("reduce_prod")) {
                    accumulator *= value;
                } else if (opName.equals("reduce_max")) {
                    accumulator = Math.max(accumulator, value);
                } else if (opName.equals("reduce_min")) {
                    accumulator = Math.min(accumulator, value);
                } else if (opName.equals("reduce_sum") ||
                        opName.equals("reduce_mean")) {
                    accumulator += value;
                } else {
                    throw new IllegalArgumentException(
                            "Unsupported base reduction oracle: " + opName);
                }
            }
            if (opName.equals("reduce_mean")) {
                accumulator /= width;
            }
            output[outer] = (float) accumulator;
        }
        return output;
    }

    private static float[] absoluteArgReductionValues(
            DataType dataType,
            int step,
            int batch,
            int rows,
            int width) {
        assertEquals(7, width,
                "Absolute-index test templates require width seven");
        final float[] template = dataType == DataType.INT32
                ? new float[]{
                        Integer.MIN_VALUE, -1.0f, 7.0f, 1.0f,
                        -23.0f, 9.0f, -4.0f}
                : new float[]{
                        9.0f, -1.0f, -5.0f, 1.0f,
                        -12.0f, 12.0f, -7.0f};
        float[] values = new float[batch * rows * width];
        for (int outer = 0; outer < batch * rows; outer++) {
            int shift = Math.floorMod(step + outer * 2, width);
            for (int column = 0; column < width; column++) {
                int templateIndex =
                        Math.floorMod(column - shift, width);
                values[outer * width + column] =
                        template[templateIndex];
            }
        }
        return values;
    }

    private static INDArray absoluteArgReductionStridedInput(
            DataType dataType,
            int step,
            int batch,
            int rows,
            int width) {
        INDArray storage = Nd4j.zeros(
                dataType, batch, rows, width * 2L);
        INDArray view = storage.get(
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, 2, width * 2L));
        view.assign(Nd4j.create(
                absoluteArgReductionValues(
                        dataType, step, batch, rows, width),
                new long[]{batch, rows, width},
                dataType));
        assertTrue(view.isView(),
                "Absolute-index input must remain a view");
        assertEquals(2L, view.stride(2),
                "Absolute-index feature dimension must have stride two");
        return view;
    }

    private static float[] absoluteArgReductionOracle(
            float[] input,
            int outerSize,
            int width,
            boolean maximum) {
        float[] output = new float[outerSize];
        for (int outer = 0; outer < outerSize; outer++) {
            int base = outer * width;
            int bestIndex = 0;
            float bestMagnitude = Math.abs(input[base]);
            for (int column = 1; column < width; column++) {
                float candidateMagnitude =
                        Math.abs(input[base + column]);
                if (maximum
                        ? candidateMagnitude > bestMagnitude
                        : candidateMagnitude < bestMagnitude) {
                    bestMagnitude = candidateMagnitude;
                    bestIndex = column;
                }
            }
            output[outer] = bestIndex;
        }
        return output;
    }

    private static INDArray denseFortranInput(
            float[] logicalValues, long[] shape) {
        INDArray result =
                Nd4j.createUninitialized(DataType.FLOAT, shape, 'f');
        result.assign(Nd4j.create(logicalValues, shape));
        assertFalse(result.isView(),
                "Fortran raw-copy payload must own its storage");
        assertEquals('f', result.ordering(),
                "Fortran raw-copy payload order");
        long expectedStride = 1L;
        for (int dimension = 0; dimension < shape.length; dimension++) {
            assertEquals(expectedStride, result.stride(dimension),
                    "Fortran raw-copy payload stride at dimension " + dimension);
            expectedStride *= shape[dimension];
        }
        return result;
    }

    private static float[] linearCopyFortranOracle(
            float[] inputLogicalValues,
            long[] inputShape,
            long[] outputShape) {
        long inputLength = 1;
        for (long dimension : inputShape) {
            inputLength *= dimension;
        }
        long outputLength = 1;
        for (long dimension : outputShape) {
            outputLength *= dimension;
        }
        assertEquals(inputLength, outputLength,
                "Full Fortran raw-copy test requires equal element counts");
        assertEquals(inputLength, inputLogicalValues.length,
                "Fortran raw-copy input value count");

        float[] physicalValues = new float[inputLogicalValues.length];
        for (int physicalOffset = 0;
                physicalOffset < physicalValues.length;
                physicalOffset++) {
            long remaining = physicalOffset;
            long cLinearIndex = 0;
            long cStride = 1;
            long[] coordinates = new long[inputShape.length];
            for (int dimension = 0;
                    dimension < inputShape.length;
                    dimension++) {
                coordinates[dimension] =
                        remaining % inputShape[dimension];
                remaining /= inputShape[dimension];
            }
            for (int dimension = inputShape.length - 1;
                    dimension >= 0;
                    dimension--) {
                cLinearIndex += coordinates[dimension] * cStride;
                cStride *= inputShape[dimension];
            }
            physicalValues[physicalOffset] =
                    inputLogicalValues[(int) cLinearIndex];
        }

        float[] outputLogicalValues = new float[physicalValues.length];
        for (int cLinearIndex = 0;
                cLinearIndex < outputLogicalValues.length;
                cLinearIndex++) {
            long remaining = cLinearIndex;
            long[] coordinates = new long[outputShape.length];
            for (int dimension = outputShape.length - 1;
                    dimension >= 0;
                    dimension--) {
                coordinates[dimension] =
                        remaining % outputShape[dimension];
                remaining /= outputShape[dimension];
            }

            long physicalOffset = 0;
            long fortranStride = 1;
            for (int dimension = 0;
                    dimension < outputShape.length;
                    dimension++) {
                physicalOffset +=
                        coordinates[dimension] * fortranStride;
                fortranStride *= outputShape[dimension];
            }
            outputLogicalValues[cLinearIndex] =
                    physicalValues[(int) physicalOffset];
        }
        return outputLogicalValues;
    }

    private static float[] biasAddNchwOracle(
            float[] input,
            float[] bias,
            int batch,
            int channels,
            int width) {
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int w = 0; w < width; w++) {
                    int index = (b * channels + channel) * width + w;
                    output[index] = input[index] + bias[channel];
                }
            }
        }
        return output;
    }

    private static float[] preluOracle(
            float[] input,
            float[] alpha,
            int batch,
            int channels,
            int width) {
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int w = 0; w < width; w++) {
                    int index = (b * channels + channel) * width + w;
                    float value = input[index];
                    output[index] = value >= 0.0f
                            ? value
                            : value * alpha[channel * width + w];
                }
            }
        }
        return output;
    }

    private static float[] positiveVarianceValues(int step, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = 0.35f + 0.07f * Math.floorMod(i * 3 + step, 9);
        }
        return values;
    }

    private static float[] batchNormOracle(
            float[] input,
            float[] mean,
            float[] variance,
            float[] gamma,
            float[] beta,
            int batch,
            int channels,
            int width,
            double epsilon) {
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int w = 0; w < width; w++) {
                    int index = (b * channels + channel) * width + w;
                    output[index] = (float) (gamma[w]
                            * ((input[index] - mean[w])
                            / Math.sqrt(variance[w] + epsilon))
                            + beta[w]);
                }
            }
        }
        return output;
    }

    private static float[] constantValues(int length, float value) {
        float[] output = new float[length];
        java.util.Arrays.fill(output, value);
        return output;
    }

    private static float[] oneHotIndexValues(int step) {
        float[] values = new float[6];
        for (int i = 0; i < values.length; i++) {
            int integral = Math.floorMod(step + i * 3, 7) - 1;
            values[i] = integral < 0
                    ? integral + 0.25f
                    : integral + (i % 2 == 0 ? 0.75f : 0.125f);
        }
        return values;
    }

    private static float[] oneHotOracle(
            float[] indices,
            int rows,
            int columns,
            int axis,
            int depth,
            float on,
            float off) {
        assertEquals(1, axis, "This strict oracle exercises axis=1");
        float[] output = new float[rows * depth * columns];
        for (int row = 0; row < rows; row++) {
            for (int category = 0; category < depth; category++) {
                for (int column = 0; column < columns; column++) {
                    int inputIndex = row * columns + column;
                    int outputIndex =
                            (row * depth + category) * columns + column;
                    output[outputIndex] =
                            category == (int) indices[inputIndex] ? on : off;
                }
            }
        }
        return output;
    }

    private static float[] eyeOracle(
            int batches, int rows, int columns) {
        float[] output = new float[batches * rows * columns];
        int diagonal = Math.min(rows, columns);
        for (int batch = 0; batch < batches; batch++) {
            for (int coordinate = 0; coordinate < diagonal; coordinate++) {
                int logicalIndex =
                        (batch * rows + coordinate) * columns + coordinate;
                output[logicalIndex] = 1.0f;
            }
        }
        return output;
    }

    private static float[] linSpaceOracle(
            double start, double end, int steps) {
        float[] output = new float[steps];
        double increment = steps == 1 ? 0.0 : (end - start) / (steps - 1.0);
        for (int i = 0; i < steps; i++) {
            output[i] = (float) (start + increment * i);
        }
        return output;
    }

    private static float[] rangeOracle(
            double start, double delta, int steps) {
        float[] output = new float[steps];
        for (int i = 0; i < steps; i++) {
            output[i] = (float) (start + delta * i);
        }
        return output;
    }

    private static float[] triuOracle(
            float[] input,
            int batch,
            int rows,
            int columns,
            int diagonal) {
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    int index = (b * rows + row) * columns + column;
                    output[index] = column - row >= diagonal ? input[index] : 0.0f;
                }
            }
        }
        return output;
    }

    private static float[] splitShardOracle(
            float[] input,
            int rows,
            int width,
            int shards,
            int selectedShard) {
        assertEquals(0, width % shards,
                "Split oracle requires equal shards");
        int shardWidth = width / shards;
        float[] output = new float[rows * shardWidth];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < shardWidth; column++) {
                output[row * shardWidth + column] =
                        input[row * width + selectedShard * shardWidth + column];
            }
        }
        return output;
    }

    private static float[] unstackSliceOracle(
            float[] input,
            int batch,
            int slices,
            int width,
            int selectedSlice) {
        float[] output = new float[batch * width];
        for (int b = 0; b < batch; b++) {
            for (int column = 0; column < width; column++) {
                output[b * width + column] =
                        input[(b * slices + selectedSlice) * width + column];
            }
        }
        return output;
    }

    private static float[] batchedMatmulOracle(
            float[] left,
            float[] right,
            int batch,
            int rows,
            int inner,
            int columns) {
        float[] output = new float[batch * rows * columns];
        for (int b = 0; b < batch; b++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float sum = 0.0f;
                    for (int k = 0; k < inner; k++) {
                        sum += left[(b * rows + row) * inner + k]
                                * right[(b * inner + k) * columns + column];
                    }
                    output[(b * rows + row) * columns + column] = sum;
                }
            }
        }
        return output;
    }

    private static float[] batchedMatrixProductOracle(
            float[] left,
            float[] right,
            int rows,
            int inner,
            int columns,
            boolean transposeLeft,
            boolean transposeRight,
            float alpha) {
        assertEquals(rows * inner, left.length,
                "Batched matrix-list left operand length");
        assertEquals(inner * columns, right.length,
                "Batched matrix-list right operand length");
        float[] output = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                float sum = 0.0f;
                for (int k = 0; k < inner; k++) {
                    int leftIndex = transposeLeft
                            ? k * rows + row
                            : row * inner + k;
                    int rightIndex = transposeRight
                            ? column * inner + k
                            : k * columns + column;
                    sum += left[leftIndex] * right[rightIndex];
                }
                output[row * columns + column] = alpha * sum;
            }
        }
        return output;
    }

    private static float[] attentionProjectionOracle(
            float[] attention,
            float[] weight,
            float[] bias,
            int batch,
            int sequence,
            int hidden,
            int outputSize) {
        float[] output = new float[batch * sequence * outputSize];
        for (int b = 0; b < batch; b++) {
            for (int token = 0; token < sequence; token++) {
                for (int column = 0; column < outputSize; column++) {
                    float sum = bias == null ? 0.0f : bias[column];
                    for (int k = 0; k < hidden; k++) {
                        sum += attention[(b * sequence + token) * hidden + k]
                                * weight[k * outputSize + column];
                    }
                    output[(b * sequence + token) * outputSize + column] = sum;
                }
            }
        }
        return output;
    }

    private static float[] logSoftmaxOracle(
            float[] input, int rows, int width) {
        float[] output = new float[input.length];
        for (int row = 0; row < rows; row++) {
            int offset = row * width;
            double maximum = Double.NEGATIVE_INFINITY;
            for (int column = 0; column < width; column++) {
                maximum = Math.max(maximum, input[offset + column]);
            }
            double exponentialSum = 0.0;
            for (int column = 0; column < width; column++) {
                exponentialSum += Math.exp(input[offset + column] - maximum);
            }
            double logDenominator = maximum + Math.log(exponentialSum);
            for (int column = 0; column < width; column++) {
                output[offset + column] =
                        (float) (input[offset + column] - logDenominator);
            }
        }
        return output;
    }

    private static float[] biasDropoutResidualOracle(
            float[] input,
            float[] bias,
            float[] residual,
            int rows,
            int width) {
        float[] output = new float[input.length];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < width; column++) {
                int index = row * width + column;
                output[index] = input[index] + bias[column] + residual[index];
            }
        }
        return output;
    }

    private static float[] softplusExtremeValues(int step) {
        float[] values = {
                -120.0f, -100.0f, -88.0f, -50.0f,
                -20.0f, -2.0f, 0.0f, 2.0f,
                20.0f, 50.0f, 88.0f, 120.0f
        };
        float delta = (Math.floorMod(step, 5) - 2) * 0.125f;
        for (int i = 0; i < values.length; i++) {
            values[i] += delta;
        }
        return values;
    }

    private static float[] softplusOracle(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            double value = input[i];
            output[i] = (float) (Math.max(value, 0.0)
                    + Math.log1p(Math.exp(-Math.abs(value))));
            assertTrue(Float.isFinite(output[i]),
                    "Stable softplus oracle must remain finite for input " + value);
        }
        return output;
    }

    private static float[] mapValues(
            float[] input, DoubleUnaryOperator operation) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) operation.applyAsDouble(input[i]);
        }
        return output;
    }

    private static double rationalTanh(double value) {
        double d = value * (2.0 / 3.0);
        double d2 = d * d;
        double denominator =
                1.0 + Math.abs(d) + d2 + 1.41645 * d2 * d2;
        double approximation = 1.0 - 1.0 / denominator;
        return 1.7159 * Math.signum(d) * approximation;
    }

    private static float[] roundingValues(int step) {
        float[] base = {
                -3.5f, -2.5f, -1.5f, -0.5f,
                0.5f, 1.5f, 2.5f, 3.5f,
                -4.25f, -0.25f, 0.25f, 4.25f
        };
        float[] values = new float[base.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = base[Math.floorMod(i + step, base.length)];
        }
        return values;
    }

    private static float[] nonNegativeIntegerValues(int step, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = Math.floorMod(i * 3 + step, 9);
        }
        return values;
    }

    private static float[] signedIntegerValues(int step, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = Math.floorMod(i * 5 + step * 2, 11) - 5;
        }
        return values;
    }

    private static float[] divisorValues(int step, int rows) {
        float[] values = new float[rows];
        for (int row = 0; row < rows; row++) {
            float magnitude =
                    1.25f + 0.25f * Math.floorMod(row * 2 + step, 5);
            values[row] =
                    Math.floorMod(row + step, 2) == 0 ? magnitude : -magnitude;
        }
        return values;
    }

    private static float[] realDivisionOracle(
            float[] numerator,
            float[] denominatorByRow,
            int batch,
            int rows,
            int width) {
        float[] output = new float[numerator.length];
        for (int b = 0; b < batch; b++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < width; column++) {
                    int index = (b * rows + row) * width + column;
                    output[index] =
                            numerator[index] / denominatorByRow[row];
                }
            }
        }
        return output;
    }

    private static float[] pairwiseOracle(
            float[] left,
            float[] right,
            DoubleBinaryOperator operation) {
        assertEquals(left.length, right.length, "Pairwise oracle lengths");
        float[] output = new float[left.length];
        for (int i = 0; i < left.length; i++) {
            output[i] =
                    (float) operation.applyAsDouble(left[i], right[i]);
        }
        return output;
    }

    private static float[] siluAndMulOracle(float[] gate, float[] up) {
        assertEquals(gate.length, up.length, "SiLU-and-mul oracle lengths");
        float[] output = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            double value = gate[i];
            double sigmoid = 1.0 / (1.0 + Math.exp(-value));
            output[i] = (float) (value * sigmoid * up[i]);
        }
        return output;
    }

    private static float[] rankThreeNormOracle(
            float[] input,
            int batch,
            int rows,
            int width,
            boolean euclidean) {
        assertEquals(
                batch * rows * width,
                input.length,
                "Rank-three norm oracle input length");
        float[] output = new float[rows];
        for (int row = 0; row < rows; row++) {
            double accumulator = 0.0;
            for (int b = 0; b < batch; b++) {
                for (int column = 0; column < width; column++) {
                    double value = input[(b * rows + row) * width + column];
                    accumulator += euclidean
                            ? value * value
                            : Math.abs(value);
                }
            }
            output[row] = (float) (euclidean
                    ? Math.sqrt(accumulator)
                    : accumulator);
        }
        return output;
    }

    private static float[] groupedAttentionOracle(
            float[] query,
            float[] key,
            float[] value,
            int batch,
            int querySteps,
            int keySteps,
            int queryHeads,
            int keyValueHeads,
            int headDimension,
            double scale,
            boolean causal) {
        assertEquals(
                0,
                queryHeads % keyValueHeads,
                "Query heads must divide evenly into KV groups");
        assertEquals(
                batch * querySteps * queryHeads * headDimension,
                query.length,
                "Attention query length");
        assertEquals(
                batch * keySteps * keyValueHeads * headDimension,
                key.length,
                "Attention key length");
        assertEquals(key.length, value.length, "Attention key/value lengths");

        float[] output = new float[query.length];
        int headsPerGroup = queryHeads / keyValueHeads;
        int causalOffset = Math.max(keySteps - querySteps, 0);
        double[] scores = new double[keySteps];

        for (int b = 0; b < batch; b++) {
            for (int queryStep = 0; queryStep < querySteps; queryStep++) {
                int lastVisibleKey = causal
                        ? Math.min(keySteps - 1, queryStep + causalOffset)
                        : keySteps - 1;
                for (int queryHead = 0; queryHead < queryHeads; queryHead++) {
                    int keyValueHead = queryHead / headsPerGroup;
                    double maximum = Double.NEGATIVE_INFINITY;
                    for (int keyStep = 0; keyStep <= lastVisibleKey; keyStep++) {
                        double dot = 0.0;
                        for (int feature = 0;
                             feature < headDimension;
                             feature++) {
                            int queryIndex =
                                    (((b * querySteps + queryStep)
                                            * queryHeads + queryHead)
                                            * headDimension + feature);
                            int keyIndex =
                                    (((b * keySteps + keyStep)
                                            * keyValueHeads + keyValueHead)
                                            * headDimension + feature);
                            dot += query[queryIndex] * key[keyIndex];
                        }
                        scores[keyStep] = dot * scale;
                        maximum = Math.max(maximum, scores[keyStep]);
                    }

                    double denominator = 0.0;
                    for (int keyStep = 0; keyStep <= lastVisibleKey; keyStep++) {
                        denominator += Math.exp(scores[keyStep] - maximum);
                    }

                    for (int feature = 0;
                         feature < headDimension;
                         feature++) {
                        double weighted = 0.0;
                        for (int keyStep = 0;
                             keyStep <= lastVisibleKey;
                             keyStep++) {
                            double probability =
                                    Math.exp(scores[keyStep] - maximum)
                                            / denominator;
                            int valueIndex =
                                    (((b * keySteps + keyStep)
                                            * keyValueHeads + keyValueHead)
                                            * headDimension + feature);
                            weighted += probability * value[valueIndex];
                        }
                        int outputIndex =
                                (((b * querySteps + queryStep)
                                        * queryHeads + queryHead)
                                        * headDimension + feature);
                        output[outputIndex] = (float) weighted;
                    }
                }
            }
        }
        return output;
    }

    private static float[] positionValues(int step, int length, int offset) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = Math.floorMod(i * 2 + step + offset, 11);
        }
        return values;
    }

    private static float[] mropeOracle(
            float[] input,
            float[] positionT,
            float[] positionH,
            float[] positionW,
            int batch,
            int sequence,
            int heads,
            int headDimension,
            int sectionT,
            int sectionH,
            int sectionW,
            boolean interleaved,
            double frequencyBase) {
        assertEquals(headDimension, sectionT + sectionH + sectionW,
                "M-RoPE section sizes must span head dimension");
        assertEquals(0, headDimension % 2,
                "M-RoPE head dimension must contain complete rotation pairs");

        float[] output = new float[input.length];
        int halfDimension = headDimension / 2;
        int halfTemporal = sectionT / 2;
        int halfHeight = sectionH / 2;
        int interleavedSectionSize = (headDimension + 2) / 3;

        for (int b = 0; b < batch; b++) {
            for (int token = 0; token < sequence; token++) {
                int positionIndex = b * sequence + token;
                for (int head = 0; head < heads; head++) {
                    int base = ((b * sequence + token) * heads + head) * headDimension;
                    for (int dimension = 0;
                         dimension < halfDimension;
                         dimension++) {
                        float position;
                        int localDimension;
                        int sectionSize;
                        double effectiveBase;

                        if (interleaved) {
                            int selector = dimension % 3;
                            position = selector == 0
                                    ? positionT[positionIndex]
                                    : selector == 1
                                            ? positionH[positionIndex]
                                            : positionW[positionIndex];
                            localDimension = dimension / 3;
                            sectionSize = interleavedSectionSize;
                            // This is the canonical CPU/CUDA interleaved contract.
                            effectiveBase = 10000.0;
                        } else if (dimension < halfTemporal) {
                            position = positionT[positionIndex];
                            localDimension = dimension;
                            sectionSize = sectionT;
                            effectiveBase = frequencyBase;
                        } else if (dimension < halfTemporal + halfHeight) {
                            position = positionH[positionIndex];
                            localDimension = dimension - halfTemporal;
                            sectionSize = sectionH;
                            effectiveBase = frequencyBase;
                        } else {
                            position = positionW[positionIndex];
                            localDimension = dimension - halfTemporal - halfHeight;
                            sectionSize = sectionW;
                            effectiveBase = frequencyBase;
                        }

                        double exponent = 2.0 * localDimension / sectionSize;
                        double angle = position / Math.pow(effectiveBase, exponent);
                        double cosine = Math.cos(angle);
                        double sine = Math.sin(angle);
                        int firstIndex = base + dimension;
                        int secondIndex = firstIndex + halfDimension;
                        float first = input[firstIndex];
                        float second = input[secondIndex];
                        output[firstIndex] =
                                (float) (first * cosine - second * sine);
                        output[secondIndex] =
                                (float) (first * sine + second * cosine);
                    }
                }
            }
        }
        return output;
    }

    private static INDArray featureStridedInput(
            int step, int batch, int sequence, int headDimension) {
        INDArray storage = Nd4j.zeros(
                DataType.FLOAT, batch, sequence, headDimension * 2L);
        INDArray view = storage.get(
                NDArrayIndex.all(),
                NDArrayIndex.all(),
                NDArrayIndex.interval(0, 2, headDimension * 2L));
        view.assign(Nd4j.create(
                generalValues(
                        step,
                        batch * sequence * headDimension,
                        0.0375f,
                        -0.65f),
                new long[]{batch, sequence, headDimension}));
        assertTrue(view.isView(), "RoPE input must remain a real NDArray view");
        assertEquals(2L, view.stride(2),
                "RoPE feature dimension must have stride two");
        return view;
    }

    private static INDArray rankOneStridedInput(float[] values) {
        INDArray storage = Nd4j.zeros(DataType.FLOAT, values.length * 2L);
        INDArray view = storage.get(
                NDArrayIndex.interval(0, 2, values.length * 2L));
        view.assign(Nd4j.create(values, new long[]{values.length}));
        assertTrue(view.isView(), "Rank-one input must remain a real NDArray view");
        assertEquals(2L, view.stride(0),
                "Rank-one input must have stride two");
        return view;
    }

    private static float[] ropeOracle(
            float[] input,
            int batch,
            int sequence,
            int heads,
            int headDimension,
            int mode,
            int positionOffset,
            int requestedRotaryDimensions,
            double frequencyBase,
            double frequencyScale) {
        int rotaryDimensions = requestedRotaryDimensions > 0
                && requestedRotaryDimensions < headDimension
                ? requestedRotaryDimensions
                : headDimension;
        int half = rotaryDimensions / 2;
        float[] output = input.clone();
        for (int b = 0; b < batch; b++) {
            for (int token = 0; token < sequence; token++) {
                double position = positionOffset + token;
                for (int head = 0; head < heads; head++) {
                    int base = ((b * sequence + token) * heads + head)
                            * headDimension;
                    for (int pair = 0; pair < half; pair++) {
                        double inverseFrequency = frequencyScale
                                / Math.pow(
                                        frequencyBase,
                                        2.0 * pair / rotaryDimensions);
                        double angle = position * inverseFrequency;
                        int first = mode == 1 ? 2 * pair : pair;
                        int second = mode == 1 ? 2 * pair + 1 : pair + half;
                        float firstValue = input[base + first];
                        float secondValue = input[base + second];
                        double cosine = Math.cos(angle);
                        double sine = Math.sin(angle);
                        output[base + first] =
                                (float) (firstValue * cosine - secondValue * sine);
                        output[base + second] =
                                (float) (firstValue * sine + secondValue * cosine);
                    }
                }
            }
        }
        return output;
    }

    private static float[] ropeBackwardOracle(
            float[] gradient,
            int batch,
            int sequence,
            int heads,
            int headDimension,
            int mode,
            int positionOffset,
            int requestedRotaryDimensions,
            double frequencyBase,
            double frequencyScale) {
        int rotaryDimensions = requestedRotaryDimensions > 0
                && requestedRotaryDimensions < headDimension
                ? requestedRotaryDimensions
                : headDimension;
        int half = rotaryDimensions / 2;
        float[] output = gradient.clone();
        for (int b = 0; b < batch; b++) {
            for (int token = 0; token < sequence; token++) {
                double position = positionOffset + token;
                for (int head = 0; head < heads; head++) {
                    int base = ((b * sequence + token) * heads + head)
                            * headDimension;
                    for (int pair = 0; pair < half; pair++) {
                        double inverseFrequency = frequencyScale
                                / Math.pow(
                                        frequencyBase,
                                        2.0 * pair / rotaryDimensions);
                        double angle = position * inverseFrequency;
                        int first = mode == 1 ? 2 * pair : pair;
                        int second = mode == 1 ? 2 * pair + 1 : pair + half;
                        float firstGradient = gradient[base + first];
                        float secondGradient = gradient[base + second];
                        double cosine = Math.cos(angle);
                        double sine = Math.sin(angle);
                        output[base + first] = (float) (
                                firstGradient * cosine
                                        + secondGradient * sine);
                        output[base + second] = (float) (
                                -firstGradient * sine
                                        + secondGradient * cosine);
                    }
                }
            }
        }
        return output;
    }

    private static float[] alibiOracle(
            float[] input,
            int batch,
            int heads,
            int queries,
            int keys) {
        float[] output = input.clone();
        double base = Math.pow(2.0, -8.0 / heads);
        for (int b = 0; b < batch; b++) {
            for (int head = 0; head < heads; head++) {
                double slope = Math.pow(base, head + 1);
                for (int query = 0; query < queries; query++) {
                    for (int key = 0; key < keys; key++) {
                        int index =
                                ((b * heads + head) * queries + query) * keys + key;
                        output[index] = (float) (input[index]
                                - slope * Math.abs(query - key));
                    }
                }
            }
        }
        return output;
    }

    private static float[] visionTokenIds(
            int step, int batch, int sequence, int targetTokenId) {
        float[] tokenIds = new float[batch * sequence];
        for (int b = 0; b < batch; b++) {
            for (int token = 0; token < sequence; token++) {
                boolean target = Math.floorMod(token + b + step, 2) == 0;
                tokenIds[b * sequence + token] =
                        target ? targetTokenId : 10 + b * sequence + token;
            }
        }
        return tokenIds;
    }

    private static float[] visionMergeOracle(
            float[] text,
            float[] vision,
            float[] tokenIds,
            int batch,
            int sequence,
            int hidden,
            int visionTokens,
            int targetTokenId) {
        float[] output = text.clone();
        for (int b = 0; b < batch; b++) {
            int visionIndex = 0;
            for (int token = 0; token < sequence; token++) {
                boolean useVision =
                        (int) tokenIds[b * sequence + token] == targetTokenId
                                && visionIndex < visionTokens;
                if (useVision) {
                    System.arraycopy(
                            vision,
                            (b * visionTokens + visionIndex) * hidden,
                            output,
                            (b * sequence + token) * hidden,
                            hidden);
                    visionIndex++;
                }
            }
        }
        return output;
    }

    private static float[] gatherNdIndices(int step) {
        return new float[]{
                -3.0f,
                Math.floorMod(step, 3),
                Math.floorMod(step + 1, 3),
                7.0f
        };
    }

    private static float[] gatherNdOracle(
            float[] input, float[] indices, int rows, int columns) {
        float[] output = new float[indices.length * columns];
        for (int i = 0; i < indices.length; i++) {
            int row = (int) indices[i];
            row = Math.max(0, Math.min(rows - 1, row));
            System.arraycopy(
                    input, row * columns,
                    output, i * columns,
                    columns);
        }
        return output;
    }

    private static float[] embeddingIndices(int step) {
        float[] indices = new float[4];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = Math.floorMod(step + i * 2, 5);
        }
        return indices;
    }

    private static float[] embeddingLookupOracle(
            float[] table,
            float[] indices,
            int rows,
            int height,
            int width) {
        assertEquals(rows * height * width, table.length,
                "Embedding oracle table length");
        int rowWidth = height * width;
        float[] output = new float[indices.length * rowWidth];
        for (int i = 0; i < indices.length; i++) {
            int row = Math.max(0, Math.min(rows - 1, (int) indices[i]));
            System.arraycopy(
                    table, row * rowWidth,
                    output, i * rowWidth,
                    rowWidth);
        }
        return output;
    }

    private static float[] tileOracle(
            float[] input,
            int rows,
            int columns,
            int rowRepetitions,
            int columnRepetitions) {
        int outputRows = rows * rowRepetitions;
        int outputColumns = columns * columnRepetitions;
        float[] output = new float[outputRows * outputColumns];
        for (int row = 0; row < outputRows; row++) {
            for (int column = 0; column < outputColumns; column++) {
                output[row * outputColumns + column] =
                        input[(row % rows) * columns + column % columns];
            }
        }
        return output;
    }

    private static float[] repeatOracle(
            float[] input, int rows, int columns, int[] repetitions) {
        int outputColumns = 0;
        for (int repetition : repetitions) {
            outputColumns += repetition;
        }
        float[] output = new float[rows * outputColumns];
        for (int row = 0; row < rows; row++) {
            int outputColumn = 0;
            for (int column = 0; column < columns; column++) {
                for (int copy = 0; copy < repetitions[column]; copy++) {
                    output[row * outputColumns + outputColumn++] =
                            input[row * columns + column];
                }
            }
        }
        return output;
    }

    private static float[] reverseRankThreeOracle(
            float[] input, int first, int second, int third) {
        float[] output = new float[input.length];
        for (int i = 0; i < first; i++) {
            for (int j = 0; j < second; j++) {
                for (int k = 0; k < third; k++) {
                    int outputIndex = (i * second + j) * third + k;
                    int inputIndex =
                            (((first - 1 - i) * second + j) * third)
                                    + (third - 1 - k);
                    output[outputIndex] = input[inputIndex];
                }
            }
        }
        return output;
    }

    private static float[] rollLinearShift(int step) {
        switch (Math.floorMod(step, 3)) {
            case 0:
                return new float[]{-2.0f};
            case 1:
                return new float[]{1.0f};
            default:
                return new float[]{7.0f};
        }
    }

    private static float[] rollAxisShifts(int step) {
        switch (Math.floorMod(step, 3)) {
            case 0:
                return new float[]{1.0f, -1.0f, 2.0f};
            case 1:
                return new float[]{-2.0f, 1.0f, 3.0f};
            default:
                return new float[]{3.0f, -4.0f, -1.0f};
        }
    }

    private static float[] rollAxes(int step) {
        switch (Math.floorMod(step, 3)) {
            case 0:
                return new float[]{0.0f, 2.0f, 0.0f};
            case 1:
                return new float[]{1.0f, -1.0f, 1.0f};
            default:
                return new float[]{-3.0f, 0.0f, 2.0f};
        }
    }

    private static float[] rollLinearOracle(float[] input, int shift) {
        float[] output = new float[input.length];
        int normalizedShift = Math.floorMod(shift, input.length);
        for (int outputIndex = 0; outputIndex < input.length; outputIndex++) {
            output[outputIndex] =
                    input[Math.floorMod(outputIndex - normalizedShift,
                            input.length)];
        }
        return output;
    }

    private static float[] rollRankThreeOracle(
            float[] input,
            int first,
            int second,
            int third,
            float[] shifts,
            float[] axes) {
        assertEquals(first * second * third, input.length,
                "Roll oracle input length");
        assertEquals(shifts.length, axes.length,
                "Roll oracle shift/axis count");
        int[] dimensions = {first, second, third};
        float[] output = new float[input.length];
        for (int i = 0; i < first; i++) {
            for (int j = 0; j < second; j++) {
                for (int k = 0; k < third; k++) {
                    int[] source = {i, j, k};
                    for (int control = 0; control < shifts.length; control++) {
                        int axis = (int) axes[control];
                        if (axis < 0) axis += dimensions.length;
                        int dimension = dimensions[axis];
                        source[axis] = Math.floorMod(
                                source[axis] -
                                        Math.floorMod((int) shifts[control],
                                                dimension),
                                dimension);
                    }
                    int outputIndex = (i * second + j) * third + k;
                    int inputIndex =
                            (source[0] * second + source[1]) * third +
                                    source[2];
                    output[outputIndex] = input[inputIndex];
                }
            }
        }
        return output;
    }

    private static float[] sliceOracle(
            float[] input,
            int inputRows,
            int inputColumns,
            int beginRow,
            int beginColumn,
            int outputRows,
            int outputColumns) {
        assertEquals(inputRows * inputColumns, input.length,
                "Slice oracle input length");
        float[] output = new float[outputRows * outputColumns];
        for (int row = 0; row < outputRows; row++) {
            for (int column = 0; column < outputColumns; column++) {
                output[row * outputColumns + column] =
                        input[(beginRow + row) * inputColumns
                                + beginColumn + column];
            }
        }
        return output;
    }

    private static float[] stridedSliceOracle(
            float[] input, int batches, int rows, int columns) {
        assertEquals(batches * rows * columns, input.length,
                "Strided-slice oracle input length");
        float[] output = new float[batches * 2 * 2];
        for (int batch = 0; batch < batches; batch++) {
            for (int outputRow = 0; outputRow < 2; outputRow++) {
                for (int outputColumn = 0; outputColumn < 2; outputColumn++) {
                    int inputRow = 1 + outputRow * 2;
                    int inputColumn = outputColumn * 2;
                    output[(batch * 2 + outputRow) * 2 + outputColumn] =
                            input[(batch * rows + inputRow) * columns
                                    + inputColumn];
                }
            }
        }
        return output;
    }

    private static float[] stackAxisOneOracle(
            float[][] inputs, int rows, int columns) {
        float[] output = new float[rows * inputs.length * columns];
        for (int row = 0; row < rows; row++) {
            for (int input = 0; input < inputs.length; input++) {
                for (int column = 0; column < columns; column++) {
                    output[(row * inputs.length + input) * columns + column] =
                            inputs[input][row * columns + column];
                }
            }
        }
        return output;
    }

    private static float[] windowPartitionOracle(
            float[] input,
            int batch,
            int height,
            int width,
            int channels,
            int window) {
        int heightBlocks = height / window;
        int widthBlocks = width / window;
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int hb = 0; hb < heightBlocks; hb++) {
                for (int wb = 0; wb < widthBlocks; wb++) {
                    int flatWindow = (b * heightBlocks + hb) * widthBlocks + wb;
                    for (int localHeight = 0; localHeight < window; localHeight++) {
                        for (int localWidth = 0; localWidth < window; localWidth++) {
                            for (int channel = 0; channel < channels; channel++) {
                                int inputIndex =
                                        (((b * height + hb * window + localHeight)
                                                * width + wb * window + localWidth)
                                                * channels + channel);
                                int outputIndex =
                                        (((flatWindow * window + localHeight)
                                                * window + localWidth)
                                                * channels + channel);
                                output[outputIndex] = input[inputIndex];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    private static float[] windowUnpartitionOracle(
            float[] input,
            int batch,
            int height,
            int width,
            int channels,
            int window) {
        int heightBlocks = height / window;
        int widthBlocks = width / window;
        float[] output = new float[input.length];
        for (int b = 0; b < batch; b++) {
            for (int row = 0; row < height; row++) {
                for (int column = 0; column < width; column++) {
                    int flatWindow =
                            (b * heightBlocks + row / window) * widthBlocks
                                    + column / window;
                    for (int channel = 0; channel < channels; channel++) {
                        int inputIndex =
                                (((flatWindow * window + row % window)
                                        * window + column % window)
                                        * channels + channel);
                        int outputIndex =
                                (((b * height + row) * width + column)
                                        * channels + channel);
                        output[outputIndex] = input[inputIndex];
                    }
                }
            }
        }
        return output;
    }

    private static float[] rmsNormOracle(
            float[] input,
            float[] gamma,
            int rows,
            int hidden,
            double epsilon) {
        float[] output = new float[rows * hidden];
        for (int row = 0; row < rows; row++) {
            double sumSquares = 0.0;
            int rowOffset = row * hidden;
            for (int column = 0; column < hidden; column++) {
                double value = input[rowOffset + column];
                sumSquares += value * value;
            }
            double inverseRootMeanSquare =
                    1.0 / Math.sqrt(sumSquares / hidden + epsilon);
            for (int column = 0; column < hidden; column++) {
                output[rowOffset + column] = (float) (
                        input[rowOffset + column]
                                * inverseRootMeanSquare
                                * gamma[column]);
            }
        }
        return output;
    }

    private static float[] skipRmsNormOracle(
            float[] input,
            float[] skip,
            float[] gamma,
            int rows,
            int hidden,
            double epsilon) {
        float[] residual = new float[rows * hidden];
        for (int index = 0; index < residual.length; index++) {
            residual[index] = input[index] + skip[index];
        }
        return rmsNormOracle(residual, gamma, rows, hidden, epsilon);
    }

    private static float[] rmsNormLinearOracle(
            float[] input,
            float[] gamma,
            float[] weight,
            int rows,
            int hidden,
            int projected,
            double epsilon) {
        float[] normalized = rmsNormOracle(input, gamma, rows, hidden, epsilon);
        return matrixProjectionOracle(normalized, weight, rows, hidden, projected);
    }

    private static float[] matrixProjectionOracle(
            float[] input,
            float[] weight,
            int rows,
            int hidden,
            int projected) {
        float[] output = new float[rows * projected];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < projected; column++) {
                double sum = 0.0;
                for (int inner = 0; inner < hidden; inner++) {
                    sum += input[row * hidden + inner]
                            * weight[inner * projected + column];
                }
                output[row * projected + column] = (float) sum;
            }
        }
        return output;
    }

    private static float[] fusedGemmSwiGluOracle(
            float[] input,
            float[] gateWeight,
            float[] upWeight,
            int rows,
            int hidden,
            int projected) {
        float[] output = new float[rows * projected];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < projected; column++) {
                double gate = 0.0;
                double up = 0.0;
                for (int inner = 0; inner < hidden; inner++) {
                    double value = input[row * hidden + inner];
                    gate += value * gateWeight[inner * projected + column];
                    up += value * upWeight[inner * projected + column];
                }
                double swish = gate / (1.0 + Math.exp(-gate));
                output[row * projected + column] = (float) (swish * up);
            }
        }
        return output;
    }

    private static float[] fusedRmsNormSwiGluOracle(
            float[] input,
            float[] gamma,
            float[] gateWeight,
            float[] upWeight,
            int rows,
            int hidden,
            int projected,
            double epsilon) {
        float[] normalized = rmsNormOracle(input, gamma, rows, hidden, epsilon);
        return fusedGemmSwiGluOracle(
                normalized, gateWeight, upWeight, rows, hidden, projected);
    }

    private static float[] rotaryCacheValues(
            int step, int length, boolean cosine) {
        float[] values = new float[length];
        for (int index = 0; index < length; index++) {
            double angle = 0.09 * (step + index + 1);
            values[index] = (float) (cosine ? Math.cos(angle) : Math.sin(angle));
        }
        return values;
    }

    private static float[] cachedRopeOracle(
            float[] input,
            float[] cosine,
            float[] sine,
            int batch,
            int sequence,
            int heads,
            int headDimension) {
        int pairs = headDimension / 2;
        float[] output = new float[input.length];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int sequenceIndex = 0;
                    sequenceIndex < sequence;
                    sequenceIndex++) {
                for (int head = 0; head < heads; head++) {
                    int vectorOffset =
                            ((batchIndex * sequence + sequenceIndex) * heads + head)
                                    * headDimension;
                    int cacheOffset = sequenceIndex * pairs;
                    for (int pair = 0; pair < pairs; pair++) {
                        int valueOffset = vectorOffset + pair * 2;
                        double even = input[valueOffset];
                        double odd = input[valueOffset + 1];
                        double cos = cosine[cacheOffset + pair];
                        double sin = sine[cacheOffset + pair];
                        output[valueOffset] = (float) (even * cos - odd * sin);
                        output[valueOffset + 1] = (float) (even * sin + odd * cos);
                    }
                }
            }
        }
        return output;
    }

    private static float[] dotProductAttentionOracle(
            float[] query,
            float[] key,
            float[] value,
            int batch,
            int queryFeatures,
            int querySteps,
            int keySteps,
            int valueFeatures,
            boolean normalize) {
        float[] output = new float[batch * valueFeatures * querySteps];
        double[] scores = new double[keySteps];
        double scale = normalize ? 1.0 / Math.sqrt(queryFeatures) : 1.0;
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int queryStep = 0; queryStep < querySteps; queryStep++) {
                double maximum = Double.NEGATIVE_INFINITY;
                for (int keyStep = 0; keyStep < keySteps; keyStep++) {
                    double score = 0.0;
                    for (int feature = 0; feature < queryFeatures; feature++) {
                        int queryOffset =
                                (batchIndex * queryFeatures + feature) * querySteps
                                        + queryStep;
                        int keyOffset =
                                (batchIndex * queryFeatures + feature) * keySteps
                                        + keyStep;
                        score += query[queryOffset] * key[keyOffset];
                    }
                    scores[keyStep] = score * scale;
                    maximum = Math.max(maximum, scores[keyStep]);
                }

                double denominator = 0.0;
                for (int keyStep = 0; keyStep < keySteps; keyStep++) {
                    scores[keyStep] = Math.exp(scores[keyStep] - maximum);
                    denominator += scores[keyStep];
                }

                for (int valueFeature = 0;
                        valueFeature < valueFeatures;
                        valueFeature++) {
                    double sum = 0.0;
                    for (int keyStep = 0; keyStep < keySteps; keyStep++) {
                        int valueOffset =
                                (batchIndex * valueFeatures + valueFeature) * keySteps
                                        + keyStep;
                        sum += scores[keyStep] / denominator * value[valueOffset];
                    }
                    int outputOffset =
                            (batchIndex * valueFeatures + valueFeature) * querySteps
                                    + queryStep;
                    output[outputOffset] = (float) sum;
                }
            }
        }
        return output;
    }

    private static float[] layerNormOracle(
            float[] input,
            float[] gain,
            float[] bias,
            int rows,
            int width,
            double epsilon) {
        float[] output = new float[rows * width];
        for (int row = 0; row < rows; row++) {
            int rowOffset = row * width;
            double mean = 0.0;
            for (int column = 0; column < width; column++) {
                mean += input[rowOffset + column];
            }
            mean /= width;

            double variance = 0.0;
            for (int column = 0; column < width; column++) {
                double centered = input[rowOffset + column] - mean;
                variance += centered * centered;
            }
            variance /= width;
            double inverseStandardDeviation = 1.0 / Math.sqrt(variance + epsilon);
            for (int column = 0; column < width; column++) {
                output[rowOffset + column] = (float) (
                        (input[rowOffset + column] - mean)
                                        * inverseStandardDeviation
                                        * gain[column]
                                + bias[column]);
            }
        }
        return output;
    }

    private static float[] gatherAxisZeroOracle(
            float[] input,
            int[] indices,
            int inputRows,
            int width) {
        float[] output = new float[indices.length * width];
        for (int outputRow = 0; outputRow < indices.length; outputRow++) {
            int inputRow = indices[outputRow];
            if (inputRow < 0) {
                inputRow += inputRows;
            }
            System.arraycopy(
                    input,
                    inputRow * width,
                    output,
                    outputRow * width,
                    width);
        }
        return output;
    }

    private static float[] concatAxisOneOracle(
            float[] left,
            float[] right,
            int rows,
            int leftWidth,
            int rightWidth) {
        int outputWidth = leftWidth + rightWidth;
        float[] output = new float[rows * outputWidth];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(
                    left,
                    row * leftWidth,
                    output,
                    row * outputWidth,
                    leftWidth);
            System.arraycopy(
                    right,
                    row * rightWidth,
                    output,
                    row * outputWidth + leftWidth,
                    rightWidth);
        }
        return output;
    }

    private static float[] permuteOracle(
            float[] input, long[] inputShape, int[] permutation) {
        int rank = inputShape.length;
        long[] outputShape = new long[rank];
        for (int outputAxis = 0; outputAxis < rank; outputAxis++) {
            outputShape[outputAxis] = inputShape[permutation[outputAxis]];
        }

        float[] output = new float[input.length];
        long[] outputCoordinates = new long[rank];
        long[] inputCoordinates = new long[rank];
        for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
            long remainder = outputIndex;
            for (int axis = rank - 1; axis >= 0; axis--) {
                outputCoordinates[axis] = remainder % outputShape[axis];
                remainder /= outputShape[axis];
            }
            for (int outputAxis = 0; outputAxis < rank; outputAxis++) {
                inputCoordinates[permutation[outputAxis]] =
                        outputCoordinates[outputAxis];
            }

            long inputIndex = 0;
            for (int axis = 0; axis < rank; axis++) {
                inputIndex = inputIndex * inputShape[axis] + inputCoordinates[axis];
            }
            output[outputIndex] = input[(int) inputIndex];
        }
        return output;
    }

    @FunctionalInterface
    private interface GraphBuilder {
        SDVariable build(SameDiff sameDiff, Map<String, SDVariable> variables);
    }

    @FunctionalInterface
    private interface MultiOutputGraphBuilder {
        SDVariable[] build(
                SameDiff sameDiff, Map<String, SDVariable> variables);
    }

    private static final class OutputSpec {
        private final DataType dataType;
        private final long[] shape;
        private final IntFunction<float[]> oracle;
        private final float absoluteTolerance;
        private final float relativeTolerance;

        private OutputSpec(
                DataType dataType,
                long[] shape,
                IntFunction<float[]> oracle,
                float absoluteTolerance,
                float relativeTolerance) {
            this.dataType = dataType;
            this.shape = shape;
            this.oracle = oracle;
            this.absoluteTolerance = absoluteTolerance;
            this.relativeTolerance = relativeTolerance;
        }
    }

    private static final class UnaryForwardSpec {
        private final String opName;
        private final double[] tArguments;
        private final DoubleUnaryOperator function;

        private UnaryForwardSpec(
                String opName,
                double[] tArguments,
                DoubleUnaryOperator function) {
            this.opName = opName;
            this.tArguments = tArguments;
            this.function = function;
        }
    }

    private static final class BinaryForwardSpec {
        private final String opName;
        private final DoubleBinaryOperator function;

        private BinaryForwardSpec(
                String opName, DoubleBinaryOperator function) {
            this.opName = opName;
            this.function = function;
        }
    }

    private static final class ActivationBackwardSpec {
        private final String opName;
        private final double[] tArguments;
        private final DoubleUnaryOperator derivative;

        private ActivationBackwardSpec(
                String opName,
                double[] tArguments,
                DoubleUnaryOperator derivative) {
            this.opName = opName;
            this.tArguments = tArguments;
            this.derivative = derivative;
        }
    }

    private static final class InputSpec {
        private final String name;
        private final DataType dataType;
        private final long[] shape;
        private final IntFunction<float[]> values;
        private final IntFunction<INDArray> arrays;

        private InputSpec(
                String name, long[] shape, IntFunction<float[]> values) {
            this(name, DataType.FLOAT, shape, values);
        }

        private InputSpec(
                String name,
                DataType dataType,
                long[] shape,
                IntFunction<float[]> values) {
            this.name = name;
            this.dataType = dataType;
            this.shape = shape;
            this.values = values;
            this.arrays = null;
        }

        private InputSpec(
                String name,
                DataType dataType,
                long[] shape,
                IntFunction<INDArray> arrays,
                boolean preserveLayout) {
            assertTrue(preserveLayout,
                    "Layout-preserving InputSpec constructor requires preserveLayout=true");
            this.name = name;
            this.dataType = dataType;
            this.shape = shape;
            this.values = null;
            this.arrays = arrays;
        }

        private INDArray arrayAt(int step) {
            INDArray result = arrays == null
                    ? Nd4j.create(values.apply(step), shape, dataType)
                    : arrays.apply(step);
            assertEquals(dataType, result.dataType(),
                    name + ": generated input dtype");
            assertArrayEquals(shape, result.shape(),
                    name + ": generated input shape");
            return result;
        }
    }

    /**
     * Attach a descriptor-backed test facade only after its dynamic operation
     * name has been initialized. This mirrors {@link SameDiff#dynamic} and avoids
     * asking the superclass constructor to resolve a virtual opName too early.
     */
    private static void attachNamedOp(
            DynamicCustomOp op, SameDiff sameDiff, SDVariable[] inputs) {
        op.setSameDiff(sameDiff);
        op.setInstanceId();
        sameDiff.addArgsFor(inputs, op);
    }

    /**
     * Test-only graph wrapper for registered native ops that do not yet have a
     * generated Java facade. Shape inference and execution still come from the
     * canonical registered descriptor identified by {@code opName}.
     */
    private static final class NamedDynamicOp extends DynamicCustomOp {
        private final DataType outputDataType;

        private NamedDynamicOp(
                String opName, SameDiff sameDiff, SDVariable... inputs) {
            this(opName, null, sameDiff, inputs);
        }

        private NamedDynamicOp(
                String opName,
                DataType outputDataType,
                SameDiff sameDiff,
                SDVariable... inputs) {
            super(opName);
            this.outputDataType = outputDataType;
            attachNamedOp(this, sameDiff, inputs);
        }

        @Override
        public List<DataType> calculateOutputDataTypes(
                List<DataType> inputDataTypes) {
            return Collections.singletonList(
                    outputDataType == null
                            ? inputDataTypes.get(0)
                            : outputDataType);
        }

        @Override
        public int getNumOutputs() {
            return 1;
        }
    }

    /** Test facade for canonical registered ops with a frozen output count. */
    private static final class NamedFixedOutputOp extends DynamicCustomOp {
        private final int outputCount;

        private NamedFixedOutputOp(
                String opName,
                int outputCount,
                SameDiff sameDiff,
                SDVariable... inputs) {
            super(opName);
            assertTrue(outputCount > 0,
                    "NamedFixedOutputOp requires at least one output");
            this.outputCount = outputCount;
            attachNamedOp(this, sameDiff, inputs);
        }

        @Override
        public List<DataType> calculateOutputDataTypes(
                List<DataType> inputDataTypes) {
            return Collections.nCopies(outputCount, inputDataTypes.get(0));
        }

        @Override
        public int getNumOutputs() {
            return outputCount;
        }
    }
}
