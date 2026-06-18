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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.api.buffer.DataType.*;

/**
 * DSP CPU operations test -- exercises op categories and edge cases under
 * all CPU-compatible execution modes (SLOT_BY_SLOT, AUTO).
 *
 * Focuses on scenarios not already covered by DspOptimizedSlotBySlotTest:
 * - Reduction ops (sum, max, min, norm) across ranks and axes
 * - Gather / dynamic indexing
 * - Mixed dtype operations
 * - Broadcast edge cases
 * - Dynamic shape changes across calls
 * - Conditional / boolean mask operations
 * - Complex op chains (layer norm, attention, residual)
 * - Plan lifecycle (auto-seal, stale data detection)
 * - Concat / split ops
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP CPU Operations")
public class DspCpuOperationsTest {

    private boolean dspWasEnabled;

    @BeforeEach
    public void setUp() {
        dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(true);
        Nd4j.getExecutioner().commit();
    }

    @AfterEach
    public void tearDown() {
        InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled);
        Nd4j.getExecutioner().commit();
    }

    // ---- Helpers ----

    private static void configureDsp(SameDiff sd, GraphExecutionMode mode) {
        sd.setGraphExecutionMode(mode);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    private static Map<String, INDArray> reference(SameDiff sd, Map<String, INDArray> inputs,
                                                   String... outputs) {
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> ref = sd.output(inputs, outputs);
        for (Map.Entry<String, INDArray> e : ref.entrySet()) {
            ref.put(e.getKey(), e.getValue().dup());
        }
        return ref;
    }

    private static void assertClose(INDArray expected, INDArray actual, double rtol, double atol,
                                    String msg) {
        assertEquals(expected.shape().length, actual.shape().length,
                msg + " rank mismatch");
        assertArrayEquals(expected.shape(), actual.shape(), msg + " shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        double scale = expected.amaxNumber().doubleValue();
        double relErr = (scale > 1e-8) ? maxDiff / scale : maxDiff;
        assertTrue(maxDiff <= atol || relErr <= rtol,
                msg + ": maxDiff=" + maxDiff + " relErr=" + relErr
                        + " (atol=" + atol + " rtol=" + rtol + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REDUCTION OPS
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "reductionAcrossAxes[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Reduction ops (sum, max, min, mean) across various axes match reference")
    public void testReductionAcrossAxes(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8, 4);
        x.sum("sum_all", 1, 2);
        sd.math.reduceMax("max_axis1", x, false, 1);
        sd.math.reduceMin("min_axis2", x, false, 2);
        x.mean("mean_keepdim", true, 1);

        INDArray input = Nd4j.randn(FLOAT, 3, 8, 4);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs,
                "sum_all", "max_axis1", "min_axis2", "mean_keepdim");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs,
                "sum_all", "max_axis1", "min_axis2", "mean_keepdim");

        for (String key : ref.keySet()) {
            assertClose(ref.get(key), result.get(key), 1e-5, 1e-5, key + "/" + mode);
        }
        sd.close();
    }

    @ParameterizedTest(name = "normReductions[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Norm2 reduction matches reference under DSP")
    public void testNormReductions(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 16);
        sd.math.norm2("norm2", x, false, 1);

        INDArray input = Nd4j.randn(FLOAT, 5, 16);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "norm2");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "norm2");

        assertClose(ref.get("norm2"), result.get("norm2"), 1e-5, 1e-5,
                "norm2/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GATHER / DYNAMIC INDEXING
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "gatherOp[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Gather op with dynamic indices matches reference")
    public void testGatherOp(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable embeddings = sd.placeHolder("embeddings", FLOAT, 10, 8);
        SDVariable indices = sd.placeHolder("indices", INT, -1);
        sd.gather("gathered", embeddings, indices, 0);

        INDArray embData = Nd4j.randn(FLOAT, 10, 8);
        INDArray idxData = Nd4j.createFromArray(0, 3, 7, 1, 5);
        Map<String, INDArray> inputs = Map.of("embeddings", embData, "indices", idxData);

        Map<String, INDArray> ref = reference(sd, inputs, "gathered");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "gathered");

        assertClose(ref.get("gathered"), result.get("gathered"), 1e-6, 1e-6,
                "gather/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MIXED DTYPE
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "mixedDtype[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Mixed FLOAT + DOUBLE graph produces correct results")
    public void testMixedDtype(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        SDVariable scale = sd.constant("scale", Nd4j.ones(DOUBLE, 4).mul(2.5));
        SDVariable casted = sd.castTo("x_double", x, DOUBLE);
        SDVariable scaled = casted.mul("scaled", scale);
        sd.castTo("out", scaled, FLOAT);

        INDArray input = Nd4j.randn(FLOAT, 3, 4);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-5, 1e-5, "mixedDtype/" + mode);
        sd.close();
    }

    @ParameterizedTest(name = "intArithmetic[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Integer arithmetic ops produce correct results under DSP")
    public void testIntegerArithmetic(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", INT, -1, 4);
        SDVariable b = sd.placeHolder("b", INT, -1, 4);
        a.add("sum", b);
        a.sub("diff", b);

        INDArray aArr = Nd4j.createFromArray(new int[][]{{1, 2, 3, 4}, {5, 6, 7, 8}});
        INDArray bArr = Nd4j.createFromArray(new int[][]{{10, 20, 30, 40}, {50, 60, 70, 80}});
        Map<String, INDArray> inputs = Map.of("a", aArr, "b", bArr);

        Map<String, INDArray> ref = reference(sd, inputs, "sum", "diff");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "sum", "diff");

        assertEquals(ref.get("sum"), result.get("sum"), "intArith sum/" + mode);
        assertEquals(ref.get("diff"), result.get("diff"), "intArith diff/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // BROADCAST EDGE CASES
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "broadcastAdd[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Broadcast add: [batch,N] + [1,N] matches reference")
    public void testBroadcastAdd(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable bias = sd.var("bias", Nd4j.linspace(0, 7, 8, FLOAT).reshape(1, 8));
        x.add("out", bias);

        INDArray input = Nd4j.randn(FLOAT, 5, 8);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-6, 1e-6, "broadcast/" + mode);
        sd.close();
    }

    @ParameterizedTest(name = "broadcast3D[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Broadcast across 3D: [B,S,D] * [1,1,D] matches reference")
    public void testBroadcast3D(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4, 8);
        SDVariable scale = sd.constant("scale", Nd4j.randn(FLOAT, 1, 1, 8));
        x.mul("out", scale);

        INDArray input = Nd4j.randn(FLOAT, 2, 4, 8);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-6, 1e-6, "broadcast3d/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DYNAMIC SHAPE CHANGES
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "dynamicBatchSize[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("DSP handles varying batch sizes correctly across calls")
    public void testDynamicBatchSize(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        SDVariable w = sd.var("w", new XavierInitScheme('c', 4, 2), FLOAT, 4, 2);
        sd.nn.relu("out", x.mmul("z", w), 0);

        configureDsp(sd, mode);

        int[] batchSizes = {1, 4, 2, 8, 1, 16};
        for (int bs : batchSizes) {
            INDArray input = Nd4j.randn(FLOAT, bs, 4);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "out");
            INDArray out = result.get("out");
            assertEquals(bs, out.shape()[0],
                    "batch size mismatch for bs=" + bs + " mode=" + mode);
            assertEquals(2, out.shape()[1],
                    "output dim mismatch for bs=" + bs + " mode=" + mode);
            assertFalse(Double.isNaN(out.sumNumber().doubleValue()),
                    "NaN output for bs=" + bs + " mode=" + mode);
            assertFalse(Double.isInfinite(out.sumNumber().doubleValue()),
                    "Inf output for bs=" + bs + " mode=" + mode);
        }
        sd.close();
    }

    @Test
    @DisplayName("DSP plan recompiles correctly when sequence length changes")
    public void testSequenceLengthChange() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, -1, 8);
        SDVariable mean = x.mean("mean", true, 2);
        x.sub("centered", mean);

        configureDsp(sd, GraphExecutionMode.AUTO);

        int[][] shapes = {{2, 4, 8}, {2, 16, 8}, {2, 1, 8}, {2, 32, 8}};
        for (int[] shape : shapes) {
            INDArray input = Nd4j.randn(FLOAT, shape);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "centered");
            INDArray out = result.get("centered");
            assertArrayEquals(shape, toIntArray(out.shape()),
                    "shape mismatch for input shape " + java.util.Arrays.toString(shape));
            double meanVal = out.meanNumber().doubleValue();
            assertTrue(Math.abs(meanVal) < 0.1,
                    "mean of centered should be ~0, got " + meanVal);
        }
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONDITIONAL / BOOLEAN MASK OPS
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "booleanMask[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Comparison + masking chain produces correct results")
    public void testBooleanMaskChain(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable zeros = sd.constant("zeros", Nd4j.zeros(FLOAT, 1, 8));
        // Manual ReLU: x * cast(x > 0, FLOAT)
        SDVariable mask = sd.gt("gt_zero", x, zeros).castTo("mask_f", FLOAT);
        x.mul("out", mask);

        INDArray input = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-6, 1e-6, "boolMask/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX OP CHAINS
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "layerNorm[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Manual layer norm (mean/var/normalize) matches reference")
    public void testLayerNorm(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 16);
        SDVariable gamma = sd.var("gamma", Nd4j.ones(FLOAT, 16));
        SDVariable beta = sd.var("beta", Nd4j.zeros(FLOAT, 16));

        SDVariable mean = x.mean("mean", true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable variance = centered.mul(centered).mean("var", true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(FLOAT, 1e-5f));
        SDVariable std = sd.math.sqrt("std", variance.add(eps));
        SDVariable normed = centered.div("normed", std);
        normed.mul("scaled", gamma).add("out", beta);

        INDArray input = Nd4j.randn(FLOAT, 4, 16).muli(3.0).addi(1.0);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-4, 1e-4, "layerNorm/" + mode);
        sd.close();
    }

    @ParameterizedTest(name = "residualBlock[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Residual block (linear + relu + skip connection) matches reference")
    public void testResidualBlock(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 16);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 16, 16), FLOAT, 16, 16);
        SDVariable b1 = sd.var("b1", new ZeroInitScheme('c'), FLOAT, 16);
        SDVariable w2 = sd.var("w2", new XavierInitScheme('c', 16, 16), FLOAT, 16, 16);
        SDVariable b2 = sd.var("b2", new ZeroInitScheme('c'), FLOAT, 16);

        SDVariable h = sd.nn.relu("relu1", sd.nn.linear("z1", x, w1, b1), 0);
        SDVariable z2 = sd.nn.linear("z2", h, w2, b2);
        x.add("out", z2);

        INDArray input = Nd4j.randn(FLOAT, 4, 16);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-5, 1e-5,
                "residualBlock/" + mode);
        sd.close();
    }

    @ParameterizedTest(name = "multiLayerMlp[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Multi-layer MLP (3 layers) matches reference under DSP")
    public void testMultiLayerMlp(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', 8, 16), FLOAT, 8, 16);
        SDVariable b0 = sd.var("b0", new ZeroInitScheme('c'), FLOAT, 16);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 16, 16), FLOAT, 16, 16);
        SDVariable b1 = sd.var("b1", new ZeroInitScheme('c'), FLOAT, 16);
        SDVariable w2 = sd.var("w2", new XavierInitScheme('c', 16, 4), FLOAT, 16, 4);
        SDVariable b2 = sd.var("b2", new ZeroInitScheme('c'), FLOAT, 4);

        SDVariable h0 = sd.nn.relu("h0", sd.nn.linear("z0", x, w0, b0), 0);
        SDVariable h1 = sd.nn.tanh("h1", sd.nn.linear("z1", h0, w1, b1));
        sd.nn.softmax("out", sd.nn.linear("z2", h1, w2, b2), 1);

        INDArray input = Nd4j.randn(FLOAT, 4, 8).muli(0.5);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-4, 1e-4,
                "multiLayerMlp/" + mode);
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // REPEATED EXECUTION / STALE DATA CHECK
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "repeatedExecDifferentInput[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Repeated execution with different inputs must produce different outputs")
    public void testRepeatedExecDifferentInput(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w = sd.var("w", new XavierInitScheme('c', 8, 4), FLOAT, 8, 4);
        SDVariable b = sd.var("b", new ZeroInitScheme('c'), FLOAT, 4);
        sd.nn.sigmoid("out", sd.nn.linear("z", x, w, b));

        configureDsp(sd, mode);

        INDArray prevOut = null;
        for (int i = 0; i < 10; i++) {
            INDArray input = Nd4j.randn(FLOAT, 4, 8).muli(i + 1);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "out");
            INDArray out = result.get("out").dup();

            if (prevOut != null) {
                double diff = out.sub(prevOut).amaxNumber().doubleValue();
                assertTrue(diff > 1e-6,
                        "iter " + i + ": output unchanged (stale replay), diff=" + diff);
            }
            prevOut = out;
        }
        sd.close();
    }

    @Test
    @DisplayName("Plan survives output().close() cycle without corrupting next call")
    public void testOutputCloseDoesNotCorruptPlan() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.ones(FLOAT, 4, 2));
        x.mmul("z", w).add("out", sd.constant("b", Nd4j.ones(FLOAT, 2)));

        configureDsp(sd, GraphExecutionMode.AUTO);

        INDArray firstInput = Nd4j.ones(FLOAT, 2, 4);
        Map<String, INDArray> firstResult = sd.output(Map.of("x", firstInput), "out");
        INDArray expected = firstResult.get("out").dup();
        for (INDArray arr : firstResult.values()) {
            if (arr.closeable() && !arr.wasClosed()) arr.close();
        }

        Map<String, INDArray> secondResult = sd.output(Map.of("x", firstInput), "out");
        assertClose(expected, secondResult.get("out"), 1e-6, 1e-6,
                "output after close");
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONCAT / SPLIT OPS
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "concatOp[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Concat along axis matches reference under DSP")
    public void testConcatOp(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", FLOAT, -1, 4);
        SDVariable b = sd.placeHolder("b", FLOAT, -1, 6);
        sd.concat("out", 1, a, b);

        INDArray aArr = Nd4j.randn(FLOAT, 3, 4);
        INDArray bArr = Nd4j.randn(FLOAT, 3, 6);
        Map<String, INDArray> inputs = Map.of("a", aArr, "b", bArr);

        Map<String, INDArray> ref = reference(sd, inputs, "out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out");

        assertClose(ref.get("out"), result.get("out"), 1e-6, 1e-6, "concat/" + mode);
        assertEquals(10, result.get("out").shape()[1], "concat dim should be 4+6=10");
        sd.close();
    }

    @ParameterizedTest(name = "sliceOp[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Slice (stridedSlice) produces correct chunks under DSP")
    public void testSliceOp(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, 3, 12);
        // Manual split into 3 chunks of 4 via stridedSlice
        SDVariable chunk0 = sd.stridedSlice("chunk0", x,
                new long[]{0, 0}, new long[]{3, 4}, new long[]{1, 1});
        SDVariable chunk1 = sd.stridedSlice("chunk1", x,
                new long[]{0, 4}, new long[]{3, 8}, new long[]{1, 1});
        SDVariable chunk2 = sd.stridedSlice("chunk2", x,
                new long[]{0, 8}, new long[]{3, 12}, new long[]{1, 1});
        chunk0.add("out0", sd.constant("one", Nd4j.scalar(FLOAT, 1.0f)));
        chunk1.mul("out1", sd.constant("two", Nd4j.scalar(FLOAT, 2.0f)));
        chunk2.sub("out2", sd.constant("half", Nd4j.scalar(FLOAT, 0.5f)));

        INDArray input = Nd4j.randn(FLOAT, 3, 12);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs, "out0", "out1", "out2");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs, "out0", "out1", "out2");

        for (String key : new String[]{"out0", "out1", "out2"}) {
            assertClose(ref.get(key), result.get(key), 1e-6, 1e-6, key + "/" + mode);
            assertEquals(4, result.get(key).shape()[1],
                    key + " should have dim=4");
        }
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ACTIVATION FUNCTION DIVERSITY
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "activationChain[{0}]")
    @EnumSource(value = GraphExecutionMode.class, names = {"SLOT_BY_SLOT", "AUTO"})
    @DisplayName("Chain of activation functions (tanh, sigmoid, gelu) matches reference")
    public void testActivationChain(GraphExecutionMode mode) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w = sd.var("w", new XavierInitScheme('c', 8, 8), FLOAT, 8, 8);
        SDVariable z = x.mmul("z", w);
        sd.nn.tanh("tanh_out", z);
        sd.nn.sigmoid("sigmoid_out", z);
        sd.nn.gelu("gelu_out", z);

        INDArray input = Nd4j.randn(FLOAT, 4, 8).muli(0.5);
        Map<String, INDArray> inputs = Map.of("x", input);

        Map<String, INDArray> ref = reference(sd, inputs,
                "tanh_out", "sigmoid_out", "gelu_out");

        configureDsp(sd, mode);
        Map<String, INDArray> result = sd.output(inputs,
                "tanh_out", "sigmoid_out", "gelu_out");

        for (String key : ref.keySet()) {
            assertClose(ref.get(key), result.get(key), 1e-5, 1e-5, key + "/" + mode);
        }
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PLAN LIFECYCLE ON CPU
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("AUTO mode advances plan past SLOT_BY_SLOT on CPU (auto-seal works)")
    public void testAutoModeAdvancesPlanPhase() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 4);
        SDVariable w = sd.var("w", Nd4j.randn(FLOAT, 4, 2));
        sd.nn.relu("out", x.mmul("z", w), 0);

        configureDsp(sd, GraphExecutionMode.AUTO);

        for (int i = 0; i < 5; i++) {
            INDArray input = Nd4j.randn(FLOAT, 2, 4);
            sd.output(Map.of("x", input), "out");
        }

        assertNotNull(sd.dsp(), "DSP handle should exist after executions");
        assertTrue(sd.dsp().isCompiled(), "Plan should be compiled");
        sd.close();
    }

    @Test
    @DisplayName("AUTO mode produces identical results to SLOT_BY_SLOT across many iterations")
    public void testAutoMatchesSlotBySlot() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", FLOAT, -1, 8);
        SDVariable w = sd.var("w", new XavierInitScheme('c', 8, 4), FLOAT, 8, 4);
        SDVariable b = sd.var("b", new ZeroInitScheme('c'), FLOAT, 4);
        SDVariable z = sd.nn.linear("z", x, w, b);
        sd.nn.softmax("out", z, 1);

        INDArray input = Nd4j.randn(FLOAT, 4, 8);
        Map<String, INDArray> inputs = Map.of("x", input);

        // Get SLOT_BY_SLOT reference
        configureDsp(sd, GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> slotRef = sd.output(inputs, "out");
        INDArray refOut = slotRef.get("out").dup();

        // Run under AUTO
        configureDsp(sd, GraphExecutionMode.AUTO);
        for (int i = 0; i < 8; i++) {
            Map<String, INDArray> result = sd.output(inputs, "out");
            assertClose(refOut, result.get("out"), 1e-5, 1e-5,
                    "AUTO iter=" + i);
        }
        sd.close();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════════════════════════════

    private static int[] toIntArray(long[] longs) {
        int[] result = new int[longs.length];
        for (int i = 0; i < longs.length; i++) result[i] = (int) longs[i];
        return result;
    }
}
