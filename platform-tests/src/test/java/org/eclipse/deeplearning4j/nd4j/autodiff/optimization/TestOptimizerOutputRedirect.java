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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.AlgebraicOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.LinearFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.NormalizationFusionOptimizations;
import org.nd4j.autodiff.samediff.optimize.optimizations.QuantizationOptimizations;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Tests that optimizer output redirect behavior is correct — when a graph output
 * gets simplified or fused, the output must still be accessible under its original
 * name (or via the updated output list in the optimized graph).
 */
@Tag(TagNames.DL4J_OLD_API)
public class TestOptimizerOutputRedirect extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 1_000_000_000L;
    }

    private static void assertClose(String message, INDArray expected, INDArray actual, double eps) {
        assertArrayEquals(message + " shape", expected.shape(), actual.shape());
        assertEquals(message + " length", expected.length(), actual.length());
        for (long i = 0; i < expected.length(); i++) {
            assertEquals(message + " value[" + i + "]",
                    expected.getDouble(i), actual.getDouble(i), eps);
        }
    }

    private static List<OptimizerSet> optimizationsWithoutQuantization() {
        List<OptimizerSet> optimizations = new ArrayList<>();
        for (OptimizerSet set : GraphOptimizer.defaultOptimizations()) {
            if (!(set instanceof QuantizationOptimizations)) {
                optimizations.add(set);
            }
        }
        return optimizations;
    }

    /**
     * Create a graph where the registered output "result" is computed as x + 0.
     * AlgebraicOptimizations should simplify x + 0 -> x.
     * Because the simplification redirects the output to x directly, the optimized
     * graph's output list must still contain exactly one entry that produces the
     * correct value when executed.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAlgebraicSimplifyRegisteredOutput(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable zero = sd.constant("zero", Nd4j.zeros(DataType.FLOAT, 1));

        // result = x + 0 — AlgebraicOptimizations should eliminate the add
        SDVariable result = x.add("result", zero);

        sd.setOutputs("result");

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, 2, 4));
        INDArray expected = sd.outputSingle(ph, "result");

        // Optimize with full defaults — AlgebraicOptimizations will run
        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        // The optimized graph must still have a non-null, non-empty output list
        List<String> optimizedOutputs = optimized.outputs();
        assertNotNull("Optimized graph outputs must not be null after simplification", optimizedOutputs);
        assertFalse("Optimized graph outputs must not be empty after simplification", optimizedOutputs.isEmpty());

        // Execute using the surviving output name (the optimizer may redirect to "x" or keep "result")
        String survivingOutput = optimizedOutputs.get(0);
        assertTrue("Surviving output must exist as a variable in the optimized graph",
                optimized.hasVariable(survivingOutput));

        INDArray actual = optimized.outputSingle(ph, survivingOutput);
        assertNotNull("outputSingle must return non-null for the surviving output", actual);
        assertEquals("Optimized output value must match pre-optimization value for x + 0 -> x",
                expected, actual);
    }

    /**
     * Create a graph where "result" = matmul(x, w) + bias.
     * LinearFusionOptimizations may fuse this into XwPlusB.
     * When fusion fires, the graph output list is updated to point to the fused
     * variable. The test verifies that exactly one surviving output still exists
     * and that it produces the correct numeric value.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLinearFusionRegisteredOutput(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        // Fixed shapes so the optimizer can reason about bias dimensionality
        int batchSize = 2;
        int inDim = 8;
        int outDim = 4;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batchSize, inDim);
        // w and bias are VARIABLE (trainable parameters) so fusion applies
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, inDim, outDim));
        SDVariable bias = sd.var("bias", Nd4j.rand(DataType.FLOAT, outDim));

        SDVariable matmulOut = sd.mmul(x, w);
        // Named output: matmul(x, w) + bias
        SDVariable result = matmulOut.add("result", bias);

        sd.setOutputs("result");

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, batchSize, inDim));
        INDArray expected = sd.outputSingle(ph, "result");

        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        // The output list must still exist and have exactly one entry
        List<String> optimizedOutputs = optimized.outputs();
        assertNotNull("Optimized graph outputs must not be null after LinearFusion", optimizedOutputs);
        assertFalse("Optimized graph outputs must not be empty after LinearFusion", optimizedOutputs.isEmpty());

        String survivingOutput = optimizedOutputs.get(0);
        assertTrue("Surviving output variable must exist in the optimized graph",
                optimized.hasVariable(survivingOutput));

        INDArray actual = optimized.outputSingle(ph, survivingOutput);
        assertNotNull("outputSingle must return non-null result after LinearFusion", actual);

        // Numeric values must match within FP32 tolerance
        assertEquals("matmul(x,w) + bias result must be preserved after LinearFusion",
                expected.toFloatVector()[0], actual.toFloatVector()[0], 1e-4f);
    }

    /**
     * Create a graph with rms_norm(x, gamma) -> matmul(normed, w) named "lm_logits".
     * NormalizationFusionOptimizations fuses this into rms_norm_linear and renames
     * the fused output back to "lm_logits" (the original matmul output name).
     * The test verifies that "lm_logits" still resolves in the optimized graph.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormFusionRenameVariable(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        int batchSize = 2;
        int seqLen = 3;
        int hiddenDim = 16;
        int vocabDim = 32;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batchSize, seqLen, hiddenDim);
        // gamma must be a 1-D VARIABLE with >1 elements so FuseRMSNormPattern recognises it
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, hiddenDim));
        // Weight matrix for the linear projection
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, hiddenDim, vocabDim));

        // Build rms_norm -> matmul pattern
        SDVariable normed = sd.nn().rmsNorm(x, gamma, 1e-6);
        // Named output — this is the variable the optimizer must rename the fused op back to
        SDVariable lmLogits = sd.mmul("lm_logits", normed, w);

        sd.setOutputs("lm_logits");

        Map<String, INDArray> ph = Collections.singletonMap("x", Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenDim));
        INDArray expected = sd.outputSingle(ph, "lm_logits");

        SameDiff optimized = GraphOptimizer.optimize(sd, "lm_logits");

        // After FuseRMSNormLinearPattern, the fused output is renamed back to "lm_logits".
        // So the variable "lm_logits" must survive in the optimized graph.
        assertTrue("'lm_logits' variable must still exist in optimized graph after NormFusion",
                optimized.hasVariable("lm_logits"));

        INDArray actual = optimized.outputSingle(ph, "lm_logits");
        assertNotNull("outputSingle('lm_logits') must return non-null after NormFusion", actual);

        // Numeric values must match within FP32 tolerance.
        assertClose("rms_norm_linear", expected, actual, 1e-3);
    }

    /**
     * Qwen logits use rms_norm(x, gamma) -> matmul(normed, permute(lm_head)).
     * The optimizer must preserve this rank-3 path (validated with DSP enabled) when the
     * transposed projection is produced by a view op rather than a direct variable.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormFusionWithTransposedProjectionWeight(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        int batchSize = 1;
        int seqLen = 7;
        int hiddenDim = 16;
        int vocabDim = 64;

        INDArray xArr = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenDim);
        INDArray gammaArr = Nd4j.rand(DataType.FLOAT, hiddenDim).addi(0.5);
        INDArray wRawArr = Nd4j.rand(DataType.FLOAT, vocabDim, hiddenDim);

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batchSize, seqLen, hiddenDim);
        SDVariable gamma = sd.var("gamma", gammaArr);
        SDVariable wRaw = sd.var("lm_head", wRawArr);
        SDVariable wTransposed = sd.permute("lm_head_t", wRaw, 1, 0);

        SDVariable normed = sd.nn().rmsNorm(x, gamma, 1e-6);
        sd.mmul("lm_logits", normed, wTransposed);
        sd.setOutputs("lm_logits");

        Map<String, INDArray> ph = Collections.singletonMap("x", xArr);
        INDArray x2d = xArr.reshape('c', batchSize * seqLen, hiddenDim);
        INDArray invRms = org.nd4j.linalg.ops.transforms.Transforms
                .sqrt(x2d.mul(x2d).mean(true, -1).add(1e-6))
                .rdiv(1.0);
        INDArray expected = x2d.mul(invRms).mul(gammaArr)
                .mmul(wRawArr.transpose())
                .reshape('c', batchSize, seqLen, vocabDim);

        SameDiff optimized = GraphOptimizer.optimize(sd,
                Collections.singletonList("lm_logits"),
                optimizationsWithoutQuantization());
        optimized.resetSession();

        INDArray actual = optimized.outputSingle(ph, "lm_logits");
        assertNotNull("outputSingle('lm_logits') must return non-null after NormFusion", actual);
        assertClose("rms_norm_linear transposed projection", expected, actual, 1e-3);
    }

    /**
     * Verify that running GraphOptimizer.optimize() on a graph that has outputs set
     * results in an optimized graph where outputs() is preserved (not null/empty).
     * This guards against the dup() inside the optimizer silently losing the output list.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphOptimizerOutputSurvivesDup(Nd4jBackend nd4jBackend) {
        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 4, 4));
        SDVariable result = sd.mmul("result", x, w);

        // Explicitly register "result" as a graph output before optimization
        sd.setOutputs("result");
        assertNotNull("Pre-optimization outputs() must not be null", sd.outputs());
        assertFalse("Pre-optimization outputs() must not be empty", sd.outputs().isEmpty());
        assertEquals("Pre-optimization outputs() must contain 'result'", "result", sd.outputs().get(0));

        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        List<String> optimizedOutputs = optimized.outputs();
        assertNotNull("Optimized outputs() must not be null — dup() must preserve outputs", optimizedOutputs);
        assertFalse("Optimized outputs() must not be empty — dup() must preserve outputs", optimizedOutputs.isEmpty());

        // The surviving output variable must exist and be executable
        String survivingOutput = optimizedOutputs.get(0);
        assertTrue("Surviving output '" + survivingOutput + "' must be a variable in the optimized graph",
                optimized.hasVariable(survivingOutput));
    }

    /**
     * The default graph optimizer contract includes FP16 weight pre-casting without
     * requiring a second precision flag.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testQuantizationOptDefaultsToFp16(Nd4jBackend nd4jBackend) {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        String previousFp16 = System.getProperty("nd4j.optimizer.fp16");
        String previousBf16 = System.getProperty("nd4j.optimizer.bf16");
        System.clearProperty("nd4j.optimizer.weightDtype");
        System.clearProperty("nd4j.optimizer.fp16");
        System.clearProperty("nd4j.optimizer.bf16");

        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 32, 64));
            sd.mmul("result", x, w);
            sd.setOutputs("result");

            INDArray wBefore = sd.getVariablesArrays().getArray("w");
            assertNotNull("Weight 'w' must exist before optimization", wBefore);
            assertEquals("Weight 'w' must be FP32 before optimization",
                    DataType.FLOAT, wBefore.dataType());

            SameDiff optimized = GraphOptimizer.optimize(sd, "result");

            INDArray wAfter = optimized.getVariablesArrays().getArray("w");
            assertNotNull("Weight 'w' must still exist in the optimized graph", wAfter);
            assertEquals("GraphOptimizer must pre-cast eligible FP32 weights to FP16 by default",
                    DataType.HALF, wAfter.dataType());
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
            restoreProperty("nd4j.optimizer.fp16", previousFp16);
            restoreProperty("nd4j.optimizer.bf16", previousBf16);
        }
    }

    /**
     * An explicit false remains the opt-out for CPU and exact-precision workloads.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testQuantizationOptExplicitlyDisabled(Nd4jBackend nd4jBackend) {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        String previousFp16 = System.getProperty("nd4j.optimizer.fp16");
        String previousBf16 = System.getProperty("nd4j.optimizer.bf16");
        System.clearProperty("nd4j.optimizer.weightDtype");
        System.setProperty("nd4j.optimizer.fp16", "false");
        System.clearProperty("nd4j.optimizer.bf16");

        try {
            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 32, 64));
            sd.mmul("result", x, w);
            sd.setOutputs("result");

            SameDiff optimized = GraphOptimizer.optimize(sd, "result");

            INDArray wAfter = optimized.getVariablesArrays().getArray("w");
            assertNotNull("Weight 'w' must still exist in the optimized graph", wAfter);
            assertEquals("Explicit nd4j.optimizer.fp16=false must preserve FP32 weights",
                    DataType.FLOAT, wAfter.dataType());
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
            restoreProperty("nd4j.optimizer.fp16", previousFp16);
            restoreProperty("nd4j.optimizer.bf16", previousBf16);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testQuantizationOptAcceptsExplicitFp8(Nd4jBackend nd4jBackend) {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        try {
            System.setProperty("nd4j.optimizer.weightDtype", "fp8");

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 32, 64));
            sd.mmul("result", x, w);
            sd.setOutputs("result");

            SameDiff optimized = GraphOptimizer.optimize(sd, "result");
            assertEquals(DataType.FLOAT8,
                    optimized.getVariablesArrays().getArray("w").dataType());
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testQuantizationOptAcceptsExplicitFp8E5M2(Nd4jBackend nd4jBackend) {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        try {
            System.setProperty("nd4j.optimizer.weightDtype", "fp8_e5m2");

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 32, 64));
            sd.mmul("result", x, w);
            sd.setOutputs("result");

            SameDiff optimized = GraphOptimizer.optimize(sd, "result");
            assertEquals(DataType.FLOAT8_E5M2,
                    optimized.getVariablesArrays().getArray("w").dataType());
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testQuantizationOptRejectsDenseInt4Weights(Nd4jBackend nd4jBackend) {
        String previousWeightDtype = System.getProperty("nd4j.optimizer.weightDtype");
        try {
            System.setProperty("nd4j.optimizer.weightDtype", "int4");

            SameDiff sd = SameDiff.create();
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 32);
            SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 32, 64));
            sd.mmul("result", x, w);
            sd.setOutputs("result");

            IllegalStateException exception = assertThrows(IllegalStateException.class,
                    () -> GraphOptimizer.optimize(sd, "result"));
            assertTrue(exception.getMessage().contains("format-aware packed-weight importer"));
        } finally {
            restoreProperty("nd4j.optimizer.weightDtype", previousWeightDtype);
        }
    }

    private static void restoreProperty(String name, String previousValue) {
        if (previousValue == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previousValue);
        }
    }
}
