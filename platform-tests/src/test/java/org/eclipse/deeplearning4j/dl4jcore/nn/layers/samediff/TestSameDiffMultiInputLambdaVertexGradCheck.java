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

package org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.gradientcheck.GraphConfig;
import org.deeplearning4j.gradientcheck.PrintMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.autodiff.util.SameDiffUtils;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for multi-input SameDiffLambdaVertex backprop (gradient-check level).
 *
 * <p>Root cause: the CPU-backend {@code mergeAvg_} and {@code mergeAvgBp_} helpers in
 * {@code libnd4j/include/ops/declarable/helpers/cpu/merge.cpp} used flat linear index
 * {@code e} to read from F-order inputs and write to C-order output, mixing coordinate
 * systems and producing element mis-pairing.  DL4J's workspace system stores DenseLayer
 * activations in F-order, so the vertex received F-order arrays that triggered the bug.
 * The fix uses coordinate-based access (same pattern as {@code mergeMax_}) to correctly
 * handle arbitrary input/output orderings.
 *
 * <p>These tests guard against regressions in that fix, and also verify that the
 * SameDiffGraphVertex backprop correctly routes epsilon arrays to upstream layers.</p>
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@Tag(TagNames.DL4J_OLD_API)
public class TestSameDiffMultiInputLambdaVertexGradCheck extends BaseDL4JTest {

    // ---------------------------------------------------------------------------
    // Test vertex implementations
    // ---------------------------------------------------------------------------

    /** averages two inputs using native mergeAvg op: out = mergeAvg(in0, in1) */
    public static class MergeAvgLambdaVertex extends SameDiffLambdaVertex {
        @Override
        public SDVariable defineVertex(SameDiff sd, VertexInputs inputs) {
            SDVariable in0 = inputs.getInput(0);
            SDVariable in1 = inputs.getInput(1);
            return sd.math.mergeAvg(in0, in1);
        }
    }

    /** averages two inputs using standard ops: out = (in0 + in1) / 2  (equivalent but different ops) */
    public static class AddHalfLambdaVertex extends SameDiffLambdaVertex {
        @Override
        public SDVariable defineVertex(SameDiff sd, VertexInputs inputs) {
            SDVariable in0 = inputs.getInput(0);
            SDVariable in1 = inputs.getInput(1);
            return in0.add(in1).mul(0.5);
        }
    }

    /** scales and adds: out = (in0 * 2 + in1) / 3   (asymmetric — different grads for each input) */
    public static class AsymmetricLambdaVertex extends SameDiffLambdaVertex {
        @Override
        public SDVariable defineVertex(SameDiff sd, VertexInputs inputs) {
            SDVariable in0 = inputs.getInput(0);
            SDVariable in1 = inputs.getInput(1);
            // (2*in0 + in1) / 3
            return in0.mul(2.0).add(in1).div(3.0);
        }
    }

    // ---------------------------------------------------------------------------
    // Helper
    // ---------------------------------------------------------------------------

    private static ComputationGraph buildNet(SameDiffLambdaVertex mergeVertex) {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .updater(new NoOp())
                .seed(12345)
                .activation(Activation.TANH)
                .graphBuilder()
                .addInputs("in")
                // Two dense layers with DISTINCT random weights feeding the same input.
                // Using different seeds / weight-init values ensures that a swapped or
                // duplicated epsilon is actually caught by the gradient check.
                .addLayer("L0", new DenseLayer.Builder()
                        .nIn(5).nOut(4)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build(), "in")
                .addLayer("L1", new DenseLayer.Builder()
                        .nIn(5).nOut(4)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.RELU)   // intentionally different from L0
                        .build(), "in")
                .addVertex("merge", mergeVertex, "L0", "L1")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build(), "merge")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /**
     * Two dense layers → mergeAvg vertex → output.
     *
     * <p>mergeAvg distributes equal gradient 0.5·ε to both inputs.
     * A swapped or shared epsilon would still be equal here but the WEIGHT
     * gradients for L0 and L1 depend on which activations are multiplied by ε,
     * so any epsilon routing error is caught.</p>
     */
    @Test
    public void testMultiInputMergeAvgGradCheck() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(67890);  // fixed seed for reproducibility

        ComputationGraph net = buildNet(new MergeAvgLambdaVertex());

        INDArray features = Nd4j.rand(DataType.DOUBLE, 2, 5);
        INDArray labels   = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}});

        boolean ok = GradientCheckUtil.checkGradients(new GraphConfig()
                .net(net)
                .epsilon(1e-6)
                .maxRelError(1e-5)
                .minAbsoluteError(1e-8)
                .print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(false)
                .inputs(new INDArray[]{features})
                .labels(new INDArray[]{labels}));

        assertTrue(ok, "Gradient check failed for 2-input MergeAvgLambdaVertex");
    }

    /**
     * Isolation: (in0 + in1) * 0.5 using standard ops — mathematically identical to mergeAvg but different impl.
     * Standard element-wise ops handle arbitrary array orderings correctly (stride-aware); this confirms
     * the gradient check framework itself is working correctly.
     */
    @Test
    public void testMultiInputAddHalfGradCheck() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Nd4j.getRandom().setSeed(67890);  // fixed seed for reproducibility

        ComputationGraph net = buildNet(new AddHalfLambdaVertex());

        INDArray features = Nd4j.rand(DataType.DOUBLE, 2, 5);
        INDArray labels   = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}});

        boolean ok = GradientCheckUtil.checkGradients(new GraphConfig()
                .net(net)
                .epsilon(1e-6)
                .maxRelError(1e-5)
                .minAbsoluteError(1e-8)
                .print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(false)
                .inputs(new INDArray[]{features})
                .labels(new INDArray[]{labels}));

        assertTrue(ok, "Gradient check failed for 2-input AddHalfLambdaVertex (standard ops equivalent of mergeAvg)");
    }

    /**
     * Two dense layers → asymmetric vertex (2·in0 + in1)/3 → output.
     *
     * <p>Each input receives a DIFFERENT gradient (2/3·ε vs 1/3·ε).
     * If the epsilon arrays are swapped or otherwise mis-mapped, both inputs
     * would receive the wrong gradient and ALL params (L0_W, L0_b, L1_W, L1_b)
     * will fail the gradient check.</p>
     */
    @Test
    public void testMultiInputAsymmetricGradCheck() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        ComputationGraph net = buildNet(new AsymmetricLambdaVertex());

        INDArray features = Nd4j.rand(DataType.DOUBLE, 2, 5);
        INDArray labels   = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}});

        boolean ok = GradientCheckUtil.checkGradients(new GraphConfig()
                .net(net)
                .epsilon(1e-6)
                .maxRelError(1e-5)
                .minAbsoluteError(1e-8)
                .print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(false)
                .inputs(new INDArray[]{features})
                .labels(new INDArray[]{labels}));

        assertTrue(ok, "Gradient check failed for 2-input AsymmetricLambdaVertex");
    }

    /**
     * Single-input SameDiffLambdaVertex regression guard:
     * existing single-input case must still pass after the multi-input fix.
     */
    @Test
    public void testSingleInputLambdaVertexGradCheckStillPasses() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        // Single-input: just tanh of the input
        SameDiffLambdaVertex singleInput = new SameDiffLambdaVertex() {
            @Override
            public SDVariable defineVertex(SameDiff sd, VertexInputs inputs) {
                return sd.math().tanh(inputs.getInput(0));
            }
        };

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .updater(new NoOp())
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .addLayer("L0", new DenseLayer.Builder()
                        .nIn(5).nOut(4)
                        .activation(Activation.TANH)
                        .build(), "in")
                .addVertex("tanh", singleInput, "L0")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(4).nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build(), "tanh")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        INDArray features = Nd4j.rand(DataType.DOUBLE, 2, 5);
        INDArray labels   = Nd4j.create(new double[][]{{1, 0, 0}, {0, 1, 0}});

        boolean ok = GradientCheckUtil.checkGradients(new GraphConfig()
                .net(net)
                .epsilon(1e-6)
                .maxRelError(1e-5)
                .minAbsoluteError(1e-8)
                .print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(false)
                .inputs(new INDArray[]{features})
                .labels(new INDArray[]{labels}));

        assertTrue(ok, "Single-input SameDiffLambdaVertex gradient check should still pass");
    }

    /**
     * Isolation test: pure SameDiff gradient check of mergeAvg with ExternalErrorsFunction.
     * This mimics exactly what SameDiffGraphVertex.doBackward does internally, without the DL4J layer wrapper.
     * If this fails, the bug is in mergeAvg's backward pass in the context of ExternalErrorsFunction.
     * If this passes but testMultiInputMergeAvgGradCheck fails, the bug is in SameDiffGraphVertex itself.
     */
    @Test
    public void testMergeAvgWithExternalErrorsFunctionIsolation() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        SameDiff sd = SameDiff.create();
        // Simulate the exact SameDiffGraphVertex setup:
        // - var_0 and var_1 are placeholders (corresponding to L0 and L1 outputs)
        // - mergeAvg(var_0, var_1) is the vertex computation
        // - ExternalErrorsFunction wraps the output
        SDVariable var0 = sd.placeHolder("var_0", DataType.DOUBLE, -1, 4);
        SDVariable var1 = sd.placeHolder("var_1", DataType.DOUBLE, -1, 4);
        SDVariable out = sd.math.mergeAvg(var0, var1);
        ExternalErrorsFunction fn = SameDiffUtils.externalErrors(sd, null, out);
        fn.outputVariable();

        // Create gradient function for the placeholders
        sd.createGradFunction("var_0", "var_1");

        // Set placeholder values and compute gradients
        INDArray a0 = Nd4j.rand(DataType.DOUBLE, 2, 4);
        INDArray a1 = Nd4j.rand(DataType.DOUBLE, 2, 4);
        INDArray eps = Nd4j.ones(DataType.DOUBLE, 2, 4);  // uniform epsilon

        Map<String, INDArray> phMap = new java.util.HashMap<>();
        phMap.put("var_0", a0);
        phMap.put("var_1", a1);
        phMap.put(fn.getGradPlaceholderName(), eps);

        Map<String, INDArray> grads = sd.calculateGradients(phMap, Arrays.asList("var_0", "var_1"));
        INDArray grad0 = grads.get("var_0");
        INDArray grad1 = grads.get("var_1");

        System.out.println("grad0 sum = " + grad0.sumNumber() + " (expected: " + eps.sumNumber().doubleValue() * 0.5 + ")");
        System.out.println("grad1 sum = " + grad1.sumNumber() + " (expected: " + eps.sumNumber().doubleValue() * 0.5 + ")");

        // Each gradient should be 0.5 * eps_element
        double expectedGrad = 0.5;  // per element, since eps[i,j] = 1.0
        double actualGrad0 = grad0.getDouble(0, 0);
        double actualGrad1 = grad1.getDouble(0, 0);

        System.out.println("grad0[0,0] = " + actualGrad0 + " (expected: " + expectedGrad + ")");
        System.out.println("grad1[0,0] = " + actualGrad1 + " (expected: " + expectedGrad + ")");

        assertTrue(Math.abs(actualGrad0 - expectedGrad) < 1e-6,
                "mergeAvg gradient for var_0 should be 0.5, got: " + actualGrad0);
        assertTrue(Math.abs(actualGrad1 - expectedGrad) < 1e-6,
                "mergeAvg gradient for var_1 should be 0.5, got: " + actualGrad1);
    }
}
