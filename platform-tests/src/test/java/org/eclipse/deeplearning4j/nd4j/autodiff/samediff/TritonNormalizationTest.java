/*
 *  ******************************************************************************
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone tests for normalization operations (rms_norm, layer_norm) with Triton GPU backend.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonNormalizationTest extends BaseNd4jTestWithBackends {

    /**
     * Test: RMS norm basic pattern (as used in SmolDocling decoder).
     * rms_norm(x, gamma, eps) = x * rsqrt(mean(x^2) + eps) * gamma
     */
    @Test
    @DisplayName("RMS Norm: Basic [1, 576] with gamma weight")
    public void testRmsNormBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 576));
        double eps = 1e-5;

        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        TritonTestUtils.runOpTest("testRmsNormBasic", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: RMS norm with non-trivial gamma weight.
     */
    @Test
    @DisplayName("RMS Norm: With non-trivial gamma weight")
    public void testRmsNormWithGamma() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        float[] gammaData = new float[576];
        for (int i = 0; i < 576; i++) {
            gammaData[i] = (i % 2 == 0) ? 0.5f : 1.5f;
        }
        SDVariable gamma = sd.constant("gamma", Nd4j.create(gammaData, new long[]{576}));
        double eps = 1e-5;

        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        TritonTestUtils.runOpTest("testRmsNormWithGamma", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: RMS norm gamma=2.0 scaling - Java executor reference.
     * Verifies the fallback path (non-Triton) applies gamma correctly.
     */
    @Test
    @DisplayName("RMS Norm: Gamma=2.0 Java fallback reference")
    public void testRmsNormGammaJavaFallback() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        float[] xData = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 8).muli(2.0));
        double eps = 1e-5;

        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        INDArray xArr = Nd4j.create(xData, new long[]{1, 8});
        Map<String, INDArray> ref = sd.output(Map.of("x", xArr), "result");
        INDArray actual = ref.get("result");
        
        // Manual calculation
        double meanSqVal = 0.0;
        for (float v : xData) meanSqVal += v * v;
        meanSqVal /= 8.0;
        double scale = 2.0 / Math.sqrt(meanSqVal + eps);  // gamma=2.0
        
        log.info("Java fallback: expected scale={:.6f}, actual first4=[{}, {}, {}, {}]",
            scale, actual.getFloat(0), actual.getFloat(1), actual.getFloat(2), actual.getFloat(3));
        
        // Verify gamma was applied (values should be 2x the normalized values)
        for (int i = 0; i < 8; i++) {
            float expected = (float)(xData[i] * scale);
            float actualVal = actual.getFloat(0, i);
            double diff = Math.abs(expected - actualVal);
            assertTrue(diff < 1e-5, String.format("Gamma scaling mismatch at index %d: expected %.6f, got %.6f", i, expected, actualVal));
        }
    }

    /**
     * Test: RMS norm with gamma=2.0 to make scaling obvious.
     * If gamma is ignored, output will be 0.5x expected.
     */
    @Test
    @DisplayName("RMS Norm: Gamma=2.0 explicit scaling test")
    public void testRmsNormGammaScaling() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        // Simple test vector where we can verify gamma scaling manually
        float[] xData = new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        // gamma = 2.0 for all elements
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 8).muli(2.0));
        double eps = 1e-5;

        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        // Expected: rms_norm(x) * 2.0
        INDArray xArr = Nd4j.create(xData, new long[]{1, 8});
        INDArray expected = xArr.dup();
        double meanSqVal = 0.0;
        for (float v : xData) meanSqVal += v * v;
        meanSqVal /= 8.0;
        double scale = 1.0 / Math.sqrt(meanSqVal + eps);
        for (int i = 0; i < 8; i++) expected.putScalar(0, i, (float)(xData[i] * scale * 2.0));

        Map<String, INDArray> ref = sd.output(Map.of("x", xArr), "result");
        INDArray actual = ref.get("result");
        
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("testRmsNormGammaScaling: expected first4=[{}, {}, {}, {}], actual first4=[{}, {}, {}, {}], maxDiff={}",
            expected.getFloat(0), expected.getFloat(1), expected.getFloat(2), expected.getFloat(3),
            actual.getFloat(0), actual.getFloat(1), actual.getFloat(2), actual.getFloat(3), maxDiff);
        
        assertTrue(maxDiff < 1e-4, "Gamma scaling test failed: maxDiff=" + maxDiff);
    }

    /**
     * Test: RMS norm with batched input [4, 576].
     */
    @Test
    @DisplayName("RMS Norm: Batched [4, 576]")
    public void testRmsNormBatched() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 576));
        double eps = 1e-5;

        SDVariable squared = x.mul("square", x);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = x.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        TritonTestUtils.runOpTest("testRmsNormBatched", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 4, 576)), "result");
    }

    /**
     * Test: RMS norm with residual connection.
     */
    @Test
    @DisplayName("RMS Norm: With residual connection")
    public void testRmsNormWithResidual() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable residual = sd.constant("residual", Nd4j.randn(DataType.FLOAT, 1, 576));
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 576));
        double eps = 1e-5;

        SDVariable added = x.add("add", residual);
        SDVariable squared = added.mul("square", added);
        SDVariable meanSq = sd.mean("mean", squared, true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = meanSq.add("add_eps", epsConst);
        SDVariable rsqrtDenom = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = added.mul("normed", rsqrtDenom);
        normed.mul("result", gamma);

        TritonTestUtils.runOpTest("testRmsNormWithResidual", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: Layer norm basic pattern.
     */
    @Test
    @DisplayName("Layer Norm: Basic [1, 576]")
    public void testLayerNormBasic() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 576);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 576));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 576));
        double eps = 1e-5;

        SDVariable mean = sd.mean("mean", x, true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable variance = sd.mean("var", centered.mul(centered), true, 1);
        SDVariable epsConst = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, (float)eps));
        SDVariable denom = variance.add("add_eps", epsConst);
        SDVariable stdInv = sd.math.rsqrt("rsqrt", denom);
        SDVariable normed = centered.mul("normed", stdInv);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        TritonTestUtils.runOpTest("testLayerNormBasic", sd, Map.of("x", Nd4j.randn(DataType.FLOAT, 1, 576)), "result");
    }

    /**
     * Test: Manual batch norm via element-wise ops.
     */
    @Test
    @DisplayName("Batch Norm: Manual [4, 8]")
    public void testManualBatchNorm() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 1, 8));
        SDVariable beta = sd.constant("beta", Nd4j.zeros(DataType.FLOAT, 1, 8));

        SDVariable mean = sd.mean("mean", x, true, 1);
        SDVariable centered = x.sub("centered", mean);
        SDVariable variance = sd.mean("variance", centered.mul(centered), true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable stdInv = sd.math.rsqrt("stdInv", variance.add(eps));
        SDVariable normed = centered.mul("normed", stdInv);
        SDVariable scaled = normed.mul("scaled", gamma);
        scaled.add("result", beta);

        TritonTestUtils.runOpTest("testManualBatchNorm", sd, Map.of("input", Nd4j.randn(DataType.FLOAT, 4, 8)), "result");
    }
}
