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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that binary pow(x, 2.0) does NOT produce NaN for negative x values.
 *
 * <p>LayerNorm computes variance as mean(pow(x - mean, 2.0)). When (x - mean) is negative
 * (roughly half the values), the naive Triton emitter exp(2*log(x)) produces NaN because
 * log(negative) = NaN. The fix uses abs(x) before log: exp(2*log(|x|)) = x^2.</p>
 */
@Slf4j
@Tag("samediff")
public class TestTritonPowNaN extends BaseNd4jTestWithBackends {

    @Test
    @DisplayName("pow(negative, 2) via binary path should not NaN")
    public void testBinaryPowNegativeBase() {
        // Build graph: input -> subtract(mean) -> pow(_, 2.0) -> output
        // The subtract creates negative values, pow with exponent 2 must handle them.
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable mean = sd.math.mean(input, true, 1);  // [batch, 1]
        SDVariable diff = input.sub("diff", mean);  // [batch, 8] — has negative values
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        // Binary pow: diff^two (both are tensors from SameDiff's perspective)
        SDVariable powered = sd.math.pow("powered", diff, two);
        SDVariable variance = sd.math.mean("variance", powered, true, 1);
        SDVariable output = sd.identity("output", variance);

        // Enable DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Input with values that guarantee negative (x - mean)
        INDArray inputArr = Nd4j.createFromArray(new float[][]{
                {-3.0f, -1.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                {10.0f, -10.0f, 5.0f, -5.0f, 3.0f, -3.0f, 1.0f, -1.0f}
        });  // mean ~ 1.4375 and 0.0, lots of negative (x-mean)

        // Reference: compute expected variance manually
        // For each row: variance = mean((x - mean)^2)
        INDArray meanArr = inputArr.mean(true, 1);
        INDArray diffArr = inputArr.sub(meanArr);
        INDArray poweredArr = diffArr.mul(diffArr);  // x^2
        INDArray expectedVariance = poweredArr.mean(true, 1);

        // Execute via DSP — this will use Triton binary pow
        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", inputArr), "output");
        INDArray actual = result.get("output");

        log.info("Expected variance: {}", expectedVariance);
        log.info("Actual variance:   {}", actual);

        // Must not contain NaN
        assertFalse(actual.isNaN().any(),
                "pow(negative, 2) produced NaN — binary pow emitter broken for negative base");

        // Must match reference within tolerance
        INDArray diff2 = Transforms.abs(actual.sub(expectedVariance));
        float maxDiff = diff2.maxNumber().floatValue();
        assertTrue(maxDiff < 0.01f,
                "pow(negative, 2) result too far from reference: maxDiff=" + maxDiff);
    }

    @Test
    @DisplayName("pow(negative, 2) multi-call stability")
    public void testBinaryPowMultiCallStability() {
        // Verify no NaN across multiple DSP executions (frozen shapes path)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 16);
        SDVariable mean = sd.math.mean(input, true, 1);
        SDVariable diff = input.sub("diff", mean);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable powered = sd.math.pow("powered", diff, two);
        SDVariable output = sd.math.mean("output", powered, true, 1);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int call = 0; call < 5; call++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 2, 16).muli(10);  // range ~ -30..+30
            Map<String, INDArray> result = sd.output(
                    Collections.singletonMap("input", inputArr), "output");
            INDArray actual = result.get("output");

            assertFalse(actual.isNaN().any(),
                    "Call " + call + ": pow(negative, 2) produced NaN");
            assertTrue(actual.minNumber().floatValue() >= 0,
                    "Call " + call + ": variance should be non-negative, got " + actual.minNumber());
        }
    }

    @Test
    @DisplayName("LayerNorm-like pattern: (x-mean)/sqrt(var+eps)")
    public void testLayerNormPattern() {
        // Full LayerNorm-like computation to verify end-to-end correctness
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable mean = sd.math.mean(input, true, 1);
        SDVariable diff = input.sub("centered", mean);
        SDVariable two = sd.constant("two", Nd4j.scalar(DataType.FLOAT, 2.0f));
        SDVariable sq = sd.math.pow("sq", diff, two);
        SDVariable variance = sd.math.mean("var", sq, true, 1);
        SDVariable eps = sd.constant("eps", Nd4j.scalar(DataType.FLOAT, 1e-5f));
        SDVariable varEps = variance.add("varEps", eps);
        SDVariable std = sd.math.sqrt("std", varEps);
        SDVariable normalized = diff.div("output", std);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.createFromArray(new float[][]{
                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}
        });

        // Reference: numpy-style layer norm
        float mean_ = 4.5f;
        float[] centered = {-3.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.5f};
        float var_ = 0;
        for (float c : centered) var_ += c * c;
        var_ /= 8;
        float std_ = (float) Math.sqrt(var_ + 1e-5f);

        Map<String, INDArray> result = sd.output(
                Collections.singletonMap("input", inputArr), "output");
        INDArray actual = result.get("output");

        assertFalse(actual.isNaN().any(), "LayerNorm produced NaN");

        // Check first and last values
        float actualFirst = actual.getFloat(0, 0);
        float expectedFirst = centered[0] / std_;
        assertEquals(expectedFirst, actualFirst, 0.01f,
                "LayerNorm first element mismatch");

        float actualLast = actual.getFloat(0, 7);
        float expectedLast = centered[7] / std_;
        assertEquals(expectedLast, actualLast, 0.01f,
                "LayerNorm last element mismatch");
    }
}
