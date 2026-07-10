/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal isolation for the BGE fp16 NaN: DSP slot 109 (reduce_mean over a finite
 * HALF [1,512,768] input, dim=2, keepDims=true) writes NaN on CUDA.
 * Splits the repro into eager vs SameDiff/DSP so the fault lands either in the
 * native reduce kernel or the DSP slot execution path.
 */
@Slf4j
public class HalfReduceMeanNaNReproTest extends BaseND4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 10 * 60 * 1000L;
    }

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public DataType getDefaultFPDataType() {
        return DataType.FLOAT;
    }

    private static INDArray embeddingsLikeHalf() {
        // Mirror the BGE embeddings-sum magnitudes (min≈-3.07, max≈2.13, mean≈-0.023)
        INDArray x = Nd4j.rand(DataType.FLOAT, 1, 512, 768).muli(5.2).subi(3.07);
        return x.castTo(DataType.HALF);
    }

    @Test
    public void testEagerHalfReduceMeanDim2() {
        INDArray xHalf = embeddingsLikeHalf();
        assertFalse(xHalf.isNaN().any(), "input must be finite");

        INDArray mean = xHalf.mean(true, 2);
        assertArrayEquals(new long[]{1, 512, 1}, mean.shape());
        boolean nan = mean.isNaN().any();
        boolean inf = mean.isInfinite().any();
        log.info("eager HALF mean dim2: dtype={} hasNaN={} hasInf={} min={} max={}",
                mean.dataType(), nan, inf,
                nan || inf ? Double.NaN : mean.minNumber().doubleValue(),
                nan || inf ? Double.NaN : mean.maxNumber().doubleValue());

        // fp32 reference
        INDArray ref = xHalf.castTo(DataType.FLOAT).mean(true, 2);
        assertFalse(nan || inf, "eager HALF reduce_mean produced non-finite values");
        double maxDiff = ref.sub(mean.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        log.info("eager HALF mean dim2 vs fp32 ref maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "eager HALF reduce_mean diverges from fp32 reference: " + maxDiff);
    }

    @Test
    public void testEagerHalfScalarMeanLargeTensor() {
        // Full-array (scalar-output) HALF mean exercises execScalarCuda's cross-block
        // staging. Pre-fix that staging narrowed float partials to HALF: per-block
        // partial SUMS of large positive tensors exceed 65504 → inf → non-finite mean.
        INDArray xHalf = Nd4j.valueArrayOf(new long[]{4 * 1024 * 1024}, 1.1, DataType.HALF);
        double mean = xHalf.meanNumber().doubleValue();
        log.info("eager HALF scalar mean of 4M x 1.1: {}", mean);
        assertTrue(Double.isFinite(mean), "scalar HALF mean overflowed cross-block staging: " + mean);
        assertEquals(1.1, mean, 2e-3, "scalar HALF mean lost precision in cross-block staging");
        xHalf.close();
    }

    @Test
    public void testEagerHalfScalarSumPrecision() {
        // reduce_same previously accumulated HALF sums in HALF: past ~32768 the fp16
        // ulp (32) exceeds the 0.5 addend, so the sum silently stops growing.
        // With InterType (float) accumulation the true value 50000 is reached; the
        // only remaining error is the final HALF output quantization (ulp 32 at 50k).
        INDArray xHalf = Nd4j.valueArrayOf(new long[]{100_000}, 0.5, DataType.HALF);
        double sum = xHalf.sumNumber().doubleValue();
        log.info("eager HALF scalar sum of 100k x 0.5: {}", sum);
        assertEquals(50_000.0, sum, 64.0, "HALF sum saturated during accumulation");
        xHalf.close();
    }

    @Test
    public void testEagerHalfSoftmaxMatchesFp32() {
        // softmax kernels previously accumulated the exp-sum in HALF; rows must now
        // match the fp32 reference within HALF output quantization.
        Nd4j.getRandom().setSeed(42);
        INDArray logits = Nd4j.rand(DataType.FLOAT, 8, 1024).muli(8.0).subi(4.0);
        INDArray logitsHalf = logits.castTo(DataType.HALF);

        INDArray smHalf = Nd4j.nn().softmax(logitsHalf, 1);
        INDArray smRef = Nd4j.nn().softmax(logitsHalf.castTo(DataType.FLOAT), 1);

        assertFalse(smHalf.isNaN().any(), "HALF softmax produced NaN");
        double maxDiff = smRef.sub(smHalf.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        double rowSumErr = smHalf.castTo(DataType.FLOAT).sum(1).sub(1.0).amaxNumber().doubleValue();
        log.info("eager HALF softmax vs fp32: maxDiff={} rowSumErr={}", maxDiff, rowSumErr);
        assertTrue(maxDiff < 2e-3, "HALF softmax diverges from fp32 reference: " + maxDiff);
        assertTrue(rowSumErr < 5e-3, "HALF softmax rows do not sum to 1: " + rowSumErr);
        logits.close();
        logitsHalf.close();
    }

    @Test
    public void testSameDiffDspHalfReduceMeanDim2() {
        Map<String, String> prev = new LinkedHashMap<>();
        prev.put(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED,
                System.getProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED));
        prev.put(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE,
                System.getProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE));
        boolean prevDsp = InferenceSession.isDynamicShapePlanEnabled();
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        System.setProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE, "AUTO");
        InferenceSession.setDynamicShapePlanEnabled(true);
        SameDiff sd = SameDiff.create();
        try {
            SDVariable in = sd.placeHolder("in", DataType.HALF, 1, 512, 768);
            SDVariable out = sd.mean("out", in, true, 2);

            for (int i = 0; i < 3; i++) {
                INDArray xHalf = embeddingsLikeHalf();
                Map<String, INDArray> result = sd.output(Map.of("in", xHalf), "out");
                INDArray mean = result.get("out");
                assertNotNull(mean, "iteration " + i + ": missing output");
                boolean nan = mean.isNaN().any();
                boolean inf = mean.isInfinite().any();
                log.info("DSP HALF mean dim2 iteration {}: dtype={} hasNaN={} hasInf={}",
                        i, mean.dataType(), nan, inf);
                assertFalse(nan || inf,
                        "iteration " + i + ": DSP HALF reduce_mean produced non-finite values");
                INDArray ref = xHalf.castTo(DataType.FLOAT).mean(true, 2);
                double maxDiff = ref.sub(mean.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-2,
                        "iteration " + i + ": DSP HALF reduce_mean diverges from fp32 reference: " + maxDiff);
            }
        } finally {
            sd.close();
            for (Map.Entry<String, String> e : prev.entrySet()) {
                if (e.getValue() == null) System.clearProperty(e.getKey());
                else System.setProperty(e.getKey(), e.getValue());
            }
            InferenceSession.setDynamicShapePlanEnabled(prevDsp);
        }
    }
}
