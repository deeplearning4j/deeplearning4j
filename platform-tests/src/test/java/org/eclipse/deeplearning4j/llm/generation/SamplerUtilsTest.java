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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SamplerUtilsTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgmax(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 3.0f, 2.0f, 0.5f});
        int idx = SamplerUtils.argmax(logits);
        assertEquals(1, idx);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgmaxBatch(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[][]{
                {1.0f, 3.0f, 2.0f},
                {5.0f, 1.0f, 2.0f}
        });
        int[] result = SamplerUtils.argmaxBatch(logits);
        assertEquals(2, result.length);
        assertEquals(1, result[0]);
        assertEquals(0, result[1]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmaxSumsToOne(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
        INDArray probs = SamplerUtils.softmax(logits);

        double sum = probs.sumNumber().doubleValue();
        assertEquals(1.0, sum, 1e-5);

        for (int i = 0; i < 4; i++) {
            assertTrue(probs.getFloat(i) > 0);
        }

        assertEquals(3, SamplerUtils.argmax(probs));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSoftmax2D(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[][]{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
        INDArray probs = SamplerUtils.softmax(logits);

        assertEquals(2, probs.rank());
        assertEquals(1.0, probs.getRow(0).sumNumber().doubleValue(), 1e-5);
        assertEquals(1.0, probs.getRow(1).sumNumber().doubleValue(), 1e-5);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKFilter(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 5.0f, 3.0f, 2.0f, 4.0f});
        INDArray filtered = SamplerUtils.topKFilter(logits, 2);

        int nonInfCount = 0;
        for (int i = 0; i < 5; i++) {
            float val = filtered.getFloat(i);
            if (!Float.isInfinite(val)) {
                nonInfCount++;
            }
        }
        assertEquals(2, nonInfCount);
        assertEquals(5.0f, filtered.getFloat(1), 1e-6);
        assertEquals(4.0f, filtered.getFloat(4), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKFilterDisabled(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f});
        INDArray filtered = SamplerUtils.topKFilter(logits, 0);
        assertSame(logits, filtered);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTemperatureScaling(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{2.0f, 4.0f, 6.0f});
        INDArray scaled = SamplerUtils.applyTemperature(logits, 2.0);
        assertEquals(1.0f, scaled.getFloat(0), 1e-6);
        assertEquals(2.0f, scaled.getFloat(1), 1e-6);
        assertEquals(3.0f, scaled.getFloat(2), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTemperatureIdentity(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f});
        INDArray result = SamplerUtils.applyTemperature(logits, 1.0);
        assertSame(logits, result);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testIsValidDistribution(Nd4jBackend backend) {
        INDArray valid = Nd4j.createFromArray(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        assertTrue(SamplerUtils.isValidDistribution(valid, 1e-5));

        INDArray invalid = Nd4j.createFromArray(new float[]{0.5f, 0.5f, 0.5f});
        assertFalse(SamplerUtils.isValidDistribution(invalid, 1e-5));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEntropy(Nd4jBackend backend) {
        INDArray uniform = Nd4j.createFromArray(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        double uniformEntropy = SamplerUtils.entropy(uniform);

        INDArray peaked = Nd4j.createFromArray(new float[]{0.97f, 0.01f, 0.01f, 0.01f});
        double peakedEntropy = SamplerUtils.entropy(peaked);

        assertTrue(uniformEntropy > peakedEntropy);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRepetitionPenalty(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        int[] generated = {1, 3};
        INDArray penalized = SamplerUtils.applyRepetitionPenalty(logits, generated, 2.0);

        assertEquals(1.0f, penalized.getFloat(0), 1e-6);
        assertEquals(1.0f, penalized.getFloat(1), 1e-6); // 2.0 / 2.0
        assertEquals(3.0f, penalized.getFloat(2), 1e-6);
        assertEquals(2.0f, penalized.getFloat(3), 1e-6); // 4.0 / 2.0
        assertEquals(5.0f, penalized.getFloat(4), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultinomialSampleDeterministic(Nd4jBackend backend) {
        INDArray probs = Nd4j.createFromArray(new float[]{0.001f, 0.001f, 0.997f, 0.001f});
        Random random = new Random(42);

        int count2 = 0;
        for (int i = 0; i < 100; i++) {
            if (SamplerUtils.multinomialSample(probs, random) == 2) {
                count2++;
            }
        }
        assertTrue(count2 > 90, "Expected most samples at index 2, got " + count2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLogSoftmax(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new float[]{1.0f, 2.0f, 3.0f});
        INDArray logProbs = SamplerUtils.logSoftmax(logits);

        assertTrue(logProbs.maxNumber().doubleValue() <= 0);

        INDArray expLogProbs = Nd4j.math().exp(logProbs);
        INDArray softmax = SamplerUtils.softmax(logits);

        for (int i = 0; i < 3; i++) {
            assertEquals(softmax.getDouble(i), expLogProbs.getDouble(i), 1e-4);
        }
    }
}
