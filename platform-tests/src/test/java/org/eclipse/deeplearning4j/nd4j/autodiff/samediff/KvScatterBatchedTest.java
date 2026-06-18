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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class KvScatterBatchedTest {

    @Test
    @DisplayName("Single KV scatter: FP32")
    public void testSingleKvScatterFloat() {
        int batch = 1, heads = 8, srcSeqLen = 10, maxKvLen = 100, dim = 64;
        long cachePos = 7;

        INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, srcSeqLen, dim);
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

        KvScatter scatter = new KvScatter(staticBuf, present, cachePos);
        INDArray[] result = Nd4j.getExecutioner().exec(scatter);

        // Expected: present[:, :, lastPos, :] copied to staticBuf[:, :, cachePos, :]
        INDArray expected = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(srcSeqLen - 1), NDArrayIndex.all()).dup();
        INDArray actual = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Single FP32 scatter maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "FP32 scatter should be exact, got maxDiff=" + maxDiff);

        // Verify other positions remain zero
        INDArray otherPos = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(0), NDArrayIndex.all()).dup();
        double otherMax = otherPos.amaxNumber().doubleValue();
        assertEquals(0.0, otherMax, 1e-10, "Unwritten positions should remain zero");

        present.close();
        staticBuf.close();
    }

    @Test
    @DisplayName("Single KV scatter: FP16")
    public void testSingleKvScatterHalf() {
        int batch = 1, heads = 4, srcSeqLen = 5, maxKvLen = 50, dim = 32;
        long cachePos = 3;

        INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, srcSeqLen, dim).castTo(DataType.HALF);
        INDArray staticBuf = Nd4j.zeros(DataType.HALF, batch, heads, maxKvLen, dim);

        KvScatter scatter = new KvScatter(staticBuf, present, cachePos);
        Nd4j.getExecutioner().exec(scatter);

        INDArray expected = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(srcSeqLen - 1), NDArrayIndex.all()).dup().castTo(DataType.FLOAT);
        INDArray actual = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup().castTo(DataType.FLOAT);

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Single FP16 scatter maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "FP16 scatter should match, got maxDiff=" + maxDiff);

        present.close();
        staticBuf.close();
    }

    @Test
    @DisplayName("Multiple sequential KV scatters simulate decode loop")
    public void testSequentialKvScatters() {
        // Simulates what happens during VLM decode: each step scatters 60 KV pairs
        // (30 layers x 2 for key+value). This test uses a smaller scale.
        int batch = 1, heads = 4, maxKvLen = 20, dim = 16;
        int numLayers = 3;  // 3 layers x 2 (key+value) = 6 scatters per step
        int numSteps = 5;

        // Create present KV and static buffers for each layer (key + value)
        INDArray[] presents = new INDArray[numLayers * 2];
        INDArray[] statics = new INDArray[numLayers * 2];
        for (int i = 0; i < numLayers * 2; i++) {
            statics[i] = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);
        }

        // Simulate multiple decode steps
        for (int step = 0; step < numSteps; step++) {
            long cachePos = step;

            // Each step generates new present KV with unique values
            for (int i = 0; i < numLayers * 2; i++) {
                presents[i] = Nd4j.randn(DataType.FLOAT, batch, heads, 1, dim)
                        .addi(step * 10 + i);  // unique per step+layer

                KvScatter scatter = new KvScatter(statics[i], presents[i], cachePos);
                Nd4j.getExecutioner().exec(scatter);
            }

            // Verify each scatter wrote to the correct position
            for (int i = 0; i < numLayers * 2; i++) {
                INDArray expected = presents[i].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(0), NDArrayIndex.all()).dup();
                INDArray actual = statics[i].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

                double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-5,
                        String.format("Step %d, layer %d: maxDiff=%e", step, i, maxDiff));

                presents[i].close();
            }
        }

        // Verify all positions 0..numSteps-1 are non-zero, positions numSteps..maxKvLen-1 are zero
        for (int i = 0; i < numLayers * 2; i++) {
            for (int pos = 0; pos < numSteps; pos++) {
                INDArray entry = statics[i].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(pos), NDArrayIndex.all()).dup();
                double norm = entry.norm2Number().doubleValue();
                assertTrue(norm > 0.1,
                        String.format("Layer %d pos %d should be non-zero, norm=%e", i, pos, norm));
                entry.close();
            }
            INDArray emptyEntry = statics[i].get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(numSteps + 1), NDArrayIndex.all()).dup();
            double emptyNorm = emptyEntry.norm2Number().doubleValue();
            assertEquals(0.0, emptyNorm, 1e-10,
                    String.format("Layer %d pos %d should be zero", i, numSteps + 1));
            emptyEntry.close();
            statics[i].close();
        }

        log.info("Sequential KV scatter test passed: {} steps x {} scatters", numSteps, numLayers * 2);
    }

    @Test
    @DisplayName("KV scatter with varying head counts")
    public void testVaryingHeadCounts() {
        // The batched kernel uses binary search over entries — exercise with different head counts
        int[] headCounts = {1, 4, 8, 16, 32};
        int batch = 1, dim = 64, maxKvLen = 50, srcSeqLen = 3;
        long cachePos = 10;

        for (int heads : headCounts) {
            INDArray present = Nd4j.randn(DataType.FLOAT, batch, heads, srcSeqLen, dim);
            INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

            KvScatter scatter = new KvScatter(staticBuf, present, cachePos);
            Nd4j.getExecutioner().exec(scatter);

            // Verify each head was scattered correctly
            for (int h = 0; h < heads; h++) {
                INDArray expected = present.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(srcSeqLen - 1), NDArrayIndex.all()).dup();
                INDArray actual = staticBuf.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();

                double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
                assertTrue(maxDiff < 1e-5,
                        String.format("heads=%d, h=%d: maxDiff=%e", heads, h, maxDiff));
                expected.close();
                actual.close();
            }

            present.close();
            staticBuf.close();
        }
        log.info("Varying head counts test passed for heads={}", Arrays.toString(headCounts));
    }

    @Test
    @DisplayName("KV scatter with large dim (SmolDocling-like: heads=4, dim=768)")
    public void testLargeDimSmolDoclingLike() {
        // SmolDocling uses heads=4, dim=768 for some layers
        int batch = 1, heads = 4, maxKvLen = 1024, dim = 768, srcSeqLen = 1;
        long cachePos = 679;  // typical prefill length

        INDArray present = Nd4j.randn(DataType.HALF, batch, heads, srcSeqLen, dim);
        INDArray staticBuf = Nd4j.zeros(DataType.HALF, batch, heads, maxKvLen, dim);

        KvScatter scatter = new KvScatter(staticBuf, present, cachePos);
        Nd4j.getExecutioner().exec(scatter);

        INDArray expected = present.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(0), NDArrayIndex.all()).dup().castTo(DataType.FLOAT);
        INDArray actual = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup().castTo(DataType.FLOAT);

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Large dim FP16 scatter maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-2, "Large dim FP16 scatter maxDiff=" + maxDiff);

        // Verify data integrity: no NaN/Inf
        assertEquals(0, Nd4j.getExecutioner().exec(new org.nd4j.linalg.api.ops.impl.reduce.longer.CountNonZero(
                actual.isNaN())).getInt(0), "Should have no NaN");

        present.close();
        staticBuf.close();
    }

    @Test
    @DisplayName("KV scatter idempotency: double scatter to same position")
    public void testDoubleScatterSamePosition() {
        int batch = 1, heads = 4, maxKvLen = 20, dim = 32;
        long cachePos = 5;

        INDArray present1 = Nd4j.ones(DataType.FLOAT, batch, heads, 1, dim).muli(1.0f);
        INDArray present2 = Nd4j.ones(DataType.FLOAT, batch, heads, 1, dim).muli(2.0f);
        INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

        // First scatter
        Nd4j.getExecutioner().exec(new KvScatter(staticBuf, present1, cachePos));
        INDArray after1 = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();
        double val1 = after1.getFloat(0);
        assertEquals(1.0f, val1, 1e-5, "First scatter should write 1.0");

        // Second scatter to same position should overwrite
        Nd4j.getExecutioner().exec(new KvScatter(staticBuf, present2, cachePos));
        INDArray after2 = staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(cachePos), NDArrayIndex.all()).dup();
        double val2 = after2.getFloat(0);
        assertEquals(2.0f, val2, 1e-5, "Second scatter should overwrite to 2.0");

        present1.close();
        present2.close();
        staticBuf.close();
        after1.close();
        after2.close();
    }
}
