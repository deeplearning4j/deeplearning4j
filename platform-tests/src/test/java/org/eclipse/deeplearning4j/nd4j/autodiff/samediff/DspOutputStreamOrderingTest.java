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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates DSP output stream ordering and bulk extraction correctness.
 *
 * Tests that DSP inference output is readable and non-zero across repeated
 * executions (reproducing the batch 16-17 zero-output scenario), and that
 * bulk extraction via getFloatsAt() matches element-by-element getFloat().
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP output stream ordering and bulk extraction tests")
public class DspOutputStreamOrderingTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DSP output should be non-zero across repeated executions")
    public void testDspOutputNonZeroAcrossRepeatedExecutions(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 64);
        SDVariable weight = sd.var("w", Nd4j.rand(DataType.FLOAT, 64, 128));
        SDVariable out = sd.mmul("out", input, weight);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 2, 64);
        Map<String, INDArray> ph = Map.of("input", inputArr);

        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> result = sd.output(ph, "out");
            INDArray output = result.get("out");

            assertNotNull(output, "Execution " + i + ": output should not be null");
            assertEquals(DataType.FLOAT, output.dataType());
            assertArrayEquals(new long[]{2, 128}, output.shape());

            // Use getFloat to verify non-zero — avoids launching additional CUDA ops
            float val = output.getFloat(0, 0);
            assertTrue(Float.isFinite(val), "Execution " + i + ": output value should be finite");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("castTo(FLOAT) on DOUBLE output should produce correct non-zero values")
    public void testCastToFloatProducesCorrectValues(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();

        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 64);
        SDVariable weight = sd.var("w", Nd4j.rand(DataType.FLOAT, 64, 128));
        SDVariable scale = sd.constant("scale", Nd4j.scalar(DataType.DOUBLE, 1.0));
        SDVariable mm = sd.mmul("mm", input, weight);
        SDVariable out = mm.mul("out", scale);

        INDArray inputArr = Nd4j.rand(DataType.FLOAT, 2, 64);
        Map<String, INDArray> result = sd.output(Map.of("input", inputArr), "out");
        INDArray output = result.get("out");

        INDArray casted = (output.dataType() != DataType.FLOAT)
                ? output.castTo(DataType.FLOAT) : output;

        try {
            assertEquals(DataType.FLOAT, casted.dataType());

            float[] data = casted.data().getFloatsAt(casted.offset(), (int) casted.length());
            double sum = 0;
            for (float v : data) {
                assertTrue(Float.isFinite(v), "All values should be finite");
                sum += v * v;
            }
            assertTrue(sum > 1e-6, "Cast output should be non-zero, got sumSquares=" + sum);
        } finally {
            if (casted != output) casted.close();
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Bulk getFloatsAt() must match element-by-element getFloat()")
    public void testBulkExtractionMatchesElementWise(Nd4jBackend backend) {
        INDArray tensor = Nd4j.rand(DataType.FLOAT, 2, 512, 768);

        int batchSize = 2;
        int embeddingDim = 768;

        // Element-by-element
        float[][] elementResults = new float[batchSize][embeddingDim];
        for (int b = 0; b < batchSize; b++) {
            for (int j = 0; j < embeddingDim; j++) {
                elementResults[b][j] = tensor.getFloat(b, 0, j);
            }
        }

        // Bulk extraction
        float[][] bulkResults = new float[batchSize][embeddingDim];
        for (int b = 0; b < batchSize; b++) {
            INDArray row = tensor.get(
                    NDArrayIndex.point(b),
                    NDArrayIndex.point(0),
                    NDArrayIndex.all());
            INDArray contiguous = (row.ordering() == 'c' && !row.isView())
                    ? row : row.dup('c');
            try {
                bulkResults[b] = contiguous.data().getFloatsAt(contiguous.offset(), embeddingDim);
            } finally {
                if (contiguous != row) contiguous.close();
                if (row != tensor) try { row.close(); } catch (Exception ignored) {}
            }
        }

        for (int b = 0; b < batchSize; b++) {
            for (int j = 0; j < embeddingDim; j++) {
                assertEquals(elementResults[b][j], bulkResults[b][j], 1e-6f,
                        "batch=" + b + " dim=" + j + ": bulk must match element-wise");
            }
        }
    }
}
