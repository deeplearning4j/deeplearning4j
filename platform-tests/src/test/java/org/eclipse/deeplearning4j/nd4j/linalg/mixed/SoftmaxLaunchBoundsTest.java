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
package org.eclipse.deeplearning4j.nd4j.linalg.mixed;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Softmax kernel launch bounds regression test.
 *
 * The softMaxCuda kernel uses __launch_bounds__(256, 2), but getSoftmaxDims()
 * could request 512 threads when tadLen >= 512 (BLOCK_SIZE_SOFTMAX_LARGE).
 * This causes CUDA error 1 "invalid argument" at launch time because the
 * register allocation exceeds what NVCC reserved for 256 threads per block.
 *
 * This test exercises softmax on attention-sized tensors [B, H, S, S] along
 * axis -1 where S >= 512, ensuring the kernel launches without error across
 * all DSP execution modes and data types.
 *
 * Shapes use batch=1, heads=4 to keep memory usage reasonable (~100MB per test).
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class SoftmaxLaunchBoundsTest {

    static Stream<Arguments> softmaxShapes() {
        return Stream.of(
                // [batch, heads, seqLen, seqLen] — attention pattern, tadLen = seqLen
                // Using heads=4 to keep memory manageable
                Arguments.of("attn_512", new long[]{1, 4, 512, 512}, -1, DataType.FLOAT),
                Arguments.of("attn_1024", new long[]{1, 4, 1024, 1024}, -1, DataType.FLOAT),
                // FP16 paths
                Arguments.of("attn_1024_fp16", new long[]{1, 4, 1024, 1024}, -1, DataType.FLOAT16),
                Arguments.of("attn_512_fp16", new long[]{1, 4, 512, 512}, -1, DataType.FLOAT16),
                // Smaller tadLen (should use 256 threads — baseline)
                Arguments.of("attn_256", new long[]{1, 4, 256, 256}, -1, DataType.FLOAT),
                // Non-attention shapes with large last dim
                Arguments.of("logits_50k", new long[]{1, 1, 49280}, -1, DataType.FLOAT),
                Arguments.of("large_vocab_fp16", new long[]{1, 1, 49280}, -1, DataType.FLOAT16)
        );
    }

    @Order(1)
    @ParameterizedTest(name = "{0}")
    @MethodSource("softmaxShapes")
    @DisplayName("Softmax with tadLen >= 512 must not crash (launch bounds)")
    public void testSoftmaxLaunchBounds(String name, long[] shape, int axis, DataType dtype) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", dtype, shape);
        SDVariable output = sd.nn.softmax("softmax_out", input, axis);

        INDArray inputData = Nd4j.randn(dtype, shape).muli(2.0);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", inputData);

        Map<String, INDArray> result = sd.output(placeholders, "softmax_out");

        INDArray softmaxOut = result.get("softmax_out");
        assertNotNull(softmaxOut, name + ": softmax output is null");
        assertArrayEquals(shape, softmaxOut.shape(), name + ": output shape mismatch");

        // Softmax values should sum to ~1.0 along the softmax axis
        int normalizedAxis = axis < 0 ? axis + shape.length : axis;
        INDArray sums = softmaxOut.sum(normalizedAxis);
        double maxDeviation = sums.sub(1.0).amaxNumber().doubleValue();
        assertTrue(maxDeviation < 0.01,
                name + ": softmax sum deviates from 1.0 by " + maxDeviation);

        // No NaN or Inf
        assertFalse(softmaxOut.isNaN().any(),
                name + ": softmax output contains NaN");
        assertFalse(softmaxOut.isInfinite().any(),
                name + ": softmax output contains Inf");

        log.info("[{}] shape={} dtype={} maxSumDeviation={} — PASS",
                name, java.util.Arrays.toString(shape), dtype, maxDeviation);

        inputData.close();
        for (INDArray arr : result.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }

    static Stream<Arguments> dspModes() {
        return Stream.of(
                Arguments.of(GraphExecutionMode.SLOT_BY_SLOT),
                Arguments.of(GraphExecutionMode.AUTO)
        );
    }

    @Order(2)
    @ParameterizedTest(name = "dsp_{0}")
    @MethodSource("dspModes")
    @DisplayName("Softmax via DSP execution on attention-sized input")
    public void testSoftmaxViaDsp(GraphExecutionMode mode) {
        long[] shape = new long[]{1, 4, 512, 512};
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, shape);
        SDVariable output = sd.nn.softmax("softmax_out", input, -1);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(mode);

        INDArray inputData = Nd4j.randn(DataType.FLOAT, shape).muli(2.0);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", inputData);

        Map<String, INDArray> result = sd.output(placeholders, "softmax_out");

        INDArray softmaxOut = result.get("softmax_out");
        assertNotNull(softmaxOut, mode + ": softmax output is null");
        assertFalse(softmaxOut.isNaN().any(), mode + ": softmax output contains NaN");

        // Verify sum along axis -1 is ~1.0
        INDArray sums = softmaxOut.sum(-1);
        double maxDev = sums.sub(1.0).amaxNumber().doubleValue();
        assertTrue(maxDev < 0.01, mode + ": softmax sum deviates from 1.0 by " + maxDev);

        log.info("[dsp_{}] shape={} maxSumDev={} — PASS",
                mode, java.util.Arrays.toString(shape), maxDev);

        inputData.close();
        for (INDArray arr : result.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
    }
}
