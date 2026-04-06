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
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for stale decode input detection via DspDebugger.
 *
 * Reproduces the VLM decoder bug where position_ids stays at 679 on every
 * step because CUDA graph capture buffers aren't refreshed with new values.
 * Uses DspDebugger.validateDecodeInputProgression() which throws a hard
 * error when stale values are detected.
 */
@Slf4j
@DisplayName("DSP Decode Input Staleness Detection")
public class DspDecodeInputStalenessTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Simple graph where position_ids feeds into a gather + add.
     * The output should change when position_ids changes.
     * If position_ids is stale, output stays the same → detected.
     *
     * Graph: embeddings[100, 8] → gather(position_ids) → add(1) → sum → out
     *
     * position_ids=0 → gather row 0 → sum ≠ sum at position_ids=1
     */
    @Test
    @DisplayName("Position ids must change output across steps")
    public void testPositionIdsMustChangeOutput() {
        SameDiff sd = SameDiff.create();

        // Simulated embedding table: 100 positions, hidden_size=8
        INDArray embeddingTable = Nd4j.randn(DataType.FLOAT, 100, 8).muli(10);
        SDVariable embeddings = sd.constant("embeddings", embeddingTable);
        SDVariable posIds = sd.placeHolder("position_ids", DataType.INT64, 1);
        SDVariable gathered = sd.gather("pos_embed", embeddings, posIds, 0);
        SDVariable out = gathered.sum("out");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Run 5 steps with incrementing position_ids
        // Each step should produce a different output (different embedding row)
        INDArray posIdsArr = Nd4j.createFromArray(0L);
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("position_ids", posIdsArr);

        Double lastSum = null;
        for (int step = 0; step < 5; step++) {
            posIdsArr.putScalar(0, (long) step);
            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            log.info("Step {}: position_ids={}, sum={}", step, step, sum);

            if (lastSum != null) {
                assertNotEquals(lastSum, sum, 1e-6,
                        "Step " + step + ": output should differ from step " + (step - 1) +
                        " because position_ids changed. If identical, position_ids is stale.");
            }
            lastSum = sum;
        }
    }

    /**
     * Test with attention mask that grows each step.
     * The graph applies mask to values and sums — output should grow.
     */
    @Test
    @DisplayName("Attention mask sum must grow across steps")
    public void testAttentionMaskMustGrow() {
        SameDiff sd = SameDiff.create();

        SDVariable values = sd.placeHolder("values", DataType.FLOAT, 1, 10);
        SDVariable mask = sd.placeHolder("attention_mask", DataType.INT64, 1, 10);
        SDVariable maskFloat = sd.castTo("mask_float", mask, DataType.FLOAT);
        SDVariable masked = values.mul("masked", maskFloat);
        SDVariable out = masked.sum("out");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray valuesArr = Nd4j.ones(DataType.FLOAT, 1, 10);
        INDArray maskArr = Nd4j.zeros(DataType.INT64, 1, 10);

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("values", valuesArr);
        placeholders.put("attention_mask", maskArr);

        for (int step = 0; step < 5; step++) {
            // Grow mask: add one more 1 per step
            maskArr.putScalar(0, step, 1L);

            Map<String, INDArray> result = sd.output(placeholders, "out");
            double sum = result.get("out").getDouble(0);
            double expectedSum = step + 1;  // step 0: 1 one, step 1: 2 ones, etc.
            log.info("Step {}: mask sum={}, output sum={}, expected={}", step,
                    maskArr.sumNumber().longValue(), sum, expectedSum);

            assertEquals(expectedSum, sum, 1e-4,
                    "Step " + step + ": masked sum should be " + expectedSum +
                    ". If wrong, attention_mask is stale.");
        }
    }

    /**
     * Uses DspDebugger.validateDecodeInputProgression to detect stale values.
     * This is the framework method that should throw on stale inputs.
     */
    @Test
    @DisplayName("DspDebugger detects stale decode inputs")
    public void testDspDebuggerDetectsStaleness() {
        SameDiff sd = SameDiff.create();

        // Simple model: position_ids → gather from table → sum
        INDArray table = Nd4j.randn(DataType.FLOAT, 50, 4).muli(10);
        SDVariable embeddings = sd.constant("embeddings", table);
        SDVariable posIds = sd.placeHolder("position_ids", DataType.INT64, 1);
        SDVariable mask = sd.placeHolder("attention_mask", DataType.INT64, 1, 50);
        SDVariable gathered = sd.gather("pos_embed", embeddings, posIds, 0);
        SDVariable maskFloat = sd.castTo("mask_float", mask, DataType.FLOAT);
        SDVariable out = gathered.sum("out");

        DspDebugger debugger = DspDebugger.attach(sd);

        INDArray posIdsArr = Nd4j.createFromArray(0L);
        INDArray maskArr = Nd4j.zeros(DataType.INT64, 1, 50);
        maskArr.putScalar(0, 0, 1L);  // Start with 1 active position

        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("position_ids", posIdsArr);
        placeholders.put("attention_mask", maskArr);

        // This should NOT throw — values change correctly when DSP is working
        try {
            DspDebugger.DecodeInputReport report = debugger.validateDecodeInputProgression(
                    placeholders, "position_ids", "attention_mask", 5, "out");
            log.info("Decode input validation report:\n{}", report);
            assertFalse(report.hasErrors(), "Should have no stale value errors: " + report);
        } catch (IllegalStateException e) {
            // If this throws, the capture buffer refresh is broken
            log.error("STALE VALUES DETECTED: {}", e.getMessage());
            fail("Stale decode inputs detected — capture buffer refresh broken: " + e.getMessage());
        }
    }
}
