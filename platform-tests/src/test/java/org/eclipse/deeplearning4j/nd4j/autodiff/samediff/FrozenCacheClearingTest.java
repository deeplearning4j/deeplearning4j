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
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that cache clearing on freeze/unfreeze transitions in DynamicShapePlanExecutor
 * does not break execution correctness.
 *
 * Background: setShapesFrozen() now clears caches on BOTH transitions (freeze and
 * unfreeze). The old code only cleared on unfreeze. If clearing on freeze destroys
 * state that the first frozen execution needs, results would be wrong.
 *
 * Test approach: Build a SameDiff graph, compile DSP, run it to populate caches,
 * then test freeze/unfreeze transitions and verify correctness.
 */
@DisplayName("Frozen cache clearing on setShapesFrozen transition test")
public class FrozenCacheClearingTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(FrozenCacheClearingTest.class);
    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    @DisplayName("Results correct after freeze transition")
    public void testCorrectAfterFreezeTransition() {
        int dim = 16;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, dim, dim).mul(0.01));
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, dim));
        SDVariable mm = sd.mmul("mm", input, weights);
        SDVariable out = mm.add("output", bias);

        // Run several times to populate caches (unfrozen)
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, dim);
        Map<String, INDArray> ph = Collections.singletonMap("input", inputArr.dup());
        Map<String, INDArray> baselineResult = null;
        for (int i = 0; i < 3; i++) {
            ph = Collections.singletonMap("input", inputArr.dup());
            baselineResult = sd.output(ph, "output");
        }
        INDArray baselineOutput = baselineResult.get("output").dup();
        log.info("Baseline output first value: {}", baselineOutput.getFloat(0));

        // Now get the DSP executor and freeze shapes
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("Shapes frozen");

            // Run with frozen shapes - should still be correct
            for (int i = 0; i < 5; i++) {
                ph = Collections.singletonMap("input", inputArr.dup());
                Map<String, INDArray> frozenResult = sd.outputDirect(ph, "output");
                INDArray frozenOutput = frozenResult.get("output");

                double maxDiff = frozenOutput.sub(baselineOutput).amaxNumber().doubleValue();
                log.info("Frozen step {}: first={}, baseline={}, maxDiff={}",
                        i, frozenOutput.getFloat(0), baselineOutput.getFloat(0), maxDiff);
                assertTrue(maxDiff < TOL,
                        "Frozen step " + i + ": output diverges from baseline. maxDiff=" + maxDiff);
            }

            // Unfreeze
            dspExec.setShapesFrozen(false);
            log.info("Shapes unfrozen");

            // Run after unfreeze - should still be correct
            for (int i = 0; i < 3; i++) {
                ph = Collections.singletonMap("input", inputArr.dup());
                Map<String, INDArray> unfrozenResult = sd.output(ph, "output");
                INDArray unfrozenOutput = unfrozenResult.get("output");

                double maxDiff = unfrozenOutput.sub(baselineOutput).amaxNumber().doubleValue();
                log.info("Unfrozen step {}: first={}, baseline={}, maxDiff={}",
                        i, unfrozenOutput.getFloat(0), baselineOutput.getFloat(0), maxDiff);
                assertTrue(maxDiff < TOL,
                        "Unfrozen step " + i + ": output diverges from baseline. maxDiff=" + maxDiff);
            }
        } else {
            log.info("No DSP executor available, testing standard path only");
            // Even without DSP, verify repeated execution is correct
            for (int i = 0; i < 5; i++) {
                ph = Collections.singletonMap("input", inputArr.dup());
                Map<String, INDArray> result = sd.output(ph, "output");
                INDArray output = result.get("output");
                double maxDiff = output.sub(baselineOutput).amaxNumber().doubleValue();
                assertTrue(maxDiff < TOL, "Standard step " + i + " diverges. maxDiff=" + maxDiff);
            }
        }
    }

    @Test
    @DisplayName("Rapid freeze/unfreeze cycling does not corrupt state")
    public void testRapidFreezeUnfreezeCycling() {
        int dim = 16;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, dim);
        SDVariable weights = sd.var("weights", Nd4j.eye(dim).castTo(DataType.FLOAT));
        SDVariable out = sd.mmul("output", input, weights);

        // Warm up
        INDArray inputArr = Nd4j.ones(DataType.FLOAT, 1, dim).mul(7.0f);
        Map<String, INDArray> ph = Collections.singletonMap("input", inputArr.dup());
        sd.output(ph, "output");
        sd.output(ph, "output");

        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();

        for (int cycle = 0; cycle < 5; cycle++) {
            // Freeze
            if (dspExec != null) {
                dspExec.setShapesFrozen(true);
            }

            // Run frozen
            for (int step = 0; step < 3; step++) {
                ph = Collections.singletonMap("input", inputArr.dup());
                Map<String, INDArray> result = sd.outputDirect(ph, "output");
                INDArray output = result.get("output");
                double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
                log.info("Cycle {} frozen step {}: maxDiff={}", cycle, step, maxDiff);
                assertTrue(maxDiff < TOL,
                        "Cycle " + cycle + " frozen step " + step + ": wrong result. maxDiff=" + maxDiff);
            }

            // Unfreeze
            if (dspExec != null) {
                dspExec.setShapesFrozen(false);
            }

            // Run unfrozen
            for (int step = 0; step < 3; step++) {
                ph = Collections.singletonMap("input", inputArr.dup());
                Map<String, INDArray> result = sd.output(ph, "output");
                INDArray output = result.get("output");
                double maxDiff = output.sub(inputArr).amaxNumber().doubleValue();
                log.info("Cycle {} unfrozen step {}: maxDiff={}", cycle, step, maxDiff);
                assertTrue(maxDiff < TOL,
                        "Cycle " + cycle + " unfrozen step " + step + ": wrong result. maxDiff=" + maxDiff);
            }
        }
    }

    @Test
    @DisplayName("Freeze with different input shapes then unfreeze")
    public void testFreezeWithDifferentShapesThenUnfreeze() {
        // Run unfrozen with shape [1, 16], freeze, run with [1, 16], unfreeze,
        // run with [2, 16] (different batch size). The cache clear on unfreeze
        // should allow the new shape to work correctly.
        int dim = 16;

        SameDiff sd = SameDiff.create();
        sd.setDspAutoCompileEnabled(true);
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, dim);
        SDVariable scale = sd.var("scale", Nd4j.ones(DataType.FLOAT, dim).mul(3.0));
        SDVariable out = input.mul("output", scale);

        // Warm up with batch=1
        INDArray input1 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(2.0f);
        Map<String, INDArray> ph = Collections.singletonMap("input", input1);
        Map<String, INDArray> result = sd.output(ph, "output");
        INDArray expected1 = Nd4j.ones(DataType.FLOAT, 1, dim).mul(6.0f); // 2.0 * 3.0
        double diff1 = result.get("output").sub(expected1).amaxNumber().doubleValue();
        log.info("Warm-up batch=1: maxDiff={}", diff1);
        assertTrue(diff1 < TOL, "Warm-up wrong");

        // Freeze
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
        }

        // Run frozen with same shape
        ph = Collections.singletonMap("input", input1.dup());
        result = sd.outputDirect(ph, "output");
        double diffFrozen = result.get("output").sub(expected1).amaxNumber().doubleValue();
        log.info("Frozen batch=1: maxDiff={}", diffFrozen);
        assertTrue(diffFrozen < TOL, "Frozen wrong");

        // Unfreeze
        if (dspExec != null) {
            dspExec.setShapesFrozen(false);
        }

        // Run with different batch size
        INDArray input2 = Nd4j.ones(DataType.FLOAT, 2, dim).mul(5.0f);
        INDArray expected2 = Nd4j.ones(DataType.FLOAT, 2, dim).mul(15.0f); // 5.0 * 3.0
        ph = Collections.singletonMap("input", input2);
        result = sd.output(ph, "output");
        INDArray output2 = result.get("output");

        assertArrayEquals(new long[]{2, dim}, output2.shape(),
                "Output shape wrong after unfreeze with different batch");
        double diff2 = output2.sub(expected2).amaxNumber().doubleValue();
        log.info("Unfrozen batch=2: shape={}, maxDiff={}", java.util.Arrays.toString(output2.shape()), diff2);
        assertTrue(diff2 < TOL, "Unfrozen with new shape wrong. maxDiff=" + diff2);
    }
}
