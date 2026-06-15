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
package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.ExpandDims;
import org.nd4j.linalg.api.ops.impl.shape.Squeeze;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for squeeze/expand_dims cancellation optimizations.
 */
@Tag(TagNames.SAMEDIFF)
public class TestSqueezeExpandElimination {

    /**
     * squeeze(expand_dims(x, 0), 0) should cancel to just x.
     */
    @Test
    void testSqueezeExpandDims_Cancelled() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 0);  // [3,4] -> [1,3,4]
        SDVariable squeezed = sd.squeeze("output", expanded, 0); // [1,3,4] -> [3,4]
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Both ops should be eliminated
        assertFalse(hasOpOfType(optimized, Squeeze.class),
                "Squeeze should be eliminated");
        assertFalse(hasOpOfType(optimized, ExpandDims.class),
                "ExpandDims should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        assertEquals("x", actualOutput, "Output should be the input placeholder");

        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "squeeze(expand_dims) cancellation diff = " + diff);
    }

    /**
     * expand_dims(squeeze(x, 2), 2) should cancel to just x (when dim 2 is size 1).
     */
    @Test
    void testExpandDimsSqueeze_Cancelled() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4, 1);
        SDVariable squeezed = sd.squeeze("squeezed", x, 2);      // [3,4,1] -> [3,4]
        SDVariable expanded = sd.expandDims("output", squeezed, 2); // [3,4] -> [3,4,1]
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4, 1);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Squeeze.class),
                "Squeeze should be eliminated");
        assertFalse(hasOpOfType(optimized, ExpandDims.class),
                "ExpandDims should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "expand_dims(squeeze) cancellation diff = " + diff);
    }

    /**
     * squeeze(expand_dims(x, 1), 0) — different axes — should NOT cancel.
     */
    @Test
    void testDifferentAxes_NotCancelled() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("expanded", x, 1);   // [3,4] -> [3,1,4]
        SDVariable squeezed = sd.squeeze("output", expanded, 0);  // This would fail at runtime,
        // but the optimizer should NOT cancel because axes differ
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Axes don't match — both ops should remain
        assertTrue(hasOpOfType(optimized, Squeeze.class) || hasOpOfType(optimized, ExpandDims.class),
                "Different-axis squeeze/expand should not cancel");
    }

    /**
     * A single expand_dims without a matching squeeze should be preserved.
     */
    @Test
    void testSingleExpandDims_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable expanded = sd.expandDims("output", x, 0);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, ExpandDims.class),
                "Single expand_dims should be preserved");
    }

    /**
     * A single squeeze without a matching expand_dims should be preserved.
     */
    @Test
    void testSingleSqueeze_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 1, 4);
        SDVariable squeezed = sd.squeeze("output", x, 1);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, Squeeze.class),
                "Single squeeze should be preserved");
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private boolean hasOpOfType(SameDiff sd, Class<?> opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) return true;
        }
        return false;
    }
}
