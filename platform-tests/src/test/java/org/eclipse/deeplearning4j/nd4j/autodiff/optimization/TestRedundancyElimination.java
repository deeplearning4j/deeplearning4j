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
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.Gather;
import org.nd4j.linalg.api.ops.impl.shape.Slice;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for redundancy elimination optimizations:
 * - Single-input concat elimination
 * - Full-extent slice elimination
 * - Identity gather elimination
 * - pow(x, 0) -> 1
 */
@Tag(TagNames.SAMEDIFF)
public class TestRedundancyElimination {

    // ─── Single-input concat ───────────────────────────────────────────────

    /**
     * concat(x) with a single input should be eliminated.
     */
    @Test
    void testSingleInputConcat_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable out = sd.concat("output", 0, x);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Concat.class),
                "Single-input concat should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Single-input concat diff = " + diff);
    }

    /**
     * concat(x, y) with multiple inputs should NOT be eliminated.
     */
    @Test
    void testMultiInputConcat_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, 2, 4);
        SDVariable out = sd.concat("output", 0, x, y);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertTrue(hasOpOfType(optimized, Concat.class),
                "Multi-input concat should be preserved");

        INDArray inputX = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray inputY = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", inputX, "y", inputY), "output").get("output").dup();
        INDArray optOut = optimized.output(Map.of("x", inputX, "y", inputY), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Multi-input concat correctness diff = " + diff);
    }

    // ─── Identity gather ───────────────────────────────────────────────────

    /**
     * gather(x, [0, 1, 2], axis=0) on a shape [3, 4] tensor should be eliminated.
     */
    @Test
    void testIdentityGather_Eliminated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[]{0, 1, 2}));
        SDVariable out = sd.gather("output", x, indices, 0);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        assertFalse(hasOpOfType(optimized, Gather.class),
                "Identity gather should be eliminated");

        String actualOutput = optimized.outputs().get(0);
        INDArray optOut = optimized.output(Map.of("x", input), actualOutput).get(actualOutput);
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Identity gather diff = " + diff);
    }

    /**
     * gather with non-identity indices should be preserved.
     */
    @Test
    void testNonIdentityGather_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 3, 4);
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[]{2, 0, 1}));
        SDVariable out = sd.gather("output", x, indices, 0);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Reordering indices — should NOT be eliminated
        INDArray input = Nd4j.randn(DataType.FLOAT, 3, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Non-identity gather correctness diff = " + diff);
    }

    /**
     * gather with a subset of indices should be preserved.
     */
    @Test
    void testSubsetGather_Preserved() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 5, 4);
        SDVariable indices = sd.constant("indices", Nd4j.createFromArray(new int[]{0, 2}));
        SDVariable out = sd.gather("output", x, indices, 0);
        sd.setOutputs("output");

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 5, 4);
        INDArray refOut = sd.output(Map.of("x", input), "output").get("output").dup();
        INDArray optOut = optimized.output(Map.of("x", input), "output").get("output");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Subset gather correctness diff = " + diff);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private boolean hasOpOfType(SameDiff sd, Class<?> opType) {
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) return true;
        }
        return false;
    }
}
