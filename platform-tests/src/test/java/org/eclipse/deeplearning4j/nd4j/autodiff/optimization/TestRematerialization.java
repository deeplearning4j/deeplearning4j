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
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for rematerialization optimization: duplicating cheap ops to reduce live ranges.
 *
 * When a cheap unary op's output feeds two consumers, rematerialization creates a copy
 * so each consumer has its own instance. This allows earlier buffer release.
 */
@Tag(TagNames.SAMEDIFF)
public class TestRematerialization {

    /**
     * exp(x) feeds both add and mul — rematerialization should duplicate exp
     * so each consumer has its own copy.
     */
    @Test
    void testCheapUnaryDuplicated() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable e = sd.math().exp("exp_out", x);
        // Two consumers of exp_out
        SDVariable out1 = e.add("add_out", sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4)));
        SDVariable out2 = e.mul("mul_out", sd.constant("scale", Nd4j.valueArrayOf(new long[]{4}, 2.0, DataType.FLOAT)));
        SDVariable result = out1.add("result", out2);
        sd.setOutputs("result");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4);
        INDArray refOut = sd.output(Map.of("x", input), "result").get("result").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        // After rematerialization, there should be 2 exp ops instead of 1
        int expCount = countOpsOfType(optimized, Exp.class);
        assertEquals(2, expCount, "Should have 2 exp ops after rematerialization (got " + expCount + ")");

        // Numerical correctness must be preserved
        INDArray optOut = optimized.output(Map.of("x", input), "result").get("result");
        double diff = refOut.sub(optOut).amaxNumber().doubleValue();
        assertTrue(diff < 1e-5, "Rematerialization changed results, max diff = " + diff);
    }

    /**
     * exp(x) feeds only one consumer — should NOT be duplicated.
     */
    @Test
    void testSingleConsumer_NoDuplication() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable e = sd.math().exp("exp_out", x);
        SDVariable result = e.add("result", sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4)));
        sd.setOutputs("result");

        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        // Single consumer — should stay as 1 exp
        int expCount = countOpsOfType(optimized, Exp.class);
        assertEquals(1, expCount, "Single-consumer exp should not be duplicated");
    }

    /**
     * Three consumers — too many, should NOT be duplicated (conservative limit = 2).
     */
    @Test
    void testThreeConsumers_NoDuplication() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable e = sd.math().exp("exp_out", x);
        SDVariable out1 = e.add("add_out", sd.constant("b1", Nd4j.ones(DataType.FLOAT, 4)));
        SDVariable out2 = e.mul("mul_out", sd.constant("s1", Nd4j.valueArrayOf(new long[]{4}, 2.0, DataType.FLOAT)));
        SDVariable out3 = e.sub("sub_out", sd.constant("b2", Nd4j.valueArrayOf(new long[]{4}, 0.5, DataType.FLOAT)));
        SDVariable result = out1.add("r1", out2).add("result", out3);
        sd.setOutputs("result");

        SameDiff optimized = GraphOptimizer.optimize(sd, "result");

        // 3 consumers — conservative limit, should NOT duplicate
        int expCount = countOpsOfType(optimized, Exp.class);
        assertEquals(1, expCount, "3-consumer exp should not be duplicated (conservative limit)");
    }

    /**
     * Graph output cannot be rematerialized.
     */
    @Test
    void testGraphOutput_NoDuplication() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable e = sd.math().exp("exp_out", x);
        SDVariable out1 = e.add("add_out", sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4)));
        // exp_out is both a graph output AND feeds add_out
        sd.setOutputs("exp_out", "add_out");

        SameDiff optimized = GraphOptimizer.optimize(sd, "exp_out", "add_out");

        // Graph outputs can't be rematerialized
        int expCount = countOpsOfType(optimized, Exp.class);
        assertEquals(1, expCount, "Graph output exp should not be duplicated");
    }

    private int countOpsOfType(SameDiff sd, Class<?> opType) {
        int count = 0;
        for (SameDiffOp op : sd.getOps().values()) {
            if (opType.isInstance(op.getOp())) count++;
        }
        return count;
    }
}
