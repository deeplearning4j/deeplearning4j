/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.ControlFlow;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The condition ControlFlow.loopWithConditions evaluates before each iteration: the iteration
 * count below the maximum, combined with the extra condition through a bitwise and. When that and
 * ran wrong on CUDA the condition was false before the first iteration, so every such loop
 * returned its inputs unchanged (SameDiffTests#testLooping, the SDVariable-indexed get and put).
 */
class ControlFlowConditionTest {

    @Test
    void conditionHoldsBelowTheMaximum() {
        assertEquals(Nd4j.createFromArray(true), condition(0, 5, true));
        assertEquals(Nd4j.createFromArray(true), condition(4, 5, true));
    }

    @Test
    void conditionFailsAtTheMaximumOrWithoutTheExtraCondition() {
        assertEquals(Nd4j.createFromArray(false), condition(5, 5, true));
        assertEquals(Nd4j.createFromArray(false), condition(0, 5, false));
    }

    private static INDArray condition(int iteration, int maxIterations, boolean extraCondition) {
        SameDiff sd = SameDiff.create();
        SDVariable[] args = ControlFlow.initializeLoopBody(
                new String[]{"curr_iteration", "max_iterations", "cond_in"}, sd, maxIterations, extraCondition);
        args[0].setArray(Nd4j.createFromArray((float) iteration));
        INDArray value = ControlFlow.condBody().define(sd, args).eval();
        assertEquals(DataType.BOOL, value.dataType());
        return value;
    }
}
