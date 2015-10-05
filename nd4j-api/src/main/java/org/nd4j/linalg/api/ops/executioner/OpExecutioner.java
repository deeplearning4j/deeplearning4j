/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.executioner;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.parallel.ParallelExecutioner;

/**
 * An operation executioner handles storage specific details of
 * executing an operation
 *
 * @author Adam Gibson
 */
public interface OpExecutioner {

    enum ExecutionMode {
        JAVA, NATIVE
    }

    ParallelExecutioner parallelExecutioner();

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    Op exec(Op op);

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    void iterateOverAllRows(Op op);

    /**
     * Iterate over every column of every slice
     *
     * @param op the operation to apply
     */
    void iterateOverAllColumns(Op op);

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    INDArray execAndReturn(TransformOp op);


    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    Accumulation execAndReturn(Accumulation op);

    /**
     * Execute and return the result from an index accumulation
     * @param op the operation to execute
     * @return the accumulated index
     */
    IndexAccumulation execAndReturn(IndexAccumulation op);

    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    INDArray execAndReturn(ScalarOp op);


    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    Op exec(Op op, int...dimension);


    /**
     * Execute an accumulation along a dimension
     * @param accumulation the accumulation
     * @param dimension the dimension
     * @return the accmulation op
     */
    INDArray exec(Accumulation accumulation, int...dimension);

    /** Execute and index accumulation along a dimension
     * @param indexAccum the index accumulation operation
     * @param dimension the dimension/s to execute along
     * @return result
     */
    INDArray exec(IndexAccumulation indexAccum, int... dimension);

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    INDArray execAndReturn(TransformOp op, int...dimension);



    /**
     * Execute and return  a result
     * ndarray from the given op
     * @param op the operation to execute
     * @return the result from the operation
     */
    INDArray execAndReturn(Op op);

    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    INDArray execAndReturn(ScalarOp op, int... dimension);


    /**
     * Get the execution mode for this
     * execuioner
     * @return the execution mode for this executioner
     */
    ExecutionMode executionMode();

    /**
     * Set the execution mode
     * @param executionMode the execution mode
     */
    void setExecutionMode(ExecutionMode executionMode);

}
