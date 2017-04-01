/*-
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
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cache.TADManager;

import java.util.List;
import java.util.Properties;

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

    enum ProfilingMode {
        DISABLED, NAN_PANIC, INF_PANIC, ANY_PANIC, OPERATIONS, METHODS, ALL
    }


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

    /**Execute a TransformOp and return the result
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
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    Accumulation execAndReturn(Variance op, boolean biasCorrected);

    /**Execute and return the result from an index accumulation
     * @param op the index accumulation operation to execute
     * @return the accumulated index
     */
    IndexAccumulation execAndReturn(IndexAccumulation op);

    /**Execute and return the result from a scalar op
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    INDArray execAndReturn(ScalarOp op);

    /** Execute and return the result from a vector op
     * @param op*/
    INDArray execAndReturn(BroadcastOp op);


    /**Execute the operation along 1 or more dimensions
     *
     * @param op the operation to execute
     */
    Op exec(Op op, int... dimension);


    /**
     * Execute an accumulation along one or more dimensions
     * @param accumulation the accumulation
     * @param dimension the dimension
     * @return the accumulation op
     */
    INDArray exec(Accumulation accumulation, int... dimension);

    /**
     * Execute an broadcast along one or more dimensions
     * @param broadcast the accumulation
     * @param dimension the dimension
     * @return the broadcast op
     */
    INDArray exec(BroadcastOp broadcast, int... dimension);

    /**
     * Execute an accumulation along one or more dimensions
     * @param accumulation the accumulation
     * @param dimension the dimension
     * @return the accmulation op
     */
    INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension);


    /** Execute an index accumulation along one or more dimensions
     * @param indexAccum the index accumulation operation
     * @param dimension the dimension/s to execute along
     * @return result
     */
    INDArray exec(IndexAccumulation indexAccum, int... dimension);



    /**Execute and return  a result
     * ndarray from the given op
     * @param op the operation to execute
     * @return the result from the operation
     */
    INDArray execAndReturn(Op op);


    /**Get the execution mode for this
     * execuioner
     * @return the execution mode for this executioner
     */
    ExecutionMode executionMode();

    /**Set the execution mode
     * @param executionMode the execution mode
     */
    void setExecutionMode(ExecutionMode executionMode);

    /**
     * Execute MetaOp
     *
     * @param op
     */
    void exec(MetaOp op);

    /**
     * Execute GridOp
     * @param op
     */
    void exec(GridOp op);


    void exec(Aggregate op);

    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    <T extends Aggregate> void exec(Batch<T> batch);

    /**
     * This method takes abritrary sized list of aggregates, and packs them into batches
     *
     * @param batch
     */
    void exec(List<Aggregate> batch);

    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    INDArray exec(RandomOp op);

    /**
     * This method executes specific RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    INDArray exec(RandomOp op, Random rng);

    /**
     * This method return set of key/value and
     * key/key/value objects,
     * describing current environment
     *
     * @return
     */
    Properties getEnvironmentInformation();

    /**
     * This method specifies desired profiling mode
     *
     * @param mode
     */
    void setProfilingMode(ProfilingMode mode);

    /**
     * Ths method returns current profiling
     *
     * @return
     */
    ProfilingMode getProfilingMode();


    /**
     * This method returns TADManager instance used for this OpExecutioner
     *
     * @return
     */
    TADManager getTADManager();


    void printEnvironmentInformation();

}
