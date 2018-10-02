/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.executioner;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cache.TADManager;

import java.util.List;
import java.util.Map;
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

    // in case of adding new executioner - list it here
    enum ExecutionerType {
        NATIVE_CPU,
        CUDA
    }

    enum ProfilingMode {
        DISABLED,
        NAN_PANIC,
        INF_PANIC,
        ANY_PANIC,
        OPERATIONS,
        METHODS,
        ALL,
        SCOPE_PANIC,
        BANDWIDTH,
    }

    /**
     * This method returns true if verbose mode is enabled, false otherwise
     * @return
     */
    boolean isVerbose();

    /**
     * This method returns true if debug mode is enabled, false otherwise
     * @return
     */
    boolean isDebug();


    /**
     * This method returns type for this executioner instance
     * @return
     */
    ExecutionerType type();


    /**
     * This method returns opName of the last invoked op
     *
     * @return
     */
    String getLastOp();


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
    ReduceOp execAndReturn(ReduceOp op);

    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    ReduceOp execAndReturn(Variance op, boolean biasCorrected);

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

    /** Execute and return the result from a vector op
     * @param op*/
    INDArray execAndReturn(ShapeOp op);


    /**Execute the operation along 1 or more dimensions
     *
     * @param op the operation to execute
     */
    Op exec(Op op, int... dimension);


    /**
     * Execute an reduceOp along one or more dimensions
     * @param reduceOp the reduceOp
     * @param dimension the dimension
     * @return the reduceOp op
     */
    INDArray exec(ReduceOp reduceOp, int... dimension);

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



    /**
     *
     * Execute and return  a result
     * ndarray from the given op
     * @param op the operation to execute
     * @return the result from the operation
     */
    INDArray execAndReturn(Op op);


    /**Get the execution mode for this
     * executioner
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

    /**
     *
     * @param op
     */
    void exec(Aggregate op);

    /**
     *
     * @param op
     */
    void exec(ShapeOp op);

    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    <T extends Aggregate> void exec(Batch<T> batch);

    /**
     * This method takes arbitrary sized list of aggregates,
     * and packs them into batches
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


    /**
     * This method prints out environmental information returned by getEnvironmentInformation() method
     */
    void printEnvironmentInformation();

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed.
     */
    void push();

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed and finished.
     */
    void commit();

    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @return encoded array is returned
     */
    INDArray thresholdEncode(INDArray input, double threshold);


    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @return encoded array is returned
     */
    INDArray thresholdEncode(INDArray input, double threshold, Integer boundary);

    /**
     * This method decodes thresholds array, and puts it into target array
     *
     * @param encoded
     * @param target
     * @return target is returned
     */
    INDArray thresholdDecode(INDArray encoded, INDArray target);

    /**
     * This method returns number of elements affected by encoder
     * @param indArray
     * @param target
     * @param threshold
     * @return
     */
    long bitmapEncode(INDArray indArray, INDArray target, double threshold);

    /**
     *
     * @param indArray
     * @param threshold
     * @return
     */
    INDArray bitmapEncode(INDArray indArray, double threshold);

    /**
     *
     * @param encoded
     * @param target
     * @return
     */
    INDArray bitmapDecode(INDArray encoded, INDArray target);

    /**
     * This method returns names of all custom operations available in current backend, and their number of input/output arguments
     * @return
     */
    Map<String, CustomOpDescriptor> getCustomOperations();

    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * @param op
     */
    void exec(CustomOp op);

    List<long[]> calculateOutputShape(CustomOp op);

    /**
     * Equivalent to calli
     */
    INDArray[] allocateOutputArrays(CustomOp op);


    void enableDebugMode(boolean reallyEnable);

    void enableVerboseMode(boolean reallyEnable);


    void registerGraph(long id, Pointer graph);

    Map<String, INDArray> executeGraph(long id, Map<String, INDArray> map, Map<String, Integer> reverseMap);

    void forgetGraph(long id);

    /**
     * This method allows to set desired number of elements per thread, for performance optimization purposes.
     * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
     *
     * Default value: 1024
     *
     * @param threshold
     */
    void setElementsThreshold(int threshold);

    /**
     * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
     * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
     * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
     *
     * Default value: 8
     * @param threshold
     */
    void setTadThreshold(int threshold);

}
