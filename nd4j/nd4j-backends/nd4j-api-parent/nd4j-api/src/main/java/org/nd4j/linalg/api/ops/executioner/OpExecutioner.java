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

package org.nd4j.linalg.api.ops.executioner;

import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.profiler.data.array.eventlog.Nd4jEventLog;
import org.nd4j.linalg.profiler.ProfilerConfig;

import java.util.List;
import java.util.Map;
import java.util.Properties;

public interface OpExecutioner {

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
     * When {@link Environment#isLogNDArrayEvents()}
     *  is true all arrays will log to {@link #getNd4jEventLog()}
     * @return
     */
    Nd4jEventLog getNd4jEventLog();

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


    OpContext injectNewContext();

    void clearOpContext();

    void setNextOpContext(OpContext context);

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
    INDArray exec(Op op);

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    INDArray exec(Op op, OpContext opContext);

    /**Execute a TransformOp and return the result
     * @param op the operation to execute
     */
    TransformOp execAndReturn(TransformOp op);

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
    Variance execAndReturn(Variance op);

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
    ScalarOp execAndReturn(ScalarOp op);

    /** Execute and return the result from a vector op
     * @param op*/
    BroadcastOp execAndReturn(BroadcastOp op);

    /**
     * Execute a reduceOp, possibly along one or more dimensions
     * @param reduceOp the reduceOp
     * @return the reduceOp op
     */
    INDArray exec(ReduceOp reduceOp);

    /**
     * Execute a broadcast op, possibly along one or more dimensions
     * @param broadcast the accumulation
     * @return the broadcast op
     */
    INDArray exec(BroadcastOp broadcast);

    /**
     * Execute ScalarOp
     * @param broadcast
     * @return
     */
    INDArray exec(ScalarOp broadcast);

    /**
     * Execute an variance accumulation op, possibly along one or more dimensions
     * @param accumulation the accumulation
     * @return the accmulation op
     */
    INDArray exec(Variance accumulation);


    /** Execute an index accumulation along one or more dimensions
     * @param indexAccum the index accumulation operation
     * @return result
     */
    INDArray exec(IndexAccumulation indexAccum);



    /**
     *
     * Execute and return  a result
     * ndarray from the given op
     * @param op the operation to execute
     * @return the result from the operation
     */
    Op execAndReturn(Op op);

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
    @Deprecated
    void setProfilingMode(ProfilingMode mode);

    /**
     * This method stores specified configuration.
     *
     * @param config
     */
    void setProfilingConfig(ProfilerConfig config);

    /**
     * Ths method returns current profiling
     *
     * @return
     */
    @Deprecated
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
    CustomOp execAndReturn(CustomOp op);

    INDArray[] exec(CustomOp op);

    /**
     * This method executes op with given context
     * @param op
     * @param context
     * @return method returns output arrays defined within context
     */
    INDArray[] exec(CustomOp op, OpContext context);

    List<DataBuffer> calculateOutputShape(CustomOp op);

    List<DataBuffer> calculateOutputShape(CustomOp op, OpContext opContext);

    /**
     * Equivalent to calli
     */
    INDArray[] allocateOutputArrays(CustomOp op);


    void enableDebugMode(boolean reallyEnable);

    void enableVerboseMode(boolean reallyEnable);

    boolean isExperimentalMode();

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

    /**
     * This method extracts String from Utf8Buffer
     * @param buffer
     * @param index
     * @return
     */
    String getString(DataBuffer buffer, long index);

    /**
     * This method returns OpContext which can be used (and reused) to execute custom ops
     * @return
     */
    OpContext buildContext();

    /**
     *
     * @param array
     */
    INDArrayStatistics inspectArray(INDArray array);


    DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty, boolean isView);

    /**
     * This method returns shapeInfo DataBuffer
     *
     * @param shape
     * @param stride
     * @param elementWiseStride
     * @param order
     * @param dtype
     * @return
     */
    DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty);

    DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extra);

    /**
     * This method returns host/device tad buffers
     */
    TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension);

    /**
     * This method returns constant buffer for the given jvm array
     * @param values
     * @return
     */
    DataBuffer createConstantBuffer(long[] values, DataType desiredType);
    DataBuffer createConstantBuffer(int[] values, DataType desiredType);
    DataBuffer createConstantBuffer(float[] values, DataType desiredType);
    DataBuffer createConstantBuffer(double[] values, DataType desiredType);

    /**
     * This method returns reference use count from the Buffer
     * @param buffer
     * @return
     */
    int useCount(DataBuffer buffer);


}
