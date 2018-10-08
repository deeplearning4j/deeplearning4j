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

package org.nd4j.autodiff.samediff;

import lombok.Getter;
import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.OpProfiler;

import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

/**
 *
 */
public class SameDiffOpExecutioner implements OpExecutioner,OpProfiler.OpProfilerListener {

    @Getter
    private Map<INDArray,SDVariable> variables;
    @Getter
    private SameDiff sameDiff;
    @Getter
    private AtomicReference<Op> opAtomicReference;
    @Getter
    private OpExecutioner backendExecutioner = Nd4j.getExecutioner();

    public SameDiffOpExecutioner() {
        variables = new IdentityHashMap<>();
        sameDiff = SameDiff.create();
        OpProfiler.getInstance().addListener(this);
    }

    private Op  processOp(Op op) {
        if(opAtomicReference == null) {
            opAtomicReference = new AtomicReference<>(op);
        }

        for(INDArray arr : new INDArray[] {op.x(),op.y(),op.z()}) {
            if(arr == null)
                continue;
            if(!variables.containsKey(arr)) {
                SDVariable sdVariable = sameDiff.var(UUID.randomUUID().toString(),arr);
                variables.put(arr,sdVariable);
            }
        }

        if(op.x() != null && op.y() != null) {
            SDVariable result = sameDiff.invoke(op, variables.get(op.x()), variables.get(op.y()));
            variables.put(op.z(),result);
        }
        else {
            SDVariable result = sameDiff.invoke(op, variables.get(op.x()));
            variables.put(op.z(),result);

        }

        return op;
    }




    /**
     * This method returns opName of the last invoked op
     *
     * @return
     */
    @Override
    public String getLastOp() {
        return opAtomicReference.get().opName();
    }

    /**
     * Execute the operation
     *
     * @param op the operation to execute
     */
    @Override
    public Op exec(Op op) {
        return processOp(op);
    }

    /**
     * Iterate over every row of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllRows(Op op) {
        throw new UnsupportedOperationException();
    }

    /**
     * Iterate over every column of every slice
     *
     * @param op the operation to apply
     */
    @Override
    public void iterateOverAllColumns(Op op) {
        throw new UnsupportedOperationException();
    }

    /**
     * Execute a TransformOp and return the result
     *
     * @param op the operation to execute
     */
    @Override
    public INDArray execAndReturn(TransformOp op) {
        return processOp(op).z();

    }

    /**
     * Execute and return the result from an accumulation
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    @Override
    public Accumulation execAndReturn(Accumulation op) {
        return (Accumulation) processOp(op).z();
    }

    /**
     * Execute and return the result from an accumulation
     *
     * @param op            the operation to execute
     * @param biasCorrected
     * @return the accumulated result
     */
    @Override
    public Accumulation execAndReturn(Variance op, boolean biasCorrected) {
        return (Accumulation) processOp(op);
    }

    /**
     * Execute and return the result from an index accumulation
     *
     * @param op the index accumulation operation to execute
     * @return the accumulated index
     */
    @Override
    public IndexAccumulation execAndReturn(IndexAccumulation op) {
        return (IndexAccumulation) processOp(op);
    }

    /**
     * Execute and return the result from a scalar op
     *
     * @param op the operation to execute
     * @return the accumulated result
     */
    @Override
    public INDArray execAndReturn(ScalarOp op) {
        return processOp(op).z();
    }

    /**
     * Execute and return the result from a vector op
     *
     * @param op
     */
    @Override
    public INDArray execAndReturn(BroadcastOp op) {
        return processOp(op).z();
    }

    /**
     * Execute and return the result from a vector op
     *
     * @param op
     */
    @Override
    public INDArray execAndReturn(ShapeOp op) {
        return backendExecutioner.execAndReturn(op);
    }

    /**
     * Execute the operation along 1 or more dimensions
     *
     * @param op        the operation to execute
     * @param dimension
     */
    @Override
    public Op exec(Op op, int... dimension) {
        return processOp(op);
    }

    /**
     * Execute an accumulation along one or more dimensions
     *
     * @param accumulation the accumulation
     * @param dimension    the dimension
     * @return the accumulation op
     */
    @Override
    public INDArray exec(Accumulation accumulation, int... dimension) {
        return processOp(accumulation).z();
    }

    /**
     * Execute an broadcast along one or more dimensions
     *
     * @param broadcast the accumulation
     * @param dimension the dimension
     * @return the broadcast op
     */
    @Override
    public INDArray exec(BroadcastOp broadcast, int... dimension) {
        return processOp(broadcast).z();
    }

    /**
     * Execute an accumulation along one or more dimensions
     *
     * @param accumulation  the accumulation
     * @param biasCorrected
     * @param dimension     the dimension  @return the accmulation op
     */
    @Override
    public INDArray exec(Variance accumulation, boolean biasCorrected, int... dimension) {
        return processOp(accumulation).z();
    }

    /**
     * Execute an index accumulation along one or more dimensions
     *
     * @param indexAccum the index accumulation operation
     * @param dimension  the dimension/s to execute along
     * @return result
     */
    @Override
    public INDArray exec(IndexAccumulation indexAccum, int... dimension) {
        return processOp(indexAccum).z();
    }

    /**
     * Execute and return  a result
     * ndarray from the given op
     *
     * @param op the operation to execute
     * @return the result from the operation
     */
    @Override
    public INDArray execAndReturn(Op op) {
        return processOp(op).z();
    }

    /**
     * Get the execution mode for this
     * execuioner
     *
     * @return the execution mode for this executioner
     */
    @Override
    public ExecutionMode executionMode() {
        return backendExecutioner.executionMode();
    }

    /**
     * Set the execution mode
     *
     * @param executionMode the execution mode
     */
    @Override
    public void setExecutionMode(ExecutionMode executionMode) {
        backendExecutioner.setExecutionMode(executionMode);
    }

    /**
     * Execute MetaOp
     *
     * @param op
     */
    @Override
    public void exec(MetaOp op) {

    }

    /**
     * Execute GridOp
     *
     * @param op
     */
    @Override
    public void exec(GridOp op) {

    }

    @Override
    public void exec(Aggregate op) {

    }

    /**
     * @param op
     */
    @Override
    public void exec(ShapeOp op) {
        backendExecutioner.exec(op);
    }

    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {

    }

    /**
     * This method takes arbitrary sized list of aggregates, and packs them into batches
     *
     * @param batch
     */
    @Override
    public void exec(List<Aggregate> batch) {

    }

    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return processOp(op).z();
    }

    /**
     * This method executes specific RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return processOp(op).z();
    }

    /**
     * This method return set of key/value and
     * key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        return backendExecutioner.getEnvironmentInformation();
    }

    /**
     * This method specifies desired profiling mode
     *
     * @param mode
     */
    @Override
    public void setProfilingMode(ProfilingMode mode) {
        backendExecutioner.setProfilingMode(mode);
    }

    /**
     * Ths method returns current profiling
     *
     * @return
     */
    @Override
    public ProfilingMode getProfilingMode() {
        return backendExecutioner.getProfilingMode();
    }

    /**
     * This method returns TADManager instance used for this OpExecutioner
     *
     * @return
     */
    @Override
    public TADManager getTADManager() {
        return backendExecutioner.getTADManager();
    }

    /**
     * This method prints out environmental information returned by getEnvironmentInformation() method
     */
    @Override
    public void printEnvironmentInformation() {
        backendExecutioner.printEnvironmentInformation();
    }

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed.
     */
    @Override
    public void push() {
        backendExecutioner.push();
    }

    /**
     * This method ensures all operations that supposed to be executed at this moment, are executed and finished.
     */
    @Override
    public void commit() {
        backendExecutioner.commit();
    }

    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @param threshold
     * @return encoded array is returned
     */
    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return backendExecutioner.thresholdEncode(input,threshold);
    }

    /**
     * This method encodes array as thresholds, updating input array at the same time
     *
     * @param input
     * @param threshold
     * @param boundary  @return encoded array is returned
     */
    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {
        return backendExecutioner.thresholdEncode(input,threshold,boundary);
    }

    /**
     * This method decodes thresholds array, and puts it into target array
     *
     * @param encoded
     * @param target
     * @return target is returned
     */
    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        return backendExecutioner.thresholdDecode(encoded,target);
    }

    /**
     * This method returns number of elements affected by encoder
     *
     * @param indArray
     * @param target
     * @param threshold
     * @return
     */
    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        return backendExecutioner.bitmapEncode(indArray,target,threshold);
    }

    @Override
    public INDArray bitmapEncode(INDArray indArray, double threshold) {
        return backendExecutioner.bitmapEncode(indArray,threshold);
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {
        return backendExecutioner.bitmapDecode(encoded,target);
    }

    @Override
    public void invoke(Op op) {
        processOp(op);

    }

    @Override
    public Map<String, CustomOpDescriptor> getCustomOperations() {
        return backendExecutioner.getCustomOperations();
    }

    /**
     * This method executes given CustomOp
     * <p>
     * PLEASE NOTE: You're responsible for input/output validation
     *
     * @param op
     */
    @Override
    public void exec(CustomOp op) {
        backendExecutioner.exec(op);
    }

    @Override
    public List<long[]> calculateOutputShape(CustomOp op) {
        return backendExecutioner.calculateOutputShape(op);
    }

    @Override
    public INDArray[] allocateOutputArrays(CustomOp op) {
        return backendExecutioner.allocateOutputArrays(op);
    }


    @Override
    public void registerGraph(long id, Pointer graph) {
        backendExecutioner.registerGraph(id, graph);
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, @NonNull Map<String, INDArray> map, @NonNull Map<String, Integer> reverseMap) {
        return backendExecutioner.executeGraph(id, map, reverseMap);
    }

    @Override
    public void forgetGraph(long id) {
        backendExecutioner.forgetGraph(id);
    }

    @Override
    public void enableDebugMode(boolean reallyEnable) {
        backendExecutioner.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        backendExecutioner.enableVerboseMode(reallyEnable);
    }

    /**
     * This method allows to set desired number of elements per thread, for performance optimization purposes.
     * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
     * <p>
     * Default value: 1024
     *
     * @param threshold
     */
    @Override
    public void setElementsThreshold(int threshold) {
        backendExecutioner.setElementsThreshold(threshold);
    }

    /**
     * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
     * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
     * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
     * <p>
     * Default value: 8
     *
     * @param threshold
     */
    @Override
    public void setTadThreshold(int threshold) {
        backendExecutioner.setTadThreshold(threshold);
    }

    @Override
    public ExecutionerType type() {
        return backendExecutioner.type();
    }

    @Override
    public boolean isVerbose() {
        return backendExecutioner.isVerbose();
    }

    @Override
    public boolean isDebug() {
        return backendExecutioner.isDebug();
    }
}
