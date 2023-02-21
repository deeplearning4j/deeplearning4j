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

package org.nd4j.linalg.profiler;

import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.profiler.data.OpContextInfo;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * An OpContextTracker handles
 * counting and aggregating data about op contexts.
 * This mainly includes information around their memory usage,
 * when they are allocated and deallocated.
 *
 * @author Adam Gibson
 */
public class OpContextTracker {

    private static OpContextTracker INSTANCE = new OpContextTracker();
    private Map<Long, OpContextInfo> opContextInfo = new ConcurrentHashMap<>();
    private AtomicBoolean enabled = new AtomicBoolean(false);
    protected OpContextTracker() {}



    public void setEnabled(boolean enabled) {
        this.enabled.set(enabled);
    }


    public boolean isEnabled() {
        return enabled.get();
    }

    public static OpContextTracker getInstance() {
        return INSTANCE;
    }


    /**
     * The number of allocated op contexts
     * @return
     */
    public long numAllocated() {
        return opContextInfo.values().stream()
                .filter(input -> input.isAllocated())
                .count();
    }

    /**
     * The number of deallocated op contexts
     * @return
     */

    public long numDeallocated() {
        return opContextInfo.values().stream()
                .filter(input -> !input.isAllocated())
                .count();
    }


    /**
     * Returns the number of allocated output bytes
     * for the op context.
     * @return
     */
    public long opContextAllocatedBytes() {
        return opContextInfo.values()
                .stream().map(input -> input.allocatedOutputBytes())
                .collect(Collectors.summingLong(Long::longValue));
    }

    /**
     * Total output bytes on op allocated contexts.
     * @return
     */
    public long totalOutputBytes() {
        return opContextInfo.values().stream()
                .filter(input -> input.isAllocated())
                .map(opContextInfo1 -> opContextInfo1.allocatedOutputBytes()).collect(Collectors.summingLong(Long::longValue));
    }

    /**
     * Total input bytes on op allocated contexts.
     * @return
     */
    public long totalInputBytes() {
        return opContextInfo.values().stream()
                .filter(input -> input.isAllocated())
                .map(opContextInfo1 -> opContextInfo1.allocatedInputBytes())
                .collect(Collectors.summingLong(Long::longValue));
    }

    /**
     * Print statistics for the current set of op contexts.
     * This will include information including:
     * 1. {@link #numAllocated()}
     * 2. {@link #numDeallocated()}
     * 3. {@link #totalInputBytes()}
     * 4. {@link #totalOutputBytes()}
     * @param printIndividualContexts whether to print individual op contexts or not
     * @return the string representation of the op contexts
     */
    public String printStats(boolean printIndividualContexts) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Op contexts allocated: " + numAllocated() + "\n");
        stringBuilder.append("Op contexts deallocated: " + numDeallocated() + "\n");
        stringBuilder.append("Total bytes as inputs to allocated op contexts: " + totalInputBytes() + "\n");
        stringBuilder.append("Total bytes as outputs of allocated op contexts: " + totalOutputBytes() + "\n");
        if(printIndividualContexts) {
            opContextInfo.values().forEach(opContextInfo1 -> {
                stringBuilder.append(opContextInfo1);
            });
        }
        return stringBuilder.toString();
    }


    public void purge(OpContext opContext) {
        opContextInfo.get(opContext.id()).purge();
    }

    /**
     * Indicates an op context is deallocated
     * @param opContext  the op context
     */
    public void deallocateContext(OpContext opContext) {
        opContextInfo.get(opContext.id()).setAllocated(false);
    }

    /**
     * Indicates an op context is deallocated
     * @param ctxId the id of the op context
     */
    public void deallocateContext(long ctxId) {
        opContextInfo.get(ctxId).setAllocated(false);
    }

    /**
     * Indicates an op context is allocated
     * This will record information about the op context uponc reation
     * @param opContext
     */
    public void allocateOpContext(OpContext opContext) {
        OpContextInfo opContextInfo1  = OpContextInfo.builder()
                .allocated(true)
                .id(opContext.id())
                .build();

        opContextInfo.put(opContext.id(),opContextInfo1);

    }


    /**
     * Associates an input with the recorded
     * op context information
     * @param input the input to record
     * @param opContext the op context to add the input to
     */
    public void associateInput(INDArray input,OpContext opContext) {
        opContextInfo.get(opContext.id()).addInput(input);
    }


    /**
     * Associates an output with the recorded
     * op context information.
     * Note this calls {@link #associateOutput(INDArray, boolean, OpContext)}
     * with a default value of true. Most of the time output arrays
     * are created by the op context on demand.
     * @param input the input to record
     * @param opContext the op context to add the output to
     */
    public void associateOutput(INDArray input,OpContext opContext) {
        //op outputs are usually allocated alongside an op context after calculateOutputShape
        //is called in op executioners, default to true
        opContextInfo.get(opContext.id()).addOutput(input,true);
    }

    /**
     * Associates an output with the recorded
     * op context information
     * @param input the input to record
     * @param createdByContext whether the array was created by the op context or not
     * @param opContext the op context to add the output to
     */
    public void associateOutput(INDArray input,boolean createdByContext,OpContext opContext) {
        opContextInfo.get(opContext.id()).addOutput(input,createdByContext);
    }



}
