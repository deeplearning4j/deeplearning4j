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

package org.nd4j.linalg.cpu.nativecpu;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.BasicMemoryManager;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class CpuMemoryManager extends BasicMemoryManager {
    /**
     * This method returns
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param bytes
     * @param kind
     * @param initialize
     */
    @Override
    public Pointer allocate(long bytes, MemoryKind kind, boolean initialize) {
        Pointer ptr = NativeOpsHolder.getInstance().getDeviceNativeOps().mallocHost(bytes, 0);

        if (ptr == null || ptr.address() == 0L)
            throw new ND4JIllegalStateException("Failed to allocate [" + bytes + "] bytes");

        //log.info("Allocating {} bytes at MemoryManager", bytes);


        if (initialize)
            Pointer.memset(ptr, 0, bytes);

        return ptr;
    }

    /**
     * This method releases previously allocated memory chunk
     *
     * @param pointer
     * @param kind
     * @return
     */
    @Override
    public void release(@NonNull Pointer pointer, MemoryKind kind) {
        Pointer.free(pointer);
    }

    /**
     * This method detaches off-heap memory from passed INDArray instances, and optionally stores them in cache for future reuse
     * PLEASE NOTE: Cache options depend on specific implementations
     *
     * @param arrays
     */
    @Override
    public void collect(INDArray... arrays) {
        super.collect(arrays);
    }

    /**
     * Nd4j-native backend doesn't use periodic GC. This method will always return false.
     *
     * @return
     */
    @Override
    public boolean isPeriodicGcActive() {
        return false;
    }

    @Override
    public void memset(INDArray array) {
        if (array.isView()) {
            array.assign(0.0);
            return;
        }

        Pointer.memset(array.data().addressPointer(), 0, array.data().length() * Nd4j.sizeOfDataType(array.data().dataType()));
    }

    @Override
    public Map<Integer, Long> getBandwidthUse() {
        return null;
    }
}
