/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.nd4j.nativeblas;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * This class is a opaque pointer to InteropDataBuffer, used for Java/C++ interop related to INDArray DataBuffer
 *
 * @author saudet
 * @author raver119@gmail.com
 */
@Slf4j
public class OpaqueDataBuffer extends Pointer {
    // TODO: make this configurable
    private static final int MAX_TRIES = 5;

    public OpaqueDataBuffer(Pointer p) { super(p); }

    /**
     * This method allocates new InteropDataBuffer and returns pointer to it
     * @param numElements
     * @param dataType
     * @param allocateBoth
     * @return
     */
    public static OpaqueDataBuffer allocateDataBuffer(long numElements, @NonNull DataType dataType, boolean allocateBoth) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to allocate data buffer
                buffer = NativeOpsHolder.getInstance().getDeviceNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);
                // check error code
                ec = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorCode();
                if (ec != 0) {
                    em = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorMessage();

                    // if allocation failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    // sleeping for 50ms
                    Thread.sleep(50);
                } else {
                    // just return the buffer
                    return buffer;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        // if MAX_TRIES is over, we'll just throw an exception
        throw new RuntimeException("Allocation failed: [" + em + "]");
    }

    /**
     * This method expands buffer, and copies content to the new buffer
     *
     * PLEASE NOTE: if InteropDataBuffer doesn't own actual buffers - original pointers won't be released
     * @param numElements
     */
    public void expand(long numElements) {
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                // try to expand the buffer
                NativeOpsHolder.getInstance().getDeviceNativeOps().dbExpand(this, numElements);

                // check error code
                ec = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorCode();
                if (ec != 0) {
                    em = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorMessage();

                    // if expansion failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    Thread.sleep(50);
                } else {
                    // just return
                    return;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        // if MAX_TRIES is over, we'll just throw an exception
        throw new RuntimeException("DataBuffer expansion failed: [" + em + "]");
    }

    /**
     * This method creates a view out of this InteropDataBuffer
     *
     * @param bytesLength
     * @param bytesOffset
     * @return
     */
    public OpaqueDataBuffer createView(long bytesLength, long bytesOffset) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                buffer = NativeOpsHolder.getInstance().getDeviceNativeOps().dbCreateView(this, bytesLength, bytesOffset);

                // check error code
                ec = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorCode();
                if (ec != 0) {
                    em = NativeOpsHolder.getInstance().getDeviceNativeOps().lastErrorMessage();

                    // if view creation failed it might be caused by casual OOM, so we'll try GC
                    System.gc();

                    // sleeping to let gc kick in
                    Thread.sleep(50);
                } else {
                    // just return
                    return buffer;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        // if MAX_TRIES is over, we'll just throw an exception
        throw new RuntimeException("DataBuffer expansion failed: [" + em + "]");
    }

    /**
     * This method returns pointer to linear buffer, primary one.
     * @return
     */
    public Pointer primaryBuffer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbPrimaryBuffer(this);
    }

    /**
     * This method returns pointer to special buffer, device one, if any.
     * @return
     */
    public Pointer specialBuffer() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbSpecialBuffer(this);
    }

    /**
     * This method returns deviceId of this DataBuffer
     * @return
     */
    public int deviceId() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().dbDeviceId(this);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setPrimaryBuffer(Pointer ptr, long numElements) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetPrimaryBuffer(this, ptr, numElements);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setSpecialBuffer(Pointer ptr, long numElements) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSetSpecialBuffer(this, ptr, numElements);
    }

    /**
     * This method synchronizes device memory
     */
    public void syncToSpecial() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSyncToSpecial(this);
    }

    /**
     * This method synchronizes host memory
     */
    public void syncToPrimary() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbSyncToPrimary(this);
    }
}
