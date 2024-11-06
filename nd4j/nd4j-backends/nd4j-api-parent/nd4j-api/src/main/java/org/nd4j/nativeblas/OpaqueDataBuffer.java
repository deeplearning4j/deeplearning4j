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

package org.nd4j.nativeblas;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

@Slf4j
public class OpaqueDataBuffer extends Pointer {
    private static final int MAX_TRIES = 5;
    private String allocationTrace = null;
    public static AtomicBoolean currentlyExecuting = new AtomicBoolean(false);

    /**
     * Record the current allocation stack trace.
     * This is mainly used when {@link NativeOps#isFuncTrace()}
     * is true. A build of the c++ library has to be generated with the library
     * in order for this to return true.
     *
     * Please do not use this in production. Only use func trace with debug builds.
     */

    public void captureTrace() {
        if(currentlyExecuting.get())
            return;
        currentlyExecuting.set(true);
        allocationTrace = currentTrace();
    }

    public void printNativeAllocationTrace() {

    }

    private String currentTrace() {
        return Arrays.toString(Thread.currentThread().getStackTrace()).replace( ',', '\n');
    }


    public OpaqueDataBuffer(Pointer p) { super(p); }


    public static void tracingSetExecuting(boolean executing) {
        currentlyExecuting.set(executing);
    }

    public static OpaqueDataBuffer externalizedDataBuffer(long numElements, @NonNull DataType dataType, Pointer primary, Pointer special) {
        OpaqueDataBuffer ret =Nd4j.getNativeOps().dbCreateExternalDataBuffer(numElements, dataType.toInt(), primary, special);
        if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
            ret.captureTrace();
        return ret;
    }

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
                buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType.toInt(), allocateBoth);
                //when  using func trace we want to print allocation traces when deallocation is called. this is used to debug
                //potential race condition and crashes. c++ prints the equivalent stack trace when func trace is enabled.
                //This allows us to check where a deallocated buffer that caused an issue was allocated.
                if(buffer != null && Nd4j.getNativeOps().isFuncTrace())
                    buffer.captureTrace();
                // check error code
                ec = Nd4j.getNativeOps().lastErrorCode();
                if (ec != 0) {
                    em = Nd4j.getNativeOps().lastErrorMessage();

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
        throw new RuntimeException("Allocation failed: [" + em + "] for amount of memory " + numElements * dataType.width() + " bytes");
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
               Nd4j.getNativeOps().dbExpand(this, numElements);

                // check error code
                ec =Nd4j.getNativeOps().lastErrorCode();
                if (ec != 0) {
                    em =Nd4j.getNativeOps().lastErrorMessage();

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
     * @return
     */
    public OpaqueDataBuffer createView(long bytesLength) {
        OpaqueDataBuffer buffer = null;
        int ec = 0;
        String em = null;

        for (int t = 0; t < MAX_TRIES; t++) {
            try {
                buffer =Nd4j.getNativeOps().dbCreateView(this, bytesLength);
                if(NativeOpsHolder.getInstance().getDeviceNativeOps().isFuncTrace())
                    buffer.captureTrace();
                // check error code
                ec =Nd4j.getNativeOps().lastErrorCode();

                if (ec != 0) {
                    em =Nd4j.getNativeOps().lastErrorMessage();

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

    public long numElements() {
        return Nd4j.getNativeOps().dbBufferLength(this);
    }

    /**
     * This method returns pointer to linear buffer, primary one.
     * @return
     */
    public Pointer primaryBuffer() {
        return Nd4j.getNativeOps().dbPrimaryBuffer(this);
    }


    /**
     * This method returns pointer to special buffer, device one, if any.
     * @return
     */
    public Pointer specialBuffer() {
        return Nd4j.getNativeOps().
                dbSpecialBuffer(this);
    }

    /**
     * This method returns deviceId of this DataBuffer
     * @return
     */
    public int deviceId() {
        return Nd4j.getNativeOps().dbDeviceId(this);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setPrimaryBuffer(Pointer ptr, long numElements) {
        //note we call print here because dbSetSpecialBuffer can deallocate on the c++ side
        printAllocationTraceIfNeeded();
       Nd4j.getNativeOps().dbSetPrimaryBuffer(this, ptr, numElements);
    }

    /**
     * This method allows to set external pointer as primary buffer.
     *
     * PLEASE NOTE: if InteropDataBuffer owns current memory buffer, it will be released
     * @param ptr
     * @param numElements
     */
    public void setSpecialBuffer(Pointer ptr, long numElements) {
        //note we call print here because dbSetSpecialBuffer can deallocate on the c++ side
        printAllocationTraceIfNeeded();

       Nd4j.getNativeOps().dbSetSpecialBuffer(this, ptr, numElements);
    }

    /**
     * This method synchronizes device memory
     */
    public void syncToSpecial() {
       Nd4j.getNativeOps().dbSyncToSpecial(this);
    }

    /**
     * This method synchronizes host memory
     */
    public void syncToPrimary() {
       Nd4j.getNativeOps().dbSyncToPrimary(this);
    }

    public void printAllocationTraceIfNeeded() {
        if(allocationTrace != null && Nd4j.getEnvironment().isFuncTracePrintAllocate()) {
            System.out.println("Java side allocation trace: \n " + allocationTrace);
        }
    }

    /**
     * This method releases underlying buffer
     */
    public  void closeBuffer() {
        printAllocationTraceIfNeeded();
        if(Nd4j.getEnvironment().isFuncTracePrintDeallocate()) {
            System.out.println("Java side deallocation current trace: \n " + currentTrace());
        }
       Nd4j.getNativeOps().dbClose(this);
    }
}
