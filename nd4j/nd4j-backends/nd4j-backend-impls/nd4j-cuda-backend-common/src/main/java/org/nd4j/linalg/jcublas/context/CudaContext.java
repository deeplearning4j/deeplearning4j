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

package org.nd4j.linalg.jcublas.context;

import lombok.*;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.jita.allocator.pointers.cuda.cublasHandle_t;
import org.nd4j.jita.allocator.pointers.cuda.cudaStream_t;
import org.nd4j.jita.allocator.pointers.cuda.cusolverDnHandle_t;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueLaunchContext;

/**
 * A higher level class for handling
 * the different primitives around the cuda apis
 * This being:
 * streams (both old and new) as well as
 * the cublas and cusolver handles.
 *
 *
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class CudaContext {

    // execution stream
    private cudaStream_t oldStream;

    // memcpy stream
    private cudaStream_t specialStream;

    // temporary buffers, exactly 1 per thread
    private Pointer bufferReduction;
    private Pointer bufferAllocation;
    private Pointer bufferScalar;

    // legacy. to be removed.
    private Pointer bufferSpecial;

    private int deviceId = -1;

    private transient final static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    @Override
    public String toString() {
        return "CudaContext{" +
                "bufferReduction=" + bufferReduction +
                ", bufferScalar=" + bufferScalar +
                ", deviceId=" + deviceId +
                '}';
    }

    /**
     * Synchronizes on the old stream.
     *
     * IMPORTANT: We must get fresh stream pointers from native each time because:
     * 1. Native code may release/recreate streams when device changes (thread_local contextBuffers)
     * 2. Using cached Java pointers to destroyed streams causes SIGSEGV crashes
     * 3. Native's execStream()/specialStream() methods handle device validation and stream recreation
     */
    public void syncOldStream() {
        // Get fresh launch context and stream pointer from native
        // This ensures we use the currently valid stream, not a potentially stale cached pointer.
        // Both are raw pointers into native state: Java owns neither and frees neither.
        OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
        Pointer freshStream = nativeOps.lcExecutionStream(lc);
        if (freshStream == null || freshStream.isNull()) {
            throw new ND4JIllegalStateException("CUDA execution stream is null - context may not be initialized");
        }
        cudaStream_t stream = new cudaStream_t(freshStream);
        if (nativeOps.streamSynchronize(stream) == 0)
            throw new ND4JIllegalStateException("CUDA stream synchronization failed");
    }

    public void syncSpecialStream() {
        // Get fresh launch context and stream pointer from native
        OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
        Pointer freshStream = nativeOps.lcCopyStream(lc);
        if (freshStream == null || freshStream.isNull()) {
            throw new ND4JIllegalStateException("CUDA special stream is null - context may not be initialized");
        }
        cudaStream_t stream = new cudaStream_t(freshStream);
        if (nativeOps.streamSynchronize(stream) == 0)
            throw new ND4JIllegalStateException("CUDA special stream synchronization failed");
    }

    public Pointer getCublasStream() {
        // FIXME: can we cache this please
        val lptr = new PointerPointer(this.getOldStream());
        return lptr.get(0);
    }

    /**
     * Returns the calling thread's cuBLAS handle for its current device. cuBLAS handles are not
     * thread-safe, so native code keeps one per thread and hands it to another thread once the
     * first exits. A context can be built on one thread and used on another, so the handle is
     * looked up on every call instead of being stored in the context.
     */
    public cublasHandle_t getCublasHandle() {
        OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
        Pointer handle = nativeOps.lcBlasHandle(lc);
        if (handle == null || handle.isNull()) {
            throw new ND4JIllegalStateException("cuBLAS handle is null or invalid for device " + deviceId +
                ". This may indicate CUDA context corruption or device reset.");
        }
        return new cublasHandle_t(new PointerPointer(handle).get(0));
    }

    /**
     * Returns the cuSolver handle for the calling thread's current device, creating it on first
     * use. Backends without cuSolver (ZLUDA) have none, and asking for it fails.
     */
    public cusolverDnHandle_t getSolverHandle() {
        OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
        Pointer handle = nativeOps.lcSolverHandle(lc);
        if (handle == null || handle.isNull()) {
            throw new ND4JIllegalStateException("cuSolver handle is null or invalid for device " + deviceId +
                ". This may indicate CUDA context corruption or device reset.");
        }
        return new cusolverDnHandle_t(new PointerPointer(handle).get(0));
    }

    /**
     * Get the execution stream.
     *
     * Returns the cached stream pointer. This is safe because:
     * 1. The context is tied to a specific device (deviceId field)
     * 2. Streams are device-specific and valid for the life of the context
     * 3. sync methods get fresh streams to handle edge cases
     *
     * @return cudaStream_t for execution operations
     */
    public cudaStream_t getOldStream() {
        if (oldStream == null || oldStream.isNull()) {
            throw new ND4JIllegalStateException("CUDA execution stream is null for device " + deviceId);
        }
        return oldStream;
    }

    /**
     * Get the special/copy stream.
     *
     * Returns the cached stream pointer. This is safe because:
     * 1. The context is tied to a specific device (deviceId field)
     * 2. Streams are device-specific and valid for the life of the context
     * 3. sync methods get fresh streams to handle edge cases
     *
     * @return cudaStream_t for copy/special operations
     */
    public cudaStream_t getSpecialStream() {
        if (specialStream == null || specialStream.isNull()) {
            throw new ND4JIllegalStateException("CUDA special stream is null for device " + deviceId);
        }
        return specialStream;
    }
}
