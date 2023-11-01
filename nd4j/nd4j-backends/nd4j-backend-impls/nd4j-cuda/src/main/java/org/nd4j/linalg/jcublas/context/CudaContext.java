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

import java.util.concurrent.atomic.AtomicReference;

/**
 * A higher level class for handling
 * the different primitives around the cuda apis
 * This being:
 * streams (both old and new) as well as
 * the cublas handles.
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

    // exactly what it says
    private cublasHandle_t cublasHandle;
    private cusolverDnHandle_t solverHandle;

    // temporary buffers, exactly 1 per thread
    private Pointer bufferReduction;
    private Pointer bufferAllocation;
    private Pointer bufferScalar;

    // legacy. to be removed.
    private Pointer bufferSpecial;

    private int deviceId = -1;

    private  final static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    private AtomicReference<PointerPointer> oldStreamRef = new AtomicReference<>();
    private AtomicReference<PointerPointer> cublasHandleRef = new AtomicReference<>();
    private AtomicReference<PointerPointer> solverRef = new AtomicReference<>();

    @Override
    public String toString() {
        return "CudaContext{" +
                "bufferReduction=" + bufferReduction +
                ", bufferScalar=" + bufferScalar +
                ", deviceId=" + deviceId +
                '}';
    }

    /**
     * Synchronizes
     * on the old stream
     */
    public void syncOldStream() {
        if (nativeOps.streamSynchronize(oldStream) == 0)
            throw new ND4JIllegalStateException("CUDA stream synchronization failed");
    }

    public void syncSpecialStream() {
        if (nativeOps.streamSynchronize(specialStream) == 0)
            throw new ND4JIllegalStateException("CUDA special stream synchronization failed");
    }

    public Pointer getCublasStream() {
        if(oldStreamRef.getAcquire() == null) {
            oldStreamRef.set(new PointerPointer(this.getOldStream()));
        }
        return oldStreamRef.getAcquire().get(0);
    }

    public cublasHandle_t getCublasHandle() {
        if(cublasHandleRef.getAcquire() == null) {
            cublasHandleRef.set(new PointerPointer(cublasHandle));
        }

        return new cublasHandle_t(cublasHandleRef.get().get(0));
    }

    public cusolverDnHandle_t getSolverHandle() {
        // FIXME: can we cache this please
        if(solverRef.getAcquire() == null) {
            solverRef.set(new PointerPointer(solverHandle));
        }
        return new cusolverDnHandle_t(solverRef.get().get(0));
    }
}
