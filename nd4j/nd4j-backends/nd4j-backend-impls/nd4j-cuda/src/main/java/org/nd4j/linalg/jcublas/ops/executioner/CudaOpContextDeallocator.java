/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.jcublas.ops.executioner;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;

/**
 * Deallocator for CudaOpContext that ensures proper cleanup order to prevent use-after-free.
 */
@Slf4j
public class CudaOpContextDeallocator implements Deallocator {
    private transient final OpaqueContext context;
    private final int deviceId;
    private volatile boolean deallocated = false;

    /**
     * Creates a deallocator that will properly clean up the CudaOpContext.
     *
     * @param ctx The CudaOpContext to deallocate
     */
    public CudaOpContextDeallocator(CudaOpContext ctx) {
        this.context = (OpaqueContext) ctx.contextPointer();
        this.deviceId = ctx.targetDevice();
    }

    @Override
    public void deallocate() {
        if (deallocated) {
            return;
        }

        synchronized (this) {
            if (deallocated) {
                return;
            }

            if (context == null || context.isNull()) {
                deallocated = true;
                return;
            }

            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
                boolean switchedDevice = false;

                if (currentDevice != deviceId) {
                    // Use unsafeSetDevice which properly updates both native device
                    // AND resets the cached CUDA context (ThreadLocal) for this thread.
                    // Just calling nativeOps.setDevice() doesn't update the ThreadLocal context,
                    // so commit() would sync the wrong device's streams.
                    Nd4j.getAffinityManager().unsafeSetDevice(deviceId);
                    switchedDevice = true;
                }

                try {
                    Nd4j.getExecutioner().commit();

                    // Step 1: Clear the native Context's fastpath pointers FIRST.
                    // This prevents the Context from holding dangling pointers to arrays.
                    // Double-check isNull() in case close() was called concurrently.
                    if (context != null && !context.isNull()) {
                        nativeOps.ctxPurge(context);
                    }

                    // Step 2: Delete the native Context object.
                    // Double-check isNull() again for same reason.
                    if (context != null && !context.isNull()) {
                        nativeOps.deleteGraphContext(context);
                    }
                } finally {
                    // Restore original device context
                    if (switchedDevice) {
                        Nd4j.getAffinityManager().unsafeSetDevice(currentDevice);
                    }
                }

            } catch (Exception e) {
                log.error("Error during CudaOpContext deallocation", e);
            } finally {
                deallocated = true;
            }
        }
    }

    @Override
    public boolean isConstant() {
        return false;
    }
}
