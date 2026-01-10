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
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueContext;

/**
 * Deallocator for CudaOpContext that ensures proper cleanup order to prevent use-after-free.
 *
 * <p><b>CRITICAL:</b> The deallocation order is essential:</p>
 * <ol>
 *   <li>Call ctxPurge() to clear native Context's fastpath pointers (prevents dangling refs)</li>
 *   <li>Delete the native Context</li>
 * </ol>
 *
 * <p>Without this order, the native Context might still hold pointers to deleted NDArray* objects,
 * causing "invalid rank" errors when shape info is accessed from freed memory.</p>
 */
@Slf4j
public class CudaOpContextDeallocator implements Deallocator {
    private transient final OpaqueContext context;
    private volatile boolean deallocated = false;

    /**
     * Creates a deallocator that will properly clean up the CudaOpContext.
     *
     * @param ctx The CudaOpContext to deallocate
     */
    public CudaOpContextDeallocator(CudaOpContext ctx) {
        this.context = (OpaqueContext) ctx.contextPointer();
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

            try {
                NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

                // Step 1: CRITICAL - Clear the native Context's fastpath pointers FIRST.
                // This prevents the Context from holding dangling pointers to arrays we're about to delete.
                // ctxPurge() calls Context::clearFastPath() which clears _fastpath_in and _fastpath_out.
                if (context != null && !context.isNull()) {
                    nativeOps.ctxPurge(context);
                }

                // Step 2: Delete the native Context object.
                // Now safe to delete since fastpath is cleared.
                if (context != null && !context.isNull()) {
                    nativeOps.deleteGraphContext(context);
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
