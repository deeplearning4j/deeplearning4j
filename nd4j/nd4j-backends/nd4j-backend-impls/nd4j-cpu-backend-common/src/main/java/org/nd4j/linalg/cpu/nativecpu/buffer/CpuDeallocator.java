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

package org.nd4j.linalg.cpu.nativecpu.buffer;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.eventlogger.EventLogger;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

@Slf4j
public class CpuDeallocator implements Deallocator {
    private final transient OpaqueDataBuffer opaqueDataBuffer;
    private LogEvent logEvent;
    private volatile boolean isConstant;

    public CpuDeallocator(BaseCpuDataBuffer buffer) {
        opaqueDataBuffer = buffer.getOpaqueDataBuffer();
        isConstant = buffer.isConstant();

        if(EventLogger.getInstance().isEnabled()) {
            logEvent = LogEvent.builder()
                    .attached(buffer.isAttached())
                    .objectId(buffer.getUniqueId())
                    .isConstant(buffer.isConstant())
                    .bytes(buffer.getElementSize() * buffer.length())
                    .dataType(buffer.dataType())
                    .eventType(EventType.DEALLOCATION)
                    .objectAllocationType(ObjectAllocationType.DATA_BUFFER)
                    .associatedWorkspace(Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getId())
                    .build();

        }
    }

    @Override
    public void deallocate() {
        if (isConstant) {
            if (log.isTraceEnabled()) {
                log.trace("Skipping deallocation of constant buffer");
            }
            return;
        }

        if (opaqueDataBuffer == null || opaqueDataBuffer.isNull()) {
            if (log.isTraceEnabled()) {
                log.trace("Skipping deallocation of already-freed buffer");
            }
            return;
        }

        //update the log event with the actual time of de allocation and then
        //perform logging
        if(logEvent != null) {
            logEvent.setEventTimeMs(System.currentTimeMillis());
            logEvent.setThreadName(Thread.currentThread().getName());
            EventLogger.getInstance().log(logEvent);
        }

        Nd4j.getNativeOps().dbClose(opaqueDataBuffer);

        opaqueDataBuffer.setNull();
    }

    @Override
    public boolean isConstant() {
        return isConstant;
    }

    /**
     * Updates the constant flag for this deallocator.
     * This is called when DataBuffer.setConstant() is invoked to ensure
     * that buffers marked as constant (like shape info) are never freed.
     *
     * @param constant true if this buffer should never be deallocated
     */
    @Override
    public void setConstant(boolean constant) {
        this.isConstant = constant;
        // Also update the log event if present
        if (logEvent != null) {
            logEvent.setConstant(constant);
        }
    }
}
