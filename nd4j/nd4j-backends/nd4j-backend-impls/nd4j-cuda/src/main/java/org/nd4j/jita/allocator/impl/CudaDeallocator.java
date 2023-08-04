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

package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.profiler.data.eventlogger.EventLogger;
import org.nd4j.linalg.profiler.data.eventlogger.EventType;
import org.nd4j.linalg.profiler.data.eventlogger.LogEvent;
import org.nd4j.linalg.profiler.data.eventlogger.ObjectAllocationType;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class CudaDeallocator implements Deallocator {

    private OpaqueDataBuffer opaqueDataBuffer;
    private LogEvent logEvent;
    private boolean isConstant;
    private AtomicBoolean deallocated = new AtomicBoolean(false);
    private AtomicInteger numTimesCalled = new AtomicInteger(0);
    public CudaDeallocator(@NonNull BaseCudaDataBuffer buffer) {
        opaqueDataBuffer = buffer.getOpaqueDataBuffer();
        isConstant = buffer.isConstant();
        if(EventLogger.getInstance().isEnabled()) {
            logEvent = LogEvent.builder()
                    .attached(buffer.isAttached())
                    .isConstant(buffer.isConstant())
                    .eventType(EventType.DEALLOCATION)
                    .objectAllocationType(ObjectAllocationType.DATA_BUFFER)
                    .associatedWorkspace(Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread().getId())
                    .build();

        }

    }

    @Override
    public synchronized void deallocate() {
        //update the log event with the actual time of de allocation and then
        //perform logging
        if(numTimesCalled.get() > 0)
            return;
        numTimesCalled.incrementAndGet();
        if(logEvent != null) {
            logEvent.setEventTimeMs(System.currentTimeMillis());
            EventLogger.getInstance().log(logEvent);
        }

      //  if(!opaqueDataBuffer.isNull())
      //      NativeOpsHolder.getInstance().getDeviceNativeOps().deleteDataBuffer(opaqueDataBuffer);
    }

    @Override
    public LogEvent logEvent() {
        return logEvent;
    }

    @Override
    public boolean isConstant() {
        return isConstant;
    }
}
