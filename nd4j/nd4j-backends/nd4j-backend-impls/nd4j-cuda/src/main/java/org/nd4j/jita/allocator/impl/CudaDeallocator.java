/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.jita.allocator.impl;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

@Slf4j
public class CudaDeallocator implements Deallocator {

    private OpaqueDataBuffer opaqueDataBuffer;

    public CudaDeallocator(@NonNull BaseCudaDataBuffer buffer) {
        opaqueDataBuffer = buffer.getOpaqueDataBuffer();
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating CUDA memory");
        NativeOpsHolder.getInstance().getDeviceNativeOps().deleteDataBuffer(opaqueDataBuffer);
    }
}
