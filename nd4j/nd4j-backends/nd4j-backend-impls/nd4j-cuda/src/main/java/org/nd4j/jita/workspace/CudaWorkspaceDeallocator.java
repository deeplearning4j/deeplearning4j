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

package org.nd4j.jita.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Queue;

/**
 * Deallocator implementation for CpuWorkspace
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaWorkspaceDeallocator implements Deallocator {
    private PointersPair pointersPair;
    private Queue<PointersPair> pinnedPointers;
    private List<PointersPair> externalPointers;

    public CudaWorkspaceDeallocator(@NonNull CudaWorkspace workspace) {
        this.pointersPair = workspace.workspace();
        this.pinnedPointers = workspace.pinnedPointers();
        this.externalPointers = workspace.externalPointers();
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating CUDA workspace");

        // purging workspace planes
        if (pointersPair != null) {
            if (pointersPair.getDevicePointer() != null) {
                //log.info("Deallocating device...");
                Nd4j.getMemoryManager().release(pointersPair.getDevicePointer(), MemoryKind.DEVICE);
            }

            if (pointersPair.getHostPointer() != null) {
                //                                log.info("Deallocating host...");
                Nd4j.getMemoryManager().release(pointersPair.getHostPointer(), MemoryKind.HOST);
            }
        }

        // purging all spilled pointers
        for (PointersPair pair2 : externalPointers) {
            if (pair2 != null) {
                if (pair2.getHostPointer() != null)
                    Nd4j.getMemoryManager().release(pair2.getHostPointer(), MemoryKind.HOST);

                if (pair2.getDevicePointer() != null)
                    Nd4j.getMemoryManager().release(pair2.getDevicePointer(), MemoryKind.DEVICE);
            }
        }

        // purging all pinned pointers
        // purging all spilled pointers
        for (PointersPair pair2 : externalPointers) {
            if (pair2 != null) {
                if (pair2.getHostPointer() != null)
                    Nd4j.getMemoryManager().release(pair2.getHostPointer(), MemoryKind.HOST);

                if (pair2.getDevicePointer() != null)
                    Nd4j.getMemoryManager().release(pair2.getDevicePointer(), MemoryKind.DEVICE);
            }
        }

        // purging all pinned pointers
        PointersPair pair = null;
        while ((pair = pinnedPointers.poll()) != null) {
            if (pair.getHostPointer() != null)
                Nd4j.getMemoryManager().release(pair.getHostPointer(), MemoryKind.HOST);

            if (pair.getDevicePointer() != null)
                Nd4j.getMemoryManager().release(pair.getDevicePointer(), MemoryKind.DEVICE);
        }

    }
}
