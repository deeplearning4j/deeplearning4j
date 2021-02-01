/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.cpu.nativecpu.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.List;
import java.util.Queue;

/**
 * Deallocator implementation for CpuWorkspace
 * @author raver119@gmail.com
 */
@Slf4j
public class CpuWorkspaceDeallocator implements Deallocator {
    private PointersPair pointersPair;
    private Queue<PointersPair> pinnedPointers;
    private List<PointersPair> externalPointers;
    private LocationPolicy location;
    private Pair<LongPointer, Long> mmapInfo;

    public CpuWorkspaceDeallocator(@NonNull CpuWorkspace workspace) {
        this.pointersPair = workspace.workspace();
        this.pinnedPointers = workspace.pinnedPointers();
        this.externalPointers = workspace.externalPointers();
        this.location = workspace.getWorkspaceConfiguration().getPolicyLocation();

        if (workspace.mappedFileSize() > 0)
            this.mmapInfo = Pair.makePair(workspace.mmap, workspace.mappedFileSize());
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating CPU workspace");

        // purging workspace planes
        if (pointersPair != null && (pointersPair.getDevicePointer() != null || pointersPair.getHostPointer() != null)) {
            if (pointersPair.getDevicePointer() != null) {
                Nd4j.getMemoryManager().release(pointersPair.getDevicePointer(), MemoryKind.DEVICE);
            }

            if (pointersPair.getHostPointer() != null) {
                if (location != LocationPolicy.MMAP)
                    Nd4j.getMemoryManager().release(pointersPair.getHostPointer(), MemoryKind.HOST);
                else
                    NativeOpsHolder.getInstance().getDeviceNativeOps().munmapFile(null, mmapInfo.getFirst(), mmapInfo.getSecond());
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
