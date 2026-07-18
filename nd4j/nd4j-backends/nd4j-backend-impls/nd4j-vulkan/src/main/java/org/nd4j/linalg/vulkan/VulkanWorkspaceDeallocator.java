/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.vulkan;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.Deallocator;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PointersPair;

import java.util.List;
import java.util.Queue;

/**
 * Releases every allocation class owned by a {@link VulkanWorkspace}.
 *
 * <p>The owning device is captured with the workspace. This is required for
 * multi-device use because a deallocator thread is not implicitly bound to the
 * device that created the allocation.</p>
 */
@Slf4j
public final class VulkanWorkspaceDeallocator implements Deallocator {

    private final VulkanMemoryManager memoryManager;
    private final int deviceId;
    private final PointersPair workspacePointers;
    private final List<PointersPair> reusableDevicePointers;
    private final List<PointersPair> externalPointers;
    private final Queue<PointersPair> pinnedPointers;

    public VulkanWorkspaceDeallocator(@NonNull VulkanWorkspace workspace) {
        this.memoryManager = workspace.vulkanMemoryManager();
        this.deviceId = workspace.workspaceDeviceId();
        this.workspacePointers = workspace.workspacePointers();
        this.reusableDevicePointers = workspace.reusableDevicePointers();
        this.externalPointers = workspace.externalPointers();
        this.pinnedPointers = workspace.pinnedPointers();
    }

    @Override
    public void deallocate() {
        log.trace("Deallocating Vulkan workspace on device {}", deviceId);

        releasePair(workspacePointers);

        for (PointersPair pair : reusableDevicePointers) {
            releasePair(pair);
        }
        reusableDevicePointers.clear();

        for (PointersPair pair : externalPointers) {
            releasePair(pair);
        }
        externalPointers.clear();

        PointersPair pair;
        while ((pair = pinnedPointers.poll()) != null) {
            releasePair(pair);
        }
    }

    private void releasePair(PointersPair pair) {
        if (pair == null) {
            return;
        }

        if (pair.getDevicePointer() != null
                && pair.getDevicePointer().address() != 0L) {
            memoryManager.releaseDevice(pair.getDevicePointer(), deviceId);
            pair.setDevicePointer(null);
        }

        if (pair.getHostPointer() != null
                && pair.getHostPointer().address() != 0L) {
            memoryManager.release(pair.getHostPointer(), MemoryKind.HOST);
            pair.setHostPointer(null);
        }
    }

    @Override
    public boolean isConstant() {
        return false;
    }
}
