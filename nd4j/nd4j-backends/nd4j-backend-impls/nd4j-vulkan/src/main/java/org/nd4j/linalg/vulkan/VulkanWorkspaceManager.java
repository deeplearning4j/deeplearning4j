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
 *  * License for the specific language governing copyright ownership.
 *  * limitations under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.vulkan;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.DebugMode;
import org.nd4j.linalg.api.memory.provider.BasicWorkspaceManager;

/**
 * Vulkan workspace manager.
 *
 * <p>This follows the CUDA backend boundary: the manager is a native-device
 * workspace provider, not a specialization of the CPU backend. Host allocations
 * remain a legitimate workspace plane for staging and mirrored buffers; device
 * requests are owned by {@link VulkanWorkspace} and are never redirected to the
 * host.</p>
 */
public class VulkanWorkspaceManager extends BasicWorkspaceManager {

    protected MemoryWorkspace newWorkspace(WorkspaceConfiguration configuration) {
        return getDebugMode() == DebugMode.BYPASS_EVERYTHING
                ? new DummyWorkspace()
                : owned(new VulkanWorkspace(configuration));
    }

    protected MemoryWorkspace newWorkspace(WorkspaceConfiguration configuration, String id) {
        return getDebugMode() == DebugMode.BYPASS_EVERYTHING
                ? new DummyWorkspace()
                : owned(new VulkanWorkspace(configuration, id));
    }

    protected MemoryWorkspace newWorkspace(WorkspaceConfiguration configuration, String id, int deviceId) {
        return getDebugMode() == DebugMode.BYPASS_EVERYTHING
                ? new DummyWorkspace()
                : owned(new VulkanWorkspace(configuration, id, deviceId));
    }

    private VulkanWorkspace owned(VulkanWorkspace workspace) {
        workspace.attachWorkspaceManager(this);
        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        ensureThreadExistense();
        MemoryWorkspace workspace = newWorkspace(configuration);
        backingMap.get().put(workspace.getId(), workspace);
        pickReferenceUnlessBypassed(workspace);
        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        ensureThreadExistense();
        MemoryWorkspace workspace = newWorkspace(defaultConfiguration);
        backingMap.get().put(workspace.getId(), workspace);
        pickReferenceUnlessBypassed(workspace);
        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(WorkspaceConfiguration configuration, String id) {
        ensureThreadExistense();
        MemoryWorkspace workspace = newWorkspace(configuration, id);
        backingMap.get().put(id, workspace);
        pickReferenceUnlessBypassed(workspace);
        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(
            WorkspaceConfiguration configuration, String id, Integer deviceId) {
        ensureThreadExistense();
        MemoryWorkspace workspace = newWorkspace(configuration, id, deviceId);
        backingMap.get().put(id, workspace);
        pickReferenceUnlessBypassed(workspace);
        return workspace;
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(
            @NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();
        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = newWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
            pickReferenceUnlessBypassed(workspace);
        }
        return workspace;
    }

    private void pickReferenceUnlessBypassed(MemoryWorkspace workspace) {
        if (getDebugMode() != DebugMode.BYPASS_EVERYTHING) {
            pickReference(workspace);
        }
    }

    @Override
    protected void pickReference(MemoryWorkspace workspace) {
        VulkanRuntime.getInstance().deallocatorService().pickObject(workspace);
    }
}
