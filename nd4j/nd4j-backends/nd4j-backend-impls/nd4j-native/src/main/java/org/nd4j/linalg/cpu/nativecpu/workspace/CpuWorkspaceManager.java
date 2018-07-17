/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.cpu.nativecpu.workspace;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.memory.provider.BasicWorkspaceManager;

/**
 * @author raver119@gmail.com
 */
public class CpuWorkspaceManager extends BasicWorkspaceManager {

    public CpuWorkspaceManager() {
        super();
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(defaultConfiguration);

        backingMap.get().put(workspace.getId(), workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration, id);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace createNewWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String id, Integer deviceId) {
        ensureThreadExistense();

        MemoryWorkspace workspace = new CpuWorkspace(configuration, id, deviceId);

        backingMap.get().put(id, workspace);
        pickReference(workspace);

        return workspace;
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = new CpuWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
            pickReference(workspace);
        }

        return workspace;
    }
}
