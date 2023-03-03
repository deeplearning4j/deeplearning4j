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
package org.nd4j.linalg.profiler.data;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.AllocationsTracker;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.WorkspaceAllocationsTracker;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Workspace info sample for logging.
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
@Builder
public class WorkspaceInfo {

    private String workspaceName;
    private long allocatedMemory;
    private long spilledBytes;
    private long externalBytes;
    private long pinnedBytes;
    private MemoryKind memoryKind;

    public static WorkspaceInfo sample(String workspaceName,MemoryKind memoryKind) {
       if(workspaceName == null || workspaceName.equals("null") || workspaceName.isEmpty())
           return WorkspaceInfo.builder()
                   .workspaceName(workspaceName)
                   .externalBytes(0)
                   .spilledBytes(0)
                   .pinnedBytes(0)
                   .allocatedMemory(0)
                   .build();
        MemoryWorkspace workspaceForCurrentThread = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceName);
        WorkspaceAllocationsTracker tracker = AllocationsTracker.getInstance().getTracker(workspaceName);

        WorkspaceInfo workspaceInfo = WorkspaceInfo.builder()
                .workspaceName(workspaceName)
                .externalBytes(tracker.currentExternalBytes(memoryKind))
                .spilledBytes(tracker.currentSpilledBytes(memoryKind))
                .pinnedBytes(tracker.currentPinnedBytes(memoryKind))
                .allocatedMemory(workspaceForCurrentThread.getCurrentSize())
                .build();
        return workspaceInfo;

    }

}
