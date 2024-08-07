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
package org.nd4j.linalg.api.memory;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * A meta data object for tracking workspace use.
 * This is used for tracking workspace use
 * for profiling and debugging purposes.
 *
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class WorkspaceUseMetaData {

    private StackTraceElement[] stackTrace;
    private String workspaceName;
    @Builder.Default
    private long eventTime = System.nanoTime();
    private EventTypes eventType;
    private String threadName;
    @Builder.Default
    private long arrayId = -1;
    private long threadId;
    private boolean workspaceActivateAtTimeOfEvent;
    private long generation;
    private long uniqueId;
    private long lastCycleAllocations;
    private long workspaceSize;
    private Enum associatedEnum;
    public enum EventTypes {
        ENTER,
        CLOSE,
        BORROW,
        SET_CURRENT

    }

    /**
     * Returns a meta data instance from the given workspace
     * @param workspace the workspace to get the meta data from
     * @return the meta data from the given workspace
     */
    public static WorkspaceUseMetaData from(MemoryWorkspace workspace) {
        if(workspace == null)
            return empty();
        return builder()
                .associatedEnum(workspace.getAssociatedEnumType())
                .workspaceName(workspace.getId())
                .eventTime(System.nanoTime())
                .workspaceActivateAtTimeOfEvent(workspace.isScopeActive())
                .workspaceSize(workspace.getCurrentSize())
                .generation(workspace.getGenerationId())
                .uniqueId(workspace.getUniqueId())
                .lastCycleAllocations(workspace.getLastCycleAllocations())
                .build();
    }


    public static WorkspaceUseMetaData[] fromArr(MemoryWorkspace workspace) {
        return new WorkspaceUseMetaData[] {from(workspace)};
    }

    /**
     * Returns an empty meta data

     * @return
     */
    public static WorkspaceUseMetaData empty() {
        return WorkspaceUseMetaData.builder().build();
    }

}
