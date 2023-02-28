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
package org.nd4j.linalg.profiler.data.eventlogger;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.provider.BasicWorkspaceManager;
import org.nd4j.linalg.profiler.data.RunTimeMemory;
import org.nd4j.linalg.profiler.data.WorkspaceInfo;

/**
 * A log event reflects what a user might want to see in log output.
 * This will include the following:
 * 1. the timestamp for the event
 * 2. the event type (usually allocation or deallocation) {@link EventType}
 * 3. object type (the type of object being acted on like a databuffer or op context): {@link ObjectAllocationType}
 * 4. The associated workspace if any, this is usually {@link BasicWorkspaceManager#getWorkspaceForCurrentThread()}
 * 5. The thread name if any
 * 6. The data type of the object if any (eg: long, int, float,..)
 * 7. The bytes of the object
 *
 * @author Adam Gibson
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class LogEvent {

    private DataType dataType;
    private EventType eventType;
    private ObjectAllocationType objectAllocationType;
    private String threadName;
    private long eventTimeMs;
    private String associatedWorkspace;
    private long bytes;
    private boolean attached;
    private boolean isConstant;
    private long objectId;
    @Builder.Default
    private long opContextId = -1;
    private RunTimeMemory runTimeMemory;
    private WorkspaceInfo workspaceInfo;

    /**
     * Parses a line and converts it to a LogEvent
     * @param line the comma separated line
     * @return
     */
    public static LogEvent eventFromLine(String line) {
        String[] split = line.split(",");
        LogEventBuilder builder = LogEvent.builder();
        builder
                .eventTimeMs(Long.parseLong(split[0]))
                .eventType(EventType.valueOf(split[1]))
                .objectAllocationType(ObjectAllocationType.valueOf(split[2]))
                .associatedWorkspace(split[3])
                .threadName(split[4])
                .dataType(split[5].equals("null") ? DataType.UNKNOWN : DataType.valueOf(split[5]))
                .bytes(Long.parseLong(split[6]))
                .attached(Boolean.parseBoolean(split[7]))
                .isConstant(Boolean.parseBoolean(split[8]))
                .objectId(Long.parseLong(split[9]))
                .workspaceInfo(WorkspaceInfo.builder()
                        .allocatedMemory(Long.parseLong(split[10]))
                        .externalBytes(Long.parseLong(split[11]))
                        .pinnedBytes(Long.parseLong(split[12]))
                        .spilledBytes(Long.parseLong(split[13]))
                        .build())
                .runTimeMemory(RunTimeMemory.builder()
                        .runtimeFreeMemory(Long.parseLong(split[14]))
                        .javacppAvailablePhysicalBytes(Long.parseLong(split[15]))
                        .javaCppMaxPhysicalBytes(Long.parseLong(split[16]))
                        .javacppMaxBytes(Long.parseLong(split[17]))
                        .runtimeMaxMemory(Long.parseLong(split[18]))
                        .build())
                .build();
        return builder.build();
    }
}
