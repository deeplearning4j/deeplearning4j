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

package org.nd4j.linalg.framework.history;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Event in the lifecycle of an INDArray.
 * 
 * Tracks creation, destruction, and other lifecycle events
 * for debugging and leak detection.
 * 
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ArrayLifecycleEvent {
    
    /**
     * Type of lifecycle event
     */
    public enum EventType {
        CREATED,
        DESTROYED,
        RESHAPED,
        DUPLICATED,
        VIEW_CREATED,
        CONSTANT_MARKED,
        WORKSPACE_ALLOCATED
    }
    
    /**
     * Timestamp when event occurred (milliseconds since epoch)
     */
    private long timestampMs;
    
    /**
     * Event type
     */
    private EventType eventType;
    
    /**
     * Unique ID of the array (from DeallocatorService)
     */
    private long arrayId;
    
    /**
     * Size of array in bytes
     */
    private long sizeBytes;
    
    /**
     * Data type of array
     */
    private String dataType;
    
    /**
     * Shape of array (comma-separated dimensions)
     */
    private String shape;
    
    /**
     * Thread name where event occurred
     */
    private String threadName;
    
    /**
     * Workspace ID if workspace-allocated (null otherwise)
     */
    private String workspaceId;
    
    /**
     * Stack trace at event creation (if enabled)
     */
    private StackTraceElement[] stackTrace;
    
    /**
     * Get formatted stack trace as string
     */
    public String getStackTraceAsString() {
        if (stackTrace == null || stackTrace.length == 0) {
            return "";
        }
        
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement element : stackTrace) {
            sb.append("\tat ").append(element).append("\n");
        }
        return sb.toString();
    }
}
