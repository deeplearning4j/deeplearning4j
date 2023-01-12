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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.enums.AllocationKind;
import org.nd4j.linalg.api.memory.enums.MemoryKind;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
public class AllocationsTracker {
    private static final AllocationsTracker INSTANCE = new AllocationsTracker();
    private Map<Integer, DeviceAllocationsTracker> devices = new ConcurrentHashMap<>();


    private Map<String,WorkspaceAllocationsTracker> workspaceAllocationsTracker = new ConcurrentHashMap<>();

    protected AllocationsTracker() {

    }

    /**
     * Returns the tracker for the given workspace id
     * @param workspaceId the id of the workspace to track
     * @return
     */
    public WorkspaceAllocationsTracker getTracker(String workspaceId) {
        return workspaceAllocationsTracker.get(workspaceId);
    }

    /**
     * Register a workspace for tracking
     * @param workspace the id of the workspace to register
     */
    public void registerWorkspace(String workspace) {
        workspaceAllocationsTracker.put(workspace,new WorkspaceAllocationsTracker());
    }


    /**
     * Deregister a workspace to track
     * @param workspace the workspace to delete
     */
    public void deregisterWorkspace(String workspace) {
        workspaceAllocationsTracker.remove(workspace);
    }


    /**
     * The set of tracked devices
     * @return
     */
    public  Set<Integer> trackedDevices() {
        return devices.keySet();
    }

    public static AllocationsTracker getInstance() {
        return INSTANCE;
    }


    public String memoryPerDevice() {
        StringBuilder stringBuilder = new StringBuilder();
        if(devices.isEmpty()) {
            stringBuilder.append("------No device memory found----------\n");
            return stringBuilder.toString();
        }

        devices.forEach((integer, deviceAllocationsTracker) -> {
            stringBuilder.append("Device: " + integer + "\n");
            deviceAllocationsTracker.getBytesMap().forEach((allocationKind, atomicLong) -> {
                stringBuilder.append("Allocation type: " + allocationKind  + " bytes allocated: " + atomicLong.get() + "\n");
            });
        });

        return stringBuilder.toString();
    }

    public String memoryInfo() {
        StringBuilder ret = new StringBuilder();
        ret.append("Javacpp pointer max bytes: " + Pointer.maxBytes() + "\n");
        ret.append("Javacpp pointer max physical bytes: " + Pointer.maxPhysicalBytes() + "\n");
        ret.append("Javacpp available physical bytes: " + Pointer.availablePhysicalBytes() + "\n");
        ret.append("Javacpp max physical bytes " + Pointer.maxPhysicalBytes() + "\n");
        ret.append("Java free memory: " + Runtime.getRuntime().freeMemory() + "\n");
        ret.append("Java max memory: " + Runtime.getRuntime().maxMemory() + "\n");
        return ret.toString();
    }

    /**
     * Prints the memory per workspace including data type and memory kind
     * @return
     */
    public String memoryPerWorkspace() {
        StringBuilder stringBuilder = new StringBuilder();
        if(workspaceAllocationsTracker.isEmpty()) {
            stringBuilder.append("------No workspaces found----------\n");
            return stringBuilder.toString();
        }

        workspaceAllocationsTracker.forEach((s, workspaceAllocationsTracker1) -> {
            stringBuilder.append("-------------Workspace: " + s + "--------------\n");
            Arrays.stream(DataType.values()).forEach(dataType -> {
                if(workspaceAllocationsTracker1.currentDataTypeCount(dataType) > 0) {
                    stringBuilder.append(dataType + " bytes allocated : " + workspaceAllocationsTracker1.currentDataTypeCount(dataType) + "\n");
                }
            });

            Arrays.stream(MemoryKind.values()).forEach(memoryKind -> {
                if(workspaceAllocationsTracker1.currentBytes(memoryKind) > 0) {
                    stringBuilder.append("Memory kind: " + memoryKind + " " + workspaceAllocationsTracker1.currentBytes(memoryKind) + "\n");
                }
            });

            stringBuilder.append("-------------End Workspace: " + s + "--------------\n");
        });

        return stringBuilder.toString();
    }


    protected DeviceAllocationsTracker trackerForDevice(Integer deviceId) {
        DeviceAllocationsTracker tracker = devices.get(deviceId);
        if (tracker == null) {
            synchronized (this) {
                tracker = devices.get(deviceId);
                if (tracker == null) {
                    tracker = new DeviceAllocationsTracker();
                    devices.put(deviceId, tracker);
                }
            }
        }

        return tracker;
    }


    public void markAllocated(AllocationKind kind, Integer deviceId, long bytes) {
        val tracker = trackerForDevice(deviceId);

        tracker.updateState(kind, bytes);
    }

    public void markReleased(AllocationKind kind, Integer deviceId, long bytes) {
        val tracker = trackerForDevice(deviceId);

        tracker.updateState(kind, -bytes);
    }

    public long bytesOnDevice(Integer deviceId) {
        return bytesOnDevice(AllocationKind.GENERAL, deviceId);
    }

    public long bytesOnDevice(AllocationKind kind, Integer deviceId) {
        val tracker = trackerForDevice(deviceId);
        return tracker.getState(kind);
    }
}
