/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.api.memory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.nd4j.linalg.api.memory.enums.AllocationKind;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This class provides methods for tracking different memory allocations
 * @author raver119@gmail.com
 */
@Slf4j
public class AllocationsTracker {
    private static final AllocationsTracker INSTANCE = new AllocationsTracker();
    private Map<Integer, DeviceAllocationsTracker> devices = new ConcurrentHashMap<>();

    protected AllocationsTracker() {

    }

    public static AllocationsTracker getInstance() {
        return INSTANCE;
    }

    protected DeviceAllocationsTracker trackerForDevice(Integer deviceId) {
        var tracker = devices.get(deviceId);
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
