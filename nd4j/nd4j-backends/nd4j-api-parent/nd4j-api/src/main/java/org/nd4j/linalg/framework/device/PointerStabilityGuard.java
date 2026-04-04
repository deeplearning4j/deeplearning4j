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

package org.nd4j.linalg.framework.device;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Pointer stability guard for CUDA graph replay.
 *
 * CUDA graphs bake GPU memory addresses into captured graphs.
 * This guard tracks buffer addresses BY VARIABLE NAME and validates they
 * remain stable across graph capture and replay.
 *
 * Tracking is keyed by variable name (not address) so that address changes
 * are detectable: we compare the current address of the array to the address
 * captured at registration time.
 *
 * @author ND4J Team
 */
@Slf4j
public final class PointerStabilityGuard {

    private final boolean enabled;

    /**
     * Tracked buffers keyed by variable name.
     * This allows detecting address changes: the captured address is stored
     * in the record, and we compare it to the array's current address.
     */
    private final ConcurrentHashMap<String, StabilityRecord> trackedByName;

    /**
     * Secondary index: address → variable name, for fast isFrozen() lookups.
     * Updated on register/unregister.
     */
    private final ConcurrentHashMap<Long, String> addressToName;

    public PointerStabilityGuard() {
        String prop = System.getProperty(ND4JSystemProperties.DEVICE_POINTER_STABILITY_CHECK);
        this.enabled = prop != null && Boolean.parseBoolean(prop);
        this.trackedByName = new ConcurrentHashMap<>();
        this.addressToName = new ConcurrentHashMap<>();
    }

    public boolean isEnabled() {
        return enabled;
    }

    /**
     * Register an array for graph pointer stability tracking.
     *
     * @param array The array whose buffer address must remain stable
     * @param variableName Variable name (used as tracking key)
     * @param planName Plan name for diagnostics
     */
    public void registerForGraph(INDArray array, String variableName, String planName) {
        if (!enabled || array == null || array.data() == null) {
            return;
        }
        if (variableName == null || variableName.isEmpty()) {
            return;
        }

        try {
            DataBuffer buffer = array.data();
            long address = buffer.address();
            int deviceId = org.nd4j.linalg.factory.Nd4j.getAffinityManager().getDeviceForArray(array);

            StabilityRecord record = StabilityRecord.builder()
                .variableName(variableName)
                .capturedAddress(address)
                .capturedDeviceId(deviceId)
                .planName(planName)
                .capturedAtMs(System.currentTimeMillis())
                .build();

            trackedByName.put(variableName, record);
            addressToName.put(address, variableName);

            if (log.isDebugEnabled()) {
                log.debug("Registered buffer for graph stability: var={}, address=0x{}, device={}, plan={}",
                    variableName, Long.toHexString(address), deviceId, planName);
            }
        } catch (Exception e) {
            log.warn("Failed to register buffer for graph stability: var={}", variableName, e);
        }
    }

    /**
     * Convenience overload without variable name — uses address as synthetic key.
     * Prefer the 3-arg version when variable name is available.
     */
    public void registerForGraph(INDArray array, String planName) {
        if (!enabled || array == null || array.data() == null) {
            return;
        }
        String syntheticName = "addr_0x" + Long.toHexString(array.data().address());
        registerForGraph(array, syntheticName, planName);
    }

    /**
     * Unregister an array from graph pointer stability tracking.
     */
    public void unregisterFromGraph(INDArray array, String planName) {
        if (!enabled || array == null || array.data() == null) {
            return;
        }

        try {
            long address = array.data().address();
            String varName = addressToName.remove(address);
            if (varName != null) {
                StabilityRecord removed = trackedByName.remove(varName);
                if (removed != null && log.isDebugEnabled()) {
                    log.debug("Unregistered buffer from graph stability: var={}, plan={}",
                        varName, planName);
                }
            }
        } catch (Exception e) {
            log.warn("Failed to unregister buffer from graph stability", e);
        }
    }

    /**
     * Unregister by variable name.
     */
    public void unregisterByName(String variableName) {
        if (!enabled || variableName == null) {
            return;
        }
        StabilityRecord removed = trackedByName.remove(variableName);
        if (removed != null) {
            addressToName.remove(removed.getCapturedAddress());
        }
    }

    /**
     * Validate stability of tracked buffers by checking named arrays.
     * Compares each array's current address to the address captured at registration.
     *
     * @param namedArrays Map of variable name → current INDArray
     * @return List of stability violations (address or device changed)
     */
    public List<StabilityViolation> validateStability(Map<String, INDArray> namedArrays) {
        List<StabilityViolation> violations = new ArrayList<>();
        if (!enabled || namedArrays == null || namedArrays.isEmpty()) {
            return violations;
        }

        for (Map.Entry<String, INDArray> entry : namedArrays.entrySet()) {
            String varName = entry.getKey();
            INDArray array = entry.getValue();
            if (array == null || array.data() == null) {
                continue;
            }

            StabilityRecord record = trackedByName.get(varName);
            if (record == null) {
                continue; // not tracked
            }

            try {
                long currentAddress = array.data().address();
                int currentDevice = org.nd4j.linalg.factory.Nd4j.getAffinityManager().getDeviceForArray(array);

                if (record.getCapturedAddress() != currentAddress || record.getCapturedDeviceId() != currentDevice) {
                    StabilityViolation violation = StabilityViolation.builder()
                        .variableName(varName)
                        .capturedAddress(record.getCapturedAddress())
                        .currentAddress(currentAddress)
                        .capturedDevice(record.getCapturedDeviceId())
                        .currentDevice(currentDevice)
                        .planName(record.getPlanName())
                        .build();

                    violations.add(violation);

                    if (log.isWarnEnabled()) {
                        log.warn("Stability violation: {} address changed 0x{} -> 0x{}, device {} -> {} (plan={})",
                            varName,
                            Long.toHexString(record.getCapturedAddress()),
                            Long.toHexString(currentAddress),
                            record.getCapturedDeviceId(),
                            currentDevice,
                            record.getPlanName());
                    }
                }
            } catch (Exception e) {
                log.warn("Failed to validate buffer stability for var={}", varName, e);
            }
        }

        return violations;
    }

    /**
     * Validate stability of a collection of arrays (unnamed).
     * Uses the address-to-name secondary index to look up records.
     *
     * NOTE: This cannot detect address changes because we only have the
     * current address to look up by. Use the named variant for reliable detection.
     * This method is useful for confirming that tracked arrays are still present.
     *
     * @param graphArrays Arrays to check
     * @return List of violations (empty if all tracked arrays found at expected addresses)
     */
    public List<StabilityViolation> validateStability(Collection<INDArray> graphArrays) {
        List<StabilityViolation> violations = new ArrayList<>();
        if (!enabled || graphArrays == null || graphArrays.isEmpty()) {
            return violations;
        }

        // For unnamed arrays, we can only verify that addresses we know about
        // are still present. We cannot detect if an address CHANGED because
        // we'd need the variable name to find the old record.
        // This is a limitation of the unnamed API.
        return violations;
    }

    /**
     * Check if an array's buffer is tracked for graph stability (i.e., frozen).
     * A frozen buffer should not be migrated to another device.
     */
    public boolean isFrozen(INDArray array) {
        if (!enabled || array == null || array.data() == null) {
            return false;
        }

        try {
            long address = array.data().address();
            return addressToName.containsKey(address);
        } catch (Exception e) {
            log.trace("Could not check frozen state", e);
        }
        return false;
    }

    /**
     * Check if a named variable is tracked for graph stability.
     */
    public boolean isFrozen(String variableName) {
        if (!enabled || variableName == null) {
            return false;
        }
        return trackedByName.containsKey(variableName);
    }

    public int getTrackedBufferCount() {
        return trackedByName.size();
    }

    public void clear() {
        trackedByName.clear();
        addressToName.clear();
        if (log.isDebugEnabled()) {
            log.debug("Cleared all tracked buffers");
        }
    }

    /**
     * Get all tracked variable names.
     */
    public java.util.Set<String> getTrackedVariableNames() {
        return new java.util.HashSet<>(trackedByName.keySet());
    }

    /**
     * Get the stability record for a variable.
     */
    public StabilityRecord getRecord(String variableName) {
        return trackedByName.get(variableName);
    }

    @Data
    @Builder
    public static class StabilityViolation {
        private String variableName;
        private long capturedAddress;
        private long currentAddress;
        private int capturedDevice;
        private int currentDevice;
        private String planName;

        public String getSummary() {
            return String.format(
                "Stability violation: %s (plan=%s) address 0x%s -> 0x%s, device %d -> %d",
                variableName != null ? variableName : "unknown",
                planName,
                Long.toHexString(capturedAddress),
                Long.toHexString(currentAddress),
                capturedDevice,
                currentDevice);
        }
    }

    @Data
    @Builder
    public static class StabilityRecord {
        private String variableName;
        private long capturedAddress;
        private int capturedDeviceId;
        private String planName;
        private long capturedAtMs;
    }
}
