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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Device pinning manager - controls per-variable device placement policies.
 *
 * Accessible via Nd4j.framework.device().pinning()
 *
 * Pin policies:
 * <ul>
 *   <li>STICKY: variable stays on whichever device it is currently on</li>
 *   <li>FOLLOW_THREAD: variable migrates to the current thread's assigned device</li>
 *   <li>EXPLICIT: variable is pinned to a specific device ID</li>
 * </ul>
 *
 * resolveDevice() semantics:
 * <ul>
 *   <li>Returns the target device ID for EXPLICIT and FOLLOW_THREAD policies</li>
 *   <li>Returns -1 for STICKY (meaning "do not migrate, use current placement")</li>
 *   <li>Returns -1 for unpinned variables</li>
 * </ul>
 *
 * @author ND4J Team
 */
@Slf4j
public final class DevicePinningManager {

    private final boolean enabled;
    private final ConcurrentHashMap<String, DevicePinning> pinnings;

    public DevicePinningManager() {
        String prop = System.getProperty(ND4JSystemProperties.DEVICE_PINNING_ENABLED);
        this.enabled = prop != null && Boolean.parseBoolean(prop);
        this.pinnings = new ConcurrentHashMap<>();
    }

    public boolean isEnabled() {
        return enabled;
    }

    /**
     * Pin a variable with STICKY policy (stay on current device).
     */
    public DevicePinningManager pin(String varName) {
        return pin(varName, DevicePinPolicy.STICKY);
    }

    /**
     * Pin a variable with the specified policy.
     */
    public DevicePinningManager pin(String varName, DevicePinPolicy policy) {
        if (varName == null || varName.isEmpty()) {
            return this;
        }

        pinnings.put(varName, DevicePinning.builder()
            .variableName(varName)
            .policy(policy)
            .explicitDeviceId(-1)
            .build());

        if (log.isDebugEnabled()) {
            log.debug("Pinned variable '{}' with policy {}", varName, policy);
        }
        return this;
    }

    /**
     * Pin a variable to an explicit device ID.
     */
    public DevicePinningManager pin(String varName, int deviceId) {
        if (varName == null || varName.isEmpty()) {
            return this;
        }

        pinnings.put(varName, DevicePinning.builder()
            .variableName(varName)
            .policy(DevicePinPolicy.EXPLICIT)
            .explicitDeviceId(deviceId)
            .build());

        if (log.isDebugEnabled()) {
            log.debug("Pinned variable '{}' to device {}", varName, deviceId);
        }
        return this;
    }

    /**
     * Remove pinning for a variable.
     */
    public DevicePinningManager unpin(String varName) {
        if (varName == null || varName.isEmpty()) {
            return this;
        }

        DevicePinning removed = pinnings.remove(varName);
        if (removed != null && log.isDebugEnabled()) {
            log.debug("Unpinned variable '{}'", varName);
        }
        return this;
    }

    /**
     * Resolve target device ID for a variable based on its pin policy.
     *
     * @param varName Variable name
     * @return Target device ID, or -1 if unpinned or STICKY (meaning "don't migrate")
     */
    public int resolveDevice(String varName) {
        if (varName == null || varName.isEmpty()) {
            return -1;
        }

        DevicePinning pinning = pinnings.get(varName);
        if (pinning == null) {
            return -1;
        }

        switch (pinning.getPolicy()) {
            case EXPLICIT:
                return pinning.getExplicitDeviceId();
            case FOLLOW_THREAD:
                try {
                    return org.nd4j.linalg.factory.Nd4j.getAffinityManager().getDeviceForCurrentThread();
                } catch (Exception e) {
                    log.debug("Failed to resolve thread device for FOLLOW_THREAD policy", e);
                    return -1;
                }
            case STICKY:
                // STICKY = don't migrate, use current placement
                return -1;
            default:
                return -1;
        }
    }

    /**
     * Check if a variable is pinned with STICKY policy.
     * Useful for callers that need to distinguish "unpinned" from "sticky".
     */
    public boolean isSticky(String varName) {
        DevicePinning pinning = pinnings.get(varName);
        return pinning != null && pinning.getPolicy() == DevicePinPolicy.STICKY;
    }

    /**
     * Check if migration is allowed for an array.
     * Migration is blocked if:
     * - The variable is STICKY-pinned (should not move)
     * - The array's buffer is registered in a PointerStabilityGuard (frozen for graph replay)
     *
     * @param varName Variable name
     * @param array The array to check
     * @param stabilityGuard Optional stability guard to check frozen state (may be null)
     * @return true if migration is allowed, false if blocked
     */
    public boolean isMigrationAllowed(String varName, INDArray array, PointerStabilityGuard stabilityGuard) {
        if (array == null) {
            return false;
        }

        // Check sticky policy
        if (isSticky(varName)) {
            if (log.isDebugEnabled()) {
                log.debug("Migration blocked for '{}': STICKY policy", varName);
            }
            return false;
        }

        // Check pointer stability (frozen for graph replay)
        if (stabilityGuard != null && stabilityGuard.isFrozen(array)) {
            if (log.isDebugEnabled()) {
                log.debug("Migration blocked for '{}': buffer frozen for graph replay", varName);
            }
            return false;
        }

        return true;
    }

    /**
     * Convenience overload without stability guard.
     */
    public boolean isMigrationAllowed(String varName, INDArray array) {
        return isMigrationAllowed(varName, array, null);
    }

    /**
     * Validate pinnings for a batch of arrays.
     * Returns list of violations (variables that are pinned but on the wrong device,
     * or variables that should migrate but can't because they're frozen).
     *
     * @param arrays Map of variable name to array
     * @return List of violation descriptions
     */
    public List<String> validatePinnings(Map<String, INDArray> arrays) {
        return validatePinnings(arrays, null);
    }

    /**
     * Validate pinnings with stability guard check.
     */
    public List<String> validatePinnings(Map<String, INDArray> arrays, PointerStabilityGuard stabilityGuard) {
        List<String> violations = new ArrayList<>();
        if (arrays == null || arrays.isEmpty()) {
            return violations;
        }

        for (Map.Entry<String, INDArray> entry : arrays.entrySet()) {
            String varName = entry.getKey();
            INDArray array = entry.getValue();
            if (varName == null || array == null) {
                continue;
            }

            DevicePinning pinning = pinnings.get(varName);
            if (pinning == null) {
                continue;
            }

            // Check frozen state
            if (stabilityGuard != null && stabilityGuard.isFrozen(array)) {
                violations.add(String.format(
                    "Variable '%s': buffer frozen for graph replay (policy=%s) — cannot migrate",
                    varName, pinning.getPolicy()));
            }

            // Check device mismatch for EXPLICIT pins
            if (pinning.getPolicy() == DevicePinPolicy.EXPLICIT) {
                try {
                    int currentDevice = org.nd4j.linalg.factory.Nd4j.getAffinityManager().getDeviceForArray(array);
                    if (currentDevice != pinning.getExplicitDeviceId()) {
                        violations.add(String.format(
                            "Variable '%s': pinned to device %d but currently on device %d",
                            varName, pinning.getExplicitDeviceId(), currentDevice));
                    }
                } catch (Exception e) {
                    log.debug("Could not check device for variable '{}'", varName, e);
                }
            }
        }

        return violations;
    }

    public DevicePinning getPinning(String varName) {
        if (varName == null || varName.isEmpty()) {
            return null;
        }
        return pinnings.get(varName);
    }

    public Map<String, DevicePinning> getAllPinnings() {
        return new ConcurrentHashMap<>(pinnings);
    }

    public int getPinnedVariableCount() {
        return pinnings.size();
    }

    public void clear() {
        pinnings.clear();
        if (log.isDebugEnabled()) {
            log.debug("Cleared all pinning records");
        }
    }
}
