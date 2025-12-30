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

package org.nd4j.linalg.api.ops.helpers;

import org.nd4j.linalg.api.device.DeviceType;

import java.util.Collections;
import java.util.Set;

/**
 * Descriptor for platform-specific operation helpers (MKLDNN, CUDNN, etc.)
 */
public class PlatformHelperDescriptor {

    private final String helperId;
    private final String helperName;
    private final DeviceType targetDevice;
    private final Set<String> supportedOps;
    private final int priority;
    private final boolean available;

    /**
     * Create a new helper descriptor
     * @param helperId unique identifier
     * @param helperName human-readable name
     * @param targetDevice the device type this helper targets
     * @param supportedOps set of operation names this helper supports
     * @param priority priority for selection (higher = preferred)
     * @param available whether the helper is currently available
     */
    public PlatformHelperDescriptor(String helperId, String helperName, DeviceType targetDevice,
                                    Set<String> supportedOps, int priority, boolean available) {
        this.helperId = helperId;
        this.helperName = helperName;
        this.targetDevice = targetDevice;
        this.supportedOps = Collections.unmodifiableSet(supportedOps);
        this.priority = priority;
        this.available = available;
    }

    /**
     * Get the unique helper identifier
     * @return helper ID
     */
    public String getHelperId() {
        return helperId;
    }

    /**
     * Get the human-readable helper name
     * @return helper name
     */
    public String getHelperName() {
        return helperName;
    }

    /**
     * Get the target device type
     * @return device type
     */
    public DeviceType getTargetDevice() {
        return targetDevice;
    }

    /**
     * Get the set of supported operation names
     * @return supported ops
     */
    public Set<String> getSupportedOps() {
        return supportedOps;
    }

    /**
     * Check if this helper supports a specific operation
     * @param opName the operation name
     * @return true if supported
     */
    public boolean supportsOp(String opName) {
        return supportedOps.contains(opName) || supportedOps.contains("*");
    }

    /**
     * Get the priority for selection
     * @return priority (higher = preferred)
     */
    public int getPriority() {
        return priority;
    }

    /**
     * Check if this helper is available
     * @return true if available
     */
    public boolean isAvailable() {
        return available;
    }

    @Override
    public String toString() {
        return String.format("Helper[%s: %s on %s, priority=%d, available=%s, ops=%d]",
                helperId, helperName, targetDevice, priority, available, supportedOps.size());
    }

    /**
     * Builder for PlatformHelperDescriptor
     */
    public static class Builder {
        private String helperId;
        private String helperName;
        private DeviceType targetDevice;
        private Set<String> supportedOps = Collections.emptySet();
        private int priority = 0;
        private boolean available = true;

        public Builder helperId(String helperId) {
            this.helperId = helperId;
            return this;
        }

        public Builder helperName(String helperName) {
            this.helperName = helperName;
            return this;
        }

        public Builder targetDevice(DeviceType targetDevice) {
            this.targetDevice = targetDevice;
            return this;
        }

        public Builder supportedOps(Set<String> supportedOps) {
            this.supportedOps = supportedOps;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public Builder available(boolean available) {
            this.available = available;
            return this;
        }

        public PlatformHelperDescriptor build() {
            return new PlatformHelperDescriptor(helperId, helperName, targetDevice,
                    supportedOps, priority, available);
        }
    }
}
