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

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.api.device.DeviceType;

import java.util.Collections;
import java.util.Set;

/**
 * Descriptor for platform-specific operation helpers (MKLDNN, CUDNN, etc.)
 */
@Builder
@Getter
public class PlatformHelperDescriptor {

    private final String helperId;
    private final String helperName;
    private final DeviceType targetDevice;
    @Builder.Default
    private final Set<String> supportedOps = Collections.emptySet();
    @Builder.Default
    private final int priority = 0;
    @Builder.Default
    private final boolean available = true;

    /**
     * Check if this helper supports a specific operation
     * @param opName the operation name
     * @return true if supported
     */
    public boolean supportsOp(String opName) {
        return supportedOps.contains(opName) || supportedOps.contains("*");
    }

    @Override
    public String toString() {
        return String.format("Helper[%s: %s on %s, priority=%d, available=%s, ops=%d]",
                helperId, helperName, targetDevice, priority, available, supportedOps.size());
    }
}
