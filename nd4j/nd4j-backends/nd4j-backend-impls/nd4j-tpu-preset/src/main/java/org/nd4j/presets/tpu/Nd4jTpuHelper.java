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

package org.nd4j.presets.tpu;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;

/**
 * Helper class for TPU native bindings.
 *
 * Provides utility methods for TPU operations and memory management.
 */
@Properties(inherit = javacpp.class,
        value = {
                @Platform(define = {"SD_TPU", "HAVE_PJRT"})
        })
public class Nd4jTpuHelper {

    /**
     * Check if PJRT/TPU is available on this system.
     * @return true if TPU runtime is available
     */
    public static boolean isTpuAvailable() {
        try {
            // TODO: Check for PJRT library availability
            String pjrtPath = System.getenv("PJRT_PATH");
            return pjrtPath != null && !pjrtPath.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get the TPU device count.
     * @return number of available TPU devices
     */
    public static int getDeviceCount() {
        // TODO: Query from native PJRT client
        return 0;
    }

    /**
     * Get the TPU version string.
     * @return TPU version (e.g., "v4", "v5")
     */
    public static String getTpuVersion() {
        String version = System.getenv("TPU_VERSION");
        return version != null ? version : "unknown";
    }

    /**
     * Get TPU High Bandwidth Memory capacity in bytes.
     * @return HBM capacity in bytes
     */
    public static long getHbmCapacity() {
        String version = getTpuVersion();
        switch (version) {
            case "v5p":
                return 96L * 1024 * 1024 * 1024; // 96GB
            case "v5e":
                return 16L * 1024 * 1024 * 1024; // 16GB
            case "v4":
                return 32L * 1024 * 1024 * 1024; // 32GB
            default:
                return 0;
        }
    }

    /**
     * Check if bfloat16 is the preferred data type for this TPU.
     * @return true if bfloat16 is preferred
     */
    public static boolean prefersBfloat16() {
        // All TPU versions prefer bfloat16 for neural network workloads
        return true;
    }
}
