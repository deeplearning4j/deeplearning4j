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

package org.nd4j.linalg.metal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Placeholder environment class for the Metal / MLX backend.
 *
 * <p>Exposes device discovery stubs that will delegate to the JavaCPP-generated
 * Nd4jMetal bindings once the native preset is complete. At that point this class
 * will implement {@code org.nd4j.linalg.factory.Environment}.</p>
 *
 * <h3>What this will expose (post-bindings)</h3>
 * <ul>
 *   <li>Apple Silicon device count ({@code MTLCreateSystemDefaultDevice} + device array)</li>
 *   <li>GPU family (Apple7 / Apple8 / Apple9 — maps to M1/M2/M3+ feature sets)</li>
 *   <li>Unified memory size (Apple Silicon has no discrete VRAM; RAM is shared)</li>
 *   <li>Metal feature set queries (indirect command buffer support, bfloat16 ops, etc.)</li>
 *   <li>MLX availability flag ({@code mx::metal::is_available()})</li>
 * </ul>
 */
public class MetalEnvironment {

    private static final Logger log = LoggerFactory.getLogger(MetalEnvironment.class);

    private MetalEnvironment() {}

    /**
     * Returns the number of Metal-capable devices on this system.
     * Stub returns 0 until native bindings are generated.
     */
    public static int getMetalDeviceCount() {
        // TODO: delegate to Nd4jMetal.getMetalDeviceCount() when bindings land
        return 0;
    }

    /**
     * Returns the name of the Metal device (e.g. "Apple M3 Pro").
     * Stub returns a placeholder until native bindings are generated.
     *
     * @param deviceId zero-based device index
     * @return device name, or "(Metal bindings not yet available)" if not bound
     */
    public static String getMetalDeviceName(int deviceId) {
        // TODO: delegate to Nd4jMetal.getMetalDeviceName(deviceId)
        return "(Metal bindings not yet available)";
    }

    /**
     * Returns true if MLX is available (mlx library loaded + Metal present).
     * Stub returns false until native bindings are generated.
     */
    public static boolean isMlxAvailable() {
        // TODO: delegate to Nd4jMetal.isMlxAvailable()
        return false;
    }

    /**
     * Returns true if this JVM is running on macOS arm64 (the minimum requirement
     * for Metal / MLX). Does not require native bindings.
     */
    public static boolean isMacOSArm64() {
        String os   = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        return os.contains("mac") && (arch.equals("aarch64") || arch.equals("arm64"));
    }
}
