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

package org.nd4j.linalg.hexagon;

import org.nd4j.linalg.factory.Environment;

/**
 * Hexagon NPU-specific environment configuration.
 *
 * Provides Hexagon-related settings and information:
 * - NPU version detection (Hexagon v68, v69, v73, v75)
 * - HVX vector width (128 bytes)
 * - TCM (Tightly Coupled Memory) size (256KB-1MB)
 * - DDR bandwidth reporting
 */
public class HexagonEnvironment implements Environment {

    private static HexagonEnvironment instance;

    private boolean blasEnabled = true;
    private boolean logNativeNDArrayCreation = false;
    private boolean checkOutputChange = false;
    private boolean checkInputChange = false;
    private boolean logNDArrayEvents = false;
    private boolean truncateNDArrayLogStrings = true;
    private int numWorkspaceEventsToKeep = 10;

    private HexagonEnvironment() {
        // Private constructor for singleton
    }

    public static synchronized HexagonEnvironment getInstance() {
        if (instance == null) {
            instance = new HexagonEnvironment();
        }
        return instance;
    }

    @Override
    public boolean isEnableBlas() {
        return blasEnabled;
    }

    @Override
    public void setEnableBlas(boolean enable) {
        this.blasEnabled = enable;
    }

    @Override
    public boolean isLogNativeNDArrayCreation() {
        return logNativeNDArrayCreation;
    }

    @Override
    public void setLogNativeNDArrayCreation(boolean logNativeNDArrayCreation) {
        this.logNativeNDArrayCreation = logNativeNDArrayCreation;
    }

    @Override
    public boolean isCheckOutputChange() {
        return checkOutputChange;
    }

    @Override
    public void setCheckOutputChange(boolean checkOutputChange) {
        this.checkOutputChange = checkOutputChange;
    }

    @Override
    public boolean isCheckInputChange() {
        return checkInputChange;
    }

    @Override
    public void setCheckInputChange(boolean checkInputChange) {
        this.checkInputChange = checkInputChange;
    }

    @Override
    public boolean isLogNDArrayEvents() {
        return logNDArrayEvents;
    }

    @Override
    public void setLogNDArrayEvents(boolean logNDArrayEvents) {
        this.logNDArrayEvents = logNDArrayEvents;
    }

    @Override
    public boolean isTruncateNDArrayLogStrings() {
        return truncateNDArrayLogStrings;
    }

    @Override
    public void setTruncateNDArrayLogStrings(boolean truncate) {
        this.truncateNDArrayLogStrings = truncate;
    }

    @Override
    public int numWorkspaceEventsToKeep() {
        return numWorkspaceEventsToKeep;
    }

    @Override
    public void setNumWorkspaceEventsToKeep(int numWorkspaceEventsToKeep) {
        this.numWorkspaceEventsToKeep = numWorkspaceEventsToKeep;
    }

    /**
     * Get the Hexagon NPU version string (e.g., "v73", "v75").
     * @return Hexagon version or "unknown" if not detected
     */
    public String getNpuVersion() {
        // TODO: Query from native library
        return System.getenv().getOrDefault("HEXAGON_VERSION", "v73");
    }

    /**
     * Get the HVX vector width in bytes.
     * All modern Hexagon NPUs use 128-byte HVX vectors.
     * @return HVX vector width in bytes
     */
    public int getHvxVectorWidth() {
        return 128;
    }

    /**
     * Get TCM (Tightly Coupled Memory) capacity in bytes.
     * TCM is high-bandwidth on-chip memory used for staging data.
     * @return TCM capacity in bytes
     */
    public long getTcmCapacity() {
        String version = getNpuVersion();
        switch (version) {
            case "v75":
                return 1024L * 1024;  // 1MB
            case "v73":
                return 512L * 1024;   // 512KB
            case "v69":
            case "v68":
            default:
                return 256L * 1024;   // 256KB
        }
    }

    /**
     * Get DDR bandwidth estimate in GB/s.
     * @return Estimated DDR bandwidth
     */
    public double getDdrBandwidthGBps() {
        // Typical LPDDR5 bandwidth on Snapdragon SoCs
        return 51.2;
    }

    /**
     * Check if INT8 quantized inference is the preferred mode.
     * Hexagon NPU is highly optimized for INT8 operations.
     * @return true if INT8 should be used by default
     */
    public boolean preferInt8() {
        return true;
    }
}
