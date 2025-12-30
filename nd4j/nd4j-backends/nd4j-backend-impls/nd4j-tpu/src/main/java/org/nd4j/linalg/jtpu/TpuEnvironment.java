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

package org.nd4j.linalg.jtpu;

import org.nd4j.linalg.factory.Environment;

/**
 * TPU-specific environment configuration.
 *
 * Provides TPU-related settings and information:
 * - TPU version detection (v4, v5)
 * - Memory capacity reporting
 * - Multi-chip topology information
 * - Precision mode configuration (FP32, BF16)
 */
public class TpuEnvironment implements Environment {

    private static TpuEnvironment instance;

    private boolean blasEnabled = true;
    private boolean logNativeNDArrayCreation = false;
    private boolean checkOutputChange = false;
    private boolean checkInputChange = false;
    private boolean logNDArrayEvents = false;
    private boolean truncateNDArrayLogStrings = true;
    private int numWorkspaceEventsToKeep = 10;

    private TpuEnvironment() {
        // Private constructor for singleton
    }

    public static synchronized TpuEnvironment getInstance() {
        if (instance == null) {
            instance = new TpuEnvironment();
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
     * Get the TPU version string (e.g., "v4", "v5").
     * @return TPU version or "unknown" if not detected
     */
    public String getTpuVersion() {
        // TODO: Query from native library
        return System.getenv().getOrDefault("TPU_VERSION", "v4");
    }

    /**
     * Get the number of TPU cores available.
     * @return Number of TPU cores
     */
    public int getTpuCoreCount() {
        // TODO: Query from native library
        return 8; // Default for TPU v4
    }

    /**
     * Get TPU High Bandwidth Memory (HBM) capacity in bytes.
     * @return HBM capacity in bytes
     */
    public long getHbmCapacity() {
        // TPU v4 has 32GB HBM per chip
        // TPU v5e has 16GB HBM per chip
        // TPU v5p has 96GB HBM per chip
        String version = getTpuVersion();
        switch (version) {
            case "v5p":
                return 96L * 1024 * 1024 * 1024;
            case "v5e":
                return 16L * 1024 * 1024 * 1024;
            case "v4":
            default:
                return 32L * 1024 * 1024 * 1024;
        }
    }

    /**
     * Check if bfloat16 is the preferred precision.
     * @return true if bfloat16 should be used by default
     */
    public boolean preferBfloat16() {
        // TPUs are highly optimized for bfloat16
        return true;
    }
}
