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

/**
 * TPU device information holder.
 *
 * NOTE: this class intentionally does NOT implement {@link org.nd4j.linalg.factory.Environment}
 * yet. The original scaffolding implemented an old, much smaller revision of that interface and
 * bit-rotted (the module was unbuildable — see ADR 0102). A real Environment implementation only
 * makes sense once the PJRT native bindings exist to answer environment queries; until then
 * {@link JTpuBackend#getEnvironment()} throws UnsupportedOperationException and this class only
 * reports static TPU device information derived from Cloud TPU VM environment variables.
 *
 * See ADR 0072 (TPU Backend) for the binding plan and TpuBackendSmokeTest for the contract tests.
 */
public class TpuEnvironment {

    private static TpuEnvironment instance;

    private TpuEnvironment() {
        // Private constructor for singleton
    }

    public static synchronized TpuEnvironment getInstance() {
        if (instance == null) {
            instance = new TpuEnvironment();
        }
        return instance;
    }

    /**
     * Get the TPU version string (e.g., "v4", "v5e", "v5p").
     * @return TPU version, or "v4" when not detected
     */
    public String getTpuVersion() {
        return System.getenv().getOrDefault("TPU_VERSION", "v4");
    }

    /**
     * Get the number of TPU cores available.
     * @return Number of TPU cores (default 8 until PJRT bindings can query the device)
     */
    public int getTpuCoreCount() {
        return 8;
    }

    /**
     * Get TPU High Bandwidth Memory (HBM) capacity in bytes for the detected version.
     */
    public long getHbmCapacity() {
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
     * @return true — TPUs are highly optimized for bfloat16
     */
    public boolean preferBfloat16() {
        return true;
    }
}
