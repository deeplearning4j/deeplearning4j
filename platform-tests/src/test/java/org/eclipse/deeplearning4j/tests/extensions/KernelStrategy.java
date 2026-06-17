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

package org.eclipse.deeplearning4j.tests.extensions;

/**
 * Kernel selection strategies.
 */
public enum KernelStrategy {
    FASTEST("fastest", "Select the fastest available kernel"),
    FIRST("first", "Select the first available kernel in priority order"),
    ROUNDROBIN("roundrobin", "Rotate between available kernels"),
    MEMORY("memory", "Select the most memory-efficient kernel"),
    BENCHMARK("benchmark", "Benchmark all kernels and select best");

    private final String id;
    private final String description;

    KernelStrategy(String id, String description) {
        this.id = id;
        this.description = description;
    }

    public String getId() { return id; }
    public String getDescription() { return description; }

    public static KernelStrategy fromString(String str) {
        if (str == null) return FASTEST;
        for (KernelStrategy s : values()) {
            if (s.id.equalsIgnoreCase(str.trim())) {
                return s;
            }
        }
        return FASTEST;
    }
}
