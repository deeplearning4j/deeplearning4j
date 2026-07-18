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

package org.nd4j.linalg.api.device;

/**
 * Enumeration of compute device types supported by ND4J.
 * Used for device discovery, routing decisions, and backend selection.
 */
public enum DeviceType {
    /**
     * Standard CPU device
     */
    CPU("cpu", true),

    /**
     * NVIDIA CUDA-capable GPU
     */
    CUDA_GPU("cuda", true),

    /**
     * Generic GPU alias for CUDA_GPU
     */
    GPU("cuda", true),

    /**
     * Google TPU (Tensor Processing Unit)
     */
    TPU("tpu", false),

    /**
     * AMD ROCm GPU
     */
    ROCM_GPU("rocm", false),

    /**
     * Apple Metal GPU
     */
    METAL_GPU("metal", false),

    /**
     * Cross-vendor Vulkan compute GPU
     */
    VULKAN_GPU("vulkan", false),

    /**
     * FPGA accelerator
     */
    FPGA("fpga", false),

    /**
     * Remote device (network-attached accelerator)
     */
    REMOTE("remote", false);

    private final String identifier;
    private final boolean fullySupported;

    DeviceType(String identifier, boolean fullySupported) {
        this.identifier = identifier;
        this.fullySupported = fullySupported;
    }

    /**
     * Get the string identifier for this device type
     * @return the identifier string
     */
    public String getIdentifier() {
        return identifier;
    }

    /**
     * Check if this device type is fully supported in ND4J
     * @return true if fully supported
     */
    public boolean isFullySupported() {
        return fullySupported;
    }

    /**
     * Check if this device type is a GPU
     * @return true if GPU type
     */
    public boolean isGpu() {
        return this == CUDA_GPU || this == GPU || this == ROCM_GPU || this == METAL_GPU || this == VULKAN_GPU;
    }

    /**
     * Check if this device type is an accelerator (non-CPU)
     * @return true if accelerator
     */
    public boolean isAccelerator() {
        return this != CPU;
    }

    /**
     * Get DeviceType from string identifier
     * @param identifier the identifier string
     * @return the matching DeviceType
     * @throws IllegalArgumentException if no match found
     */
    public static DeviceType fromIdentifier(String identifier) {
        if (identifier == null) {
            throw new IllegalArgumentException("Identifier cannot be null");
        }
        String lower = identifier.toLowerCase().trim();
        for (DeviceType type : values()) {
            if (type.identifier.equals(lower)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown device type identifier: " + identifier);
    }
}
