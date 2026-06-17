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

package org.nd4j.linalg.api.ops.routing;

import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.reduce.floating.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.*;

import java.util.*;

/**
 * Routing policy that selects devices based on operation category.
 * Uses predefined mappings of operation types to preferred device types.
 */
public class CategoryBasedPolicy implements RoutingPolicy {

    /**
     * Operation categories for routing decisions
     */
    public enum OpCategory {
        /** BLAS operations (matmul, gemm) - prefer GPU */
        BLAS,
        /** Element-wise operations - prefer GPU for large arrays */
        ELEMENTWISE,
        /** Reduction operations (sum, mean) - prefer GPU */
        REDUCE,
        /** Index operations (argmax, argmin) - prefer CPU */
        INDEX,
        /** Random operations - either device */
        RANDOM,
        /** Convolution operations - prefer GPU */
        CONVOLUTION,
        /** Recurrent operations - prefer GPU */
        RECURRENT,
        /** Scalar operations - prefer CPU */
        SCALAR,
        /** Unknown/default category */
        UNKNOWN
    }

    private final Map<OpCategory, DeviceType> categoryPreferences;
    private final Map<String, OpCategory> opCategoryOverrides;
    private final long gpuThreshold;  // Minimum elements to prefer GPU

    public CategoryBasedPolicy() {
        this(10000);  // Default threshold
    }

    public CategoryBasedPolicy(long gpuThreshold) {
        this.gpuThreshold = gpuThreshold;
        this.categoryPreferences = new EnumMap<>(OpCategory.class);
        this.opCategoryOverrides = new HashMap<>();

        // Default category preferences
        categoryPreferences.put(OpCategory.BLAS, DeviceType.CUDA_GPU);
        categoryPreferences.put(OpCategory.ELEMENTWISE, DeviceType.CUDA_GPU);
        categoryPreferences.put(OpCategory.REDUCE, DeviceType.CUDA_GPU);
        categoryPreferences.put(OpCategory.INDEX, DeviceType.CPU);
        categoryPreferences.put(OpCategory.RANDOM, DeviceType.CPU);
        categoryPreferences.put(OpCategory.CONVOLUTION, DeviceType.CUDA_GPU);
        categoryPreferences.put(OpCategory.RECURRENT, DeviceType.CUDA_GPU);
        categoryPreferences.put(OpCategory.SCALAR, DeviceType.CPU);
        categoryPreferences.put(OpCategory.UNKNOWN, DeviceType.CPU);
    }

    @Override
    public String getName() {
        return "CategoryBased";
    }

    @Override
    public RoutingDecision selectDevice(Op op, List<DeviceDescriptor> availableDevices) {
        OpCategory category = categorizeOp(op);
        long elementCount = getElementCount(op);
        return selectDeviceForCategory(category, elementCount, availableDevices, op.opName());
    }

    @Override
    public RoutingDecision selectDevice(CustomOp op, List<DeviceDescriptor> availableDevices) {
        OpCategory category = categorizeCustomOp(op);
        long elementCount = getElementCount(op);
        return selectDeviceForCategory(category, elementCount, availableDevices, op.opName());
    }

    private RoutingDecision selectDeviceForCategory(OpCategory category, long elementCount,
                                                     List<DeviceDescriptor> availableDevices, String opName) {
        if (availableDevices == null || availableDevices.isEmpty()) {
            throw new IllegalArgumentException("No available devices");
        }

        DeviceType preferredType = categoryPreferences.get(category);

        // For small operations, prefer CPU regardless of category
        if (elementCount < gpuThreshold && preferredType.isGpu()) {
            preferredType = DeviceType.CPU;
        }

        // Find device of preferred type
        DeviceDescriptor selectedDevice = null;
        for (DeviceDescriptor device : availableDevices) {
            if (device.getDeviceType() == preferredType) {
                selectedDevice = device;
                break;
            }
        }

        // Fallback to any available device
        if (selectedDevice == null) {
            selectedDevice = availableDevices.get(0);
        }

        return RoutingDecision.builder(selectedDevice)
                .reason("category " + category + " prefers " + preferredType +
                        " (elements: " + elementCount + ")")
                .confidence(0.8)
                .build();
    }

    private OpCategory categorizeOp(Op op) {
        // Check overrides first
        String opName = op.opName();
        if (opCategoryOverrides.containsKey(opName)) {
            return opCategoryOverrides.get(opName);
        }

        // Categorize by type
        if (op instanceof ReduceOp) {
            return OpCategory.REDUCE;
        } else if (op instanceof IndexAccumulation) {
            return OpCategory.INDEX;
        } else if (op instanceof ScalarOp) {
            return OpCategory.SCALAR;
        } else if (op instanceof TransformOp) {
            return OpCategory.ELEMENTWISE;
        } else if (op instanceof RandomOp) {
            return OpCategory.RANDOM;
        } else if (op instanceof BroadcastOp) {
            return OpCategory.ELEMENTWISE;
        }

        // Check by name patterns
        String lowerName = opName.toLowerCase();
        if (lowerName.contains("matmul") || lowerName.contains("gemm") ||
                lowerName.contains("mmul") || lowerName.contains("dot")) {
            return OpCategory.BLAS;
        } else if (lowerName.contains("conv")) {
            return OpCategory.CONVOLUTION;
        } else if (lowerName.contains("lstm") || lowerName.contains("gru") ||
                lowerName.contains("rnn")) {
            return OpCategory.RECURRENT;
        }

        return OpCategory.UNKNOWN;
    }

    private OpCategory categorizeCustomOp(CustomOp op) {
        String opName = op.opName();

        // Check overrides
        if (opCategoryOverrides.containsKey(opName)) {
            return opCategoryOverrides.get(opName);
        }

        String lowerName = opName.toLowerCase();

        // Pattern matching for custom ops
        if (lowerName.contains("matmul") || lowerName.contains("gemm") ||
                lowerName.contains("batchedgemm") || lowerName.equals("mmul")) {
            return OpCategory.BLAS;
        } else if (lowerName.contains("conv")) {
            return OpCategory.CONVOLUTION;
        } else if (lowerName.contains("lstm") || lowerName.contains("gru") ||
                lowerName.contains("sru") || lowerName.contains("rnn")) {
            return OpCategory.RECURRENT;
        } else if (lowerName.contains("reduce") || lowerName.contains("sum") ||
                lowerName.contains("mean") || lowerName.contains("max") ||
                lowerName.contains("min") || lowerName.contains("prod")) {
            return OpCategory.REDUCE;
        } else if (lowerName.contains("argmax") || lowerName.contains("argmin") ||
                lowerName.contains("where") || lowerName.contains("topk")) {
            return OpCategory.INDEX;
        } else if (lowerName.contains("random")) {
            return OpCategory.RANDOM;
        }

        return OpCategory.UNKNOWN;
    }

    private long getElementCount(Op op) {
        long count = 0;
        if (op.x() != null) count += op.x().length();
        if (op.y() != null) count += op.y().length();
        if (op.z() != null) count += op.z().length();
        return count;
    }

    private long getElementCount(CustomOp op) {
        long count = 0;
        if (op.inputArguments() != null) {
            for (var input : op.inputArguments()) {
                if (input != null) count += input.length();
            }
        }
        return count;
    }

    /**
     * Set the preferred device type for a category
     * @param category the operation category
     * @param deviceType the preferred device type
     */
    public void setCategoryPreference(OpCategory category, DeviceType deviceType) {
        categoryPreferences.put(category, deviceType);
    }

    /**
     * Override the category for a specific operation
     * @param opName the operation name
     * @param category the category to assign
     */
    public void setOpCategory(String opName, OpCategory category) {
        opCategoryOverrides.put(opName, category);
    }

    /**
     * Get the GPU threshold (minimum elements to prefer GPU)
     * @return threshold
     */
    public long getGpuThreshold() {
        return gpuThreshold;
    }

    @Override
    public int getPriority() {
        return 75;  // Higher than performance, lower than data locality
    }
}
