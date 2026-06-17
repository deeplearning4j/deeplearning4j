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

package org.nd4j.autodiff.samediff.execution;

import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Device descriptor for cache management.
 */
public class DeviceKey {
    public enum Type {
        CPU(0), CUDA_GPU(1), METAL_GPU(2), VULKAN_GPU(3),
        OPENCL_GPU(4), TPU(5), ACCELERATOR(6);

        private final int value;
        Type(int value) { this.value = value; }
        public int getValue() { return value; }

        public static Type fromOrdinal(int ordinal) {
            for (Type t : values()) {
                if (t.value == ordinal) return t;
            }
            return CPU;
        }
    }

    public Type type;
    public int index;
    public String archId;

    public DeviceKey(Type type, int index) {
        this.type = type;
        this.index = index;
        this.archId = "";
    }

    public DeviceKey(Type type, int index, String archId) {
        this.type = type;
        this.index = index;
        this.archId = archId;
    }

    public int typeOrdinal() { return type.getValue(); }

    public boolean isCompatibleWith(DeviceKey other) {
        return type == other.type && archId.equals(other.archId);
    }

    public static DeviceKey currentDevice() {
        try {
            DeviceType dt = Nd4j.getBackendDeviceType();
            if (dt != null) {
                switch (dt) {
                    case CUDA_GPU: return new DeviceKey(Type.CUDA_GPU, 0);
                    case METAL_GPU: return new DeviceKey(Type.METAL_GPU, 0);
                    case ROCM_GPU: return new DeviceKey(Type.CUDA_GPU, 0);
                    case TPU: return new DeviceKey(Type.TPU, 0);
                    case FPGA: return new DeviceKey(Type.ACCELERATOR, 0);
                    case REMOTE: return new DeviceKey(Type.ACCELERATOR, 0);
                    default: return new DeviceKey(Type.CPU, 0);
                }
            }
        } catch (Exception e) {
        }
        return new DeviceKey(Type.CPU, 0);
    }

    @Override
    public String toString() {
        return type.name().toLowerCase() + "_" + index + "_" + archId;
    }
}
