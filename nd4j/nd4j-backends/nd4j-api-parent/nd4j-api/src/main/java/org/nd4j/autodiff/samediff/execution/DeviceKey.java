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

import org.nd4j.linalg.factory.Nd4j;

/**
 * Identifies a specific compute device (e.g. a CUDA GPU or CPU backend) for
 * replay-cache operations in {@link BackendPlanManager} and
 * {@link ReplayProfileManager}.
 *
 * <p>A {@code DeviceKey} is a (type, index) pair where {@code type} is an
 * ordinal from {@link DeviceType} and {@code index} is the zero-based device
 * number within that type (e.g. GPU 0, GPU 1, …).</p>
 */
public final class DeviceKey {

    /** Supported device types. */
    public enum DeviceType {
        CPU,
        CUDA,
        MPS,
        OPENCL,
        UNKNOWN;
    }

    /** The type of device. */
    public final DeviceType type;

    /** The zero-based device index within the device type. */
    public final int index;

    // ── Constructors ─────────────────────────────────────────────────────────

    public DeviceKey(DeviceType type, int index) {
        this.type  = type;
        this.index = index;
    }

    public DeviceKey(int typeOrdinal, int index) {
        DeviceType[] values = DeviceType.values();
        this.type  = (typeOrdinal >= 0 && typeOrdinal < values.length)
                ? values[typeOrdinal] : DeviceType.UNKNOWN;
        this.index = index;
    }

    // ── Factory methods ───────────────────────────────────────────────────────

    /**
     * Return a {@code DeviceKey} for the device currently active on this thread.
     *
     * <p>If a CUDA backend is present the current GPU device index is used;
     * otherwise a CPU key with index 0 is returned.</p>
     */
    public static DeviceKey currentDevice() {
        try {
            // Ask Nd4j for the affinity-assigned device for this thread.
            int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            // If the backend is CPU-only, getDeviceForCurrentThread() still
            // returns 0 which maps to CPU device 0.
            boolean hasCuda = Nd4j.getBackend().getClass().getName().contains("Cuda")
                    || Nd4j.getBackend().getClass().getName().contains("CUDA");
            DeviceType type = hasCuda ? DeviceType.CUDA : DeviceType.CPU;
            return new DeviceKey(type, deviceId);
        } catch (Exception e) {
            return new DeviceKey(DeviceType.CPU, 0);
        }
    }

    /**
     * Return a CPU {@code DeviceKey} for the given index.
     */
    public static DeviceKey cpu(int index) {
        return new DeviceKey(DeviceType.CPU, index);
    }

    /**
     * Return a CUDA {@code DeviceKey} for the given GPU index.
     */
    public static DeviceKey cuda(int index) {
        return new DeviceKey(DeviceType.CUDA, index);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * Returns the ordinal of {@link #type} for use in native JNI calls.
     */
    public int typeOrdinal() {
        return type.ordinal();
    }

    // ── Object overrides ──────────────────────────────────────────────────────

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DeviceKey)) return false;
        DeviceKey other = (DeviceKey) o;
        return type == other.type && index == other.index;
    }

    @Override
    public int hashCode() {
        return 31 * type.ordinal() + index;
    }

    @Override
    public String toString() {
        return type.name() + ":" + index;
    }
}
