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
import org.nd4j.nativeblas.NativeOps;

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

    /**
     * Compute architecture of the device, encoded as {@code major*10 + minor}
     * (e.g. {@code 86} for CUDA sm_86, {@code 89} for sm_89). {@code 0} when not
     * applicable or unknown (CPU backend, or the arch could not be queried).
     *
     * <p>Arch is what makes two GPUs binary-incompatible for compiled kernels
     * (a cubin built for sm_86 will not run on sm_89), so {@link #isCompatibleWith}
     * uses it to keep {@link DevicePlacementPlanner} from spreading arch-specific
     * compiled work onto an incompatible GPU.</p>
     */
    public final int arch;

    // ── Constructors ─────────────────────────────────────────────────────────

    public DeviceKey(DeviceType type, int index) {
        this(type, index, 0);
    }

    public DeviceKey(DeviceType type, int index, int arch) {
        this.type  = type;
        this.index = index;
        this.arch  = arch;
    }

    public DeviceKey(int typeOrdinal, int index) {
        DeviceType[] values = DeviceType.values();
        this.type  = (typeOrdinal >= 0 && typeOrdinal < values.length)
                ? values[typeOrdinal] : DeviceType.UNKNOWN;
        this.index = index;
        this.arch  = 0;
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
            int arch = (type == DeviceType.CUDA) ? queryCudaArch(deviceId) : 0;
            return new DeviceKey(type, deviceId, arch);
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
     * Return a CUDA {@code DeviceKey} for the given GPU index, with its compute
     * architecture queried from the backend so the key carries real sm_XX info.
     */
    public static DeviceKey cuda(int index) {
        return new DeviceKey(DeviceType.CUDA, index, queryCudaArch(index));
    }

    /**
     * Query the CUDA compute architecture ({@code major*10 + minor}) for the given
     * device index. Returns {@code 0} on CPU-only backends or if the query fails.
     */
    public static int queryCudaArch(int index) {
        try {
            NativeOps ops = Nd4j.getNativeOps();
            int major = ops.getDeviceMajor(index);
            int minor = ops.getDeviceMinor(index);
            if (major <= 0) return 0;
            return major * 10 + minor;
        } catch (Throwable t) {
            return 0;
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * Returns the ordinal of {@link #type} for use in native JNI calls.
     */
    public int typeOrdinal() {
        return type.ordinal();
    }

    /**
     * Returns {@code true} if work compiled for this device can run on {@code other}.
     *
     * <p>Devices of different {@link DeviceType} are never compatible. Two CUDA
     * devices are compatible only when they share the same compute architecture
     * ({@link #arch}) — a cubin built for one sm_XX is not binary-compatible with a
     * different one. CPU keys are compatible with any other CPU key. A key whose
     * arch is unknown ({@code arch == 0}) is treated conservatively: it is only
     * compatible with another key whose arch is also unknown.</p>
     */
    public boolean isCompatibleWith(DeviceKey other) {
        if (other == null || type != other.type) return false;
        if (type == DeviceType.CUDA) return arch == other.arch;
        return true;
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
        return arch > 0 ? type.name() + ":" + index + ":sm_" + arch
                        : type.name() + ":" + index;
    }
}
