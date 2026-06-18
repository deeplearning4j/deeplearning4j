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

package org.nd4j.autodiff.samediff.serde;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Data class containing model size information extracted from manifest files.
 * Used for pre-analysis of model loading to enable intelligent device selection
 * and resource allocation.
 *
 * <p>This class provides information about:
 * <ul>
 *   <li>Total model size in bytes</li>
 *   <li>Number of arrays in the model</li>
 *   <li>Size of the largest array (for memory planning)</li>
 *   <li>Per-array size information</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Parse from a manifest
 * Map<String, Pair<Long, Long>> manifest = ...; // varName -> (offset, length)
 * ModelSizeInfo sizeInfo = ModelSizeInfo.fromManifest(manifest);
 *
 * // Check if model fits on a device
 * DeviceDescriptor gpu0 = DeviceDescriptor.cuda(0);
 * if (sizeInfo.fitsOnDevice(gpu0)) {
 *     // Load model to GPU
 * }
 *
 * // Get loading statistics
 * System.out.println("Total model size: " + sizeInfo.getTotalBytes() / (1024*1024) + " MB");
 * System.out.println("Number of arrays: " + sizeInfo.getArrayCount());
 * }</pre>
 *
 * Adam Gibson
 * @see ModelLoadingContext
 */
@Data
@Builder
public class ModelSizeInfo {

    /**
     * Total size of all arrays in bytes
     */
    @Getter
    private final long totalBytes;

    /**
     * Number of arrays in the model
     */
    @Getter
    private final int arrayCount;

    /**
     * Size of the largest individual array in bytes
     */
    @Getter
    private final long largestArrayBytes;

    /**
     * Map of variable names to their sizes in bytes
     */
    @Getter
    private final Map<String, Long> arraySizes;

    /**
     * Creates an empty ModelSizeInfo representing a model with no arrays.
     *
     * @return empty ModelSizeInfo instance
     */
    public static ModelSizeInfo empty() {
        return ModelSizeInfo.builder()
                .totalBytes(0)
                .arrayCount(0)
                .largestArrayBytes(0)
                .arraySizes(Collections.emptyMap())
                .build();
    }

    /**
     * Creates ModelSizeInfo from a manifest map.
     * The manifest maps variable names to (offset, lengthBytes) pairs.
     *
     * @param manifest map of variable names to (offset, length) pairs
     * @return ModelSizeInfo instance with computed statistics
     */
    public static ModelSizeInfo fromManifest(Map<String, Pair<Long, Long>> manifest) {
        if (manifest == null || manifest.isEmpty()) {
            return empty();
        }

        long total = 0;
        long largest = 0;
        Map<String, Long> sizes = new HashMap<>();

        for (Map.Entry<String, Pair<Long, Long>> entry : manifest.entrySet()) {
            String varName = entry.getKey();
            long size = entry.getValue().getSecond(); // Second element is length in bytes

            sizes.put(varName, size);
            total += size;
            largest = Math.max(largest, size);
        }

        return ModelSizeInfo.builder()
                .totalBytes(total)
                .arrayCount(manifest.size())
                .largestArrayBytes(largest)
                .arraySizes(Collections.unmodifiableMap(sizes))
                .build();
    }

    /**
     * Check if the model fits on the specified device using DeviceMemoryManager.
     *
     * @param device the device to check
     * @return true if the device can accommodate the total model size
     */
    public boolean fitsOnDevice(DeviceDescriptor device) {
        return DeviceMemoryManager.getInstance().canAllocate(device, totalBytes);
    }

    /**
     * Check if a single array of the specified name fits on the device.
     *
     * @param device the device to check
     * @param arrayName the name of the array
     * @return true if the device can accommodate the array
     */
    public boolean arrayFitsOnDevice(DeviceDescriptor device, String arrayName) {
        Long size = arraySizes.get(arrayName);
        if (size == null) {
            return true; // Unknown array, assume it fits
        }
        return DeviceMemoryManager.getInstance().canAllocate(device, size);
    }

    /**
     * Get the size of a specific array by name.
     *
     * @param arrayName the variable name
     * @return size in bytes, or 0 if not found
     */
    public long getArraySize(String arrayName) {
        return arraySizes.getOrDefault(arrayName, 0L);
    }

    /**
     * Get total size in megabytes (for logging/display).
     *
     * @return total size in MB
     */
    public double getTotalMegabytes() {
        return totalBytes / (1024.0 * 1024.0);
    }

    /**
     * Get total size in gigabytes (for logging/display).
     *
     * @return total size in GB
     */
    public double getTotalGigabytes() {
        return totalBytes / (1024.0 * 1024.0 * 1024.0);
    }

    /**
     * Calculate the average array size in bytes.
     *
     * @return average array size, or 0 if no arrays
     */
    public long getAverageArrayBytes() {
        return arrayCount > 0 ? totalBytes / arrayCount : 0;
    }

    /**
     * Check if this represents a large model (> 1GB total).
     *
     * @return true if total size exceeds 1GB
     */
    public boolean isLargeModel() {
        return totalBytes > 1024L * 1024L * 1024L;
    }

    /**
     * Check if any individual array is large (> 100MB).
     * Large arrays benefit more from async transfer.
     *
     * @return true if largest array exceeds 100MB
     */
    public boolean hasLargeArrays() {
        return largestArrayBytes > 100L * 1024L * 1024L;
    }

    /**
     * Get a human-readable summary string.
     *
     * @return formatted summary of the model size info
     */
    public String toSummaryString() {
        return String.format("ModelSizeInfo[arrays=%d, total=%.2f MB, largest=%.2f MB, avg=%.2f MB]",
                arrayCount,
                totalBytes / (1024.0 * 1024.0),
                largestArrayBytes / (1024.0 * 1024.0),
                getAverageArrayBytes() / (1024.0 * 1024.0));
    }

    @Override
    public String toString() {
        return toSummaryString();
    }
}
