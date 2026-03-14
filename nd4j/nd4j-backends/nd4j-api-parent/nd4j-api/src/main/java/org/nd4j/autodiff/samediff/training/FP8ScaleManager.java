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

package org.nd4j.autodiff.samediff.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.FP8TrainingConfig;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages per-tensor FP8 scale factors for mixed precision training.
 *
 * For each tensor, tracks a rolling window of absolute maximum (amax) values.
 * The scale factor is computed as: scale = fp8_max / max(amax_history)
 *
 * FP8 E4M3 max = 448.0
 * FP8 E5M2 max = 57344.0
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
public class FP8ScaleManager {

    public static final double FP8_E4M3_MAX = 448.0;
    public static final double FP8_E5M2_MAX = 57344.0;

    @Getter
    private final FP8TrainingConfig config;

    // Per-tensor amax histories
    private final Map<String, Deque<Double>> amaxHistories = new ConcurrentHashMap<>();

    // Per-tensor current scales
    private final Map<String, Double> currentScales = new ConcurrentHashMap<>();

    public FP8ScaleManager(FP8TrainingConfig config) {
        this.config = config;
    }

    /**
     * Update the amax for a specific tensor and recompute its scale.
     *
     * @param tensorName The tensor identifier
     * @param amax       The absolute maximum value observed in the current step
     * @param isForward  Whether this is for forward pass (E4M3) or backward (E5M2)
     */
    public void updateAmax(String tensorName, double amax, boolean isForward) {
        Deque<Double> history = amaxHistories.computeIfAbsent(tensorName,
                k -> new ArrayDeque<>(config.getAmaxHistoryLength()));

        // Add to history, keeping only the last N entries
        if (history.size() >= config.getAmaxHistoryLength()) {
            history.removeFirst();
        }
        history.addLast(amax);

        // Recompute scale
        double scale = computeScale(history, isForward);
        currentScales.put(tensorName, scale);
    }

    /**
     * Get the current scale factor for a tensor.
     *
     * @param tensorName The tensor identifier
     * @return The scale factor (defaults to 1.0 if not yet computed)
     */
    public double getScale(String tensorName) {
        return currentScales.getOrDefault(tensorName, 1.0);
    }

    /**
     * Get the inverse scale (for descaling after computation).
     */
    public double getInverseScale(String tensorName) {
        double scale = getScale(tensorName);
        return scale > 0 ? 1.0 / scale : 1.0;
    }

    /**
     * Compute the scale factor from amax history.
     * scale = fp8_max / max(amax_history)
     */
    private double computeScale(Deque<Double> history, boolean isForward) {
        if (history.isEmpty()) {
            return 1.0;
        }

        double maxAmax = 0.0;
        for (double val : history) {
            maxAmax = Math.max(maxAmax, val);
        }

        if (maxAmax == 0.0) {
            return 1.0;
        }

        double fp8Max = isForward ? FP8_E4M3_MAX : FP8_E5M2_MAX;
        return fp8Max / maxAmax;
    }

    /**
     * Clear all tracked state. Call when switching models or resetting training.
     */
    public void clear() {
        amaxHistories.clear();
        currentScales.clear();
    }

    /**
     * Get the number of tensors being tracked.
     */
    public int getTrackedTensorCount() {
        return amaxHistories.size();
    }
}
