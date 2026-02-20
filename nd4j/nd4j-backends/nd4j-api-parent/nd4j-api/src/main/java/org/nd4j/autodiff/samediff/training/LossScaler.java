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
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Handles loss scaling for mixed precision training.
 *
 * <p>Loss scaling multiplies the loss value before the backward pass to prevent
 * gradient underflow when using FP16. After computing gradients, they are unscaled
 * (divided by the loss scale) before applying parameter updates.</p>
 *
 * <p>Dynamic loss scaling automatically adjusts the scale based on whether
 * gradient overflow (inf/nan) is detected:</p>
 * <ul>
 *   <li>If overflow is detected, the scale is reduced by the backoff factor</li>
 *   <li>If no overflow for a number of consecutive iterations, the scale is increased</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * LossScaleConfig config = LossScaleConfig.dynamicScaling();
 * LossScaler scaler = new LossScaler(config);
 *
 * // During training:
 * INDArray scaledLoss = scaler.scaleLoss(loss);
 * // ... backward pass ...
 * scaler.unscaleGradients(gradients);
 * if (scaler.areGradientsFinite(gradients)) {
 *     // Apply updates
 *     scaler.update(true);
 * } else {
 *     // Skip update
 *     scaler.update(false);
 * }
 * </pre>
 *
 * @author Adam Gibson
 */
@Slf4j
public class LossScaler {

    private final LossScaleConfig config;

    @Getter
    private double currentScale;

    private int consecutiveFiniteCount;

    /**
     * Create a LossScaler with the given configuration.
     *
     * @param config The loss scale configuration
     */
    public LossScaler(LossScaleConfig config) {
        this.config = config;
        this.currentScale = config.getInitialScale();
        this.consecutiveFiniteCount = 0;
    }

    /**
     * Scale the loss value for backward pass.
     *
     * @param loss The original loss value
     * @return The scaled loss (loss * currentScale)
     */
    public INDArray scaleLoss(INDArray loss) {
        if (!config.isEnabled()) {
            return loss;
        }
        return loss.mul(currentScale);
    }

    /**
     * Scale a loss value (scalar).
     *
     * @param loss The original loss value
     * @return The scaled loss (loss * currentScale)
     */
    public double scaleLoss(double loss) {
        if (!config.isEnabled()) {
            return loss;
        }
        return loss * currentScale;
    }

    /**
     * Unscale gradients after backward pass.
     * This divides the gradients by the current scale to restore them to
     * their true magnitude.
     *
     * @param gradients The scaled gradients (modified in-place)
     */
    public void unscaleGradients(INDArray gradients) {
        if (!config.isEnabled() || currentScale == 1.0) {
            return;
        }
        gradients.divi(currentScale);
    }

    /**
     * Check if all gradient values are finite (not inf or nan).
     *
     * @param gradients The gradients to check
     * @return true if all values are finite, false if any inf/nan detected
     */
    public boolean areGradientsFinite(INDArray gradients) {
        // Check for NaN by comparing sum to itself (NaN != NaN)
        double sum = gradients.sumNumber().doubleValue();
        if (Double.isNaN(sum) || Double.isInfinite(sum)) {
            return false;
        }

        // Additional check: verify max/min are finite
        double max = gradients.maxNumber().doubleValue();
        double min = gradients.minNumber().doubleValue();
        if (Double.isNaN(max) || Double.isInfinite(max) ||
            Double.isNaN(min) || Double.isInfinite(min)) {
            return false;
        }

        return true;
    }

    /**
     * Unscale gradients and check if they are finite.
     * This is a convenience method that combines unscaleGradients and areGradientsFinite.
     *
     * @param gradients The scaled gradients (modified in-place)
     * @return true if gradients are finite after unscaling, false otherwise
     */
    public boolean unscaleGradientsAndCheck(INDArray gradients) {
        unscaleGradients(gradients);
        return areGradientsFinite(gradients);
    }

    /**
     * Update the loss scale based on whether gradients were finite.
     * For dynamic scaling:
     * - If gradients were finite, increment consecutive count and possibly increase scale
     * - If gradients overflowed, reset count and decrease scale
     *
     * @param gradientsFinite true if gradients were finite, false if overflow detected
     */
    public void update(boolean gradientsFinite) {
        if (!config.isDynamic()) {
            return;
        }

        if (gradientsFinite) {
            consecutiveFiniteCount++;

            // Check if we should increase the scale
            if (consecutiveFiniteCount >= config.getGrowthInterval()) {
                double newScale = currentScale * config.getGrowthFactor();
                if (newScale <= config.getMaxScale()) {
                    currentScale = newScale;
                    log.debug("Loss scale increased to {}", currentScale);
                }
                consecutiveFiniteCount = 0;
            }
        } else {
            // Overflow detected - reduce scale
            double newScale = currentScale * config.getBackoffFactor();
            if (newScale >= config.getMinScale()) {
                currentScale = newScale;
                log.debug("Loss scale decreased to {} due to gradient overflow", currentScale);
            } else {
                log.warn("Loss scale at minimum ({}) but still seeing overflow", config.getMinScale());
            }
            consecutiveFiniteCount = 0;
        }
    }

    /**
     * Reset the loss scaler to its initial state.
     */
    public void reset() {
        this.currentScale = config.getInitialScale();
        this.consecutiveFiniteCount = 0;
    }

    /**
     * Get the configuration for this loss scaler.
     *
     * @return The loss scale configuration
     */
    public LossScaleConfig getConfig() {
        return config;
    }

    /**
     * Get the number of consecutive iterations with finite gradients.
     *
     * @return The consecutive finite count
     */
    public int getConsecutiveFiniteCount() {
        return consecutiveFiniteCount;
    }
}
