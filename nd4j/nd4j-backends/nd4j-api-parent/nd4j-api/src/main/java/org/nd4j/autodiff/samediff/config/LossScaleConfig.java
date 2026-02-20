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

package org.nd4j.autodiff.samediff.config;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

/**
 * Configuration for loss scaling in mixed precision training.
 * Loss scaling helps prevent gradient underflow when training with FP16.
 *
 * <p>Three modes are supported:</p>
 * <ul>
 *   <li>{@link Mode#NONE} - No loss scaling (default)</li>
 *   <li>{@link Mode#STATIC} - Fixed loss scale value throughout training</li>
 *   <li>{@link Mode#DYNAMIC} - Automatically adjusts scale based on gradient overflow detection</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>
 * LossScaleConfig config = LossScaleConfig.builder()
 *     .mode(LossScaleConfig.Mode.DYNAMIC)
 *     .initialScale(65536.0)
 *     .growthInterval(2000)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LossScaleConfig {

    /**
     * Loss scaling mode.
     */
    public enum Mode {
        /** No loss scaling applied */
        NONE,
        /** Fixed loss scale value */
        STATIC,
        /** Dynamic loss scaling that adjusts based on gradient overflow */
        DYNAMIC
    }

    /**
     * The loss scaling mode to use.
     * Default: NONE
     */
    @Builder.Default
    private Mode mode = Mode.NONE;

    /**
     * Initial loss scale value.
     * For static mode, this is the fixed scale.
     * For dynamic mode, this is the starting scale.
     * Default: 65536.0 (2^16)
     */
    @Builder.Default
    private double initialScale = 65536.0;

    /**
     * Minimum loss scale for dynamic scaling.
     * The scale will not go below this value.
     * Default: 1.0
     */
    @Builder.Default
    private double minScale = 1.0;

    /**
     * Maximum loss scale for dynamic scaling.
     * The scale will not exceed this value.
     * Default: 65536.0 (2^16)
     */
    @Builder.Default
    private double maxScale = 65536.0;

    /**
     * Factor by which to increase the loss scale when no overflow is detected.
     * Default: 2.0
     */
    @Builder.Default
    private double growthFactor = 2.0;

    /**
     * Factor by which to decrease the loss scale when overflow is detected.
     * Default: 0.5
     */
    @Builder.Default
    private double backoffFactor = 0.5;

    /**
     * Number of consecutive iterations without overflow before increasing scale.
     * Default: 2000
     */
    @Builder.Default
    private int growthInterval = 2000;

    /**
     * Create a static loss scaling configuration with the specified scale.
     *
     * @param scale The fixed loss scale value
     * @return A LossScaleConfig with static mode
     */
    public static LossScaleConfig staticScaling(double scale) {
        return LossScaleConfig.builder()
                .mode(Mode.STATIC)
                .initialScale(scale)
                .build();
    }

    /**
     * Create a dynamic loss scaling configuration with default parameters.
     *
     * @return A LossScaleConfig with dynamic mode and default settings
     */
    public static LossScaleConfig dynamicScaling() {
        return LossScaleConfig.builder()
                .mode(Mode.DYNAMIC)
                .build();
    }

    /**
     * Create a dynamic loss scaling configuration with custom initial scale.
     *
     * @param initialScale The initial loss scale value
     * @return A LossScaleConfig with dynamic mode
     */
    public static LossScaleConfig dynamicScaling(double initialScale) {
        return LossScaleConfig.builder()
                .mode(Mode.DYNAMIC)
                .initialScale(initialScale)
                .build();
    }

    /**
     * Check if loss scaling is enabled.
     *
     * @return true if mode is not NONE
     */
    public boolean isEnabled() {
        return mode != Mode.NONE;
    }

    /**
     * Check if dynamic loss scaling is enabled.
     *
     * @return true if mode is DYNAMIC
     */
    public boolean isDynamic() {
        return mode == Mode.DYNAMIC;
    }
}
