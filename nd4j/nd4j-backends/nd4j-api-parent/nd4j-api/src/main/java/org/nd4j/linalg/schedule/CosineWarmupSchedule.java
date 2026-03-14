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

package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Cosine annealing schedule with linear warmup.
 *
 * During warmup (iteration &lt; warmupSteps):
 *   lr = maxLR * (iteration / warmupSteps)
 *
 * After warmup:
 *   lr = minLR + 0.5 * (maxLR - minLR) * (1 + cos(pi * progress))
 *   where progress = (iteration - warmupSteps) / (totalSteps - warmupSteps)
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Data
public class CosineWarmupSchedule implements ISchedule {

    private final double maxLR;
    private final double minLR;
    private final int warmupSteps;
    private final int totalSteps;

    /**
     * @param maxLR       Peak learning rate (reached at end of warmup)
     * @param minLR       Minimum learning rate (reached at end of training)
     * @param warmupSteps Number of warmup steps with linear ramp
     * @param totalSteps  Total number of training steps
     */
    public CosineWarmupSchedule(@JsonProperty("maxLR") double maxLR,
                                @JsonProperty("minLR") double minLR,
                                @JsonProperty("warmupSteps") int warmupSteps,
                                @JsonProperty("totalSteps") int totalSteps) {
        this.maxLR = maxLR;
        this.minLR = minLR;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;
    }

    /**
     * Convenience constructor with minLR = 0.
     */
    public CosineWarmupSchedule(double maxLR, int warmupSteps, int totalSteps) {
        this(maxLR, 0.0, warmupSteps, totalSteps);
    }

    /**
     * Create from warmup ratio (e.g., 0.1 = 10% of steps are warmup).
     */
    public static CosineWarmupSchedule fromRatio(double maxLR, double minLR,
                                                  double warmupRatio, int totalSteps) {
        int warmup = (int) (warmupRatio * totalSteps);
        return new CosineWarmupSchedule(maxLR, minLR, warmup, totalSteps);
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        if (iteration < warmupSteps) {
            // Linear warmup
            return maxLR * ((double) iteration / warmupSteps);
        }

        if (iteration >= totalSteps) {
            return minLR;
        }

        // Cosine decay
        double progress = (double) (iteration - warmupSteps) / (totalSteps - warmupSteps);
        return minLR + 0.5 * (maxLR - minLR) * (1.0 + Math.cos(Math.PI * progress));
    }

    @Override
    public ISchedule clone() {
        return new CosineWarmupSchedule(maxLR, minLR, warmupSteps, totalSteps);
    }
}
