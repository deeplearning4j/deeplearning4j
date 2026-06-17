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

import lombok.Builder;
import lombok.Data;
import lombok.Singular;

import java.util.List;

/**
 * Summary statistics produced by {@link RLAlignmentPipeline} after training completes.
 *
 * <p>Contains the per-step loss history and high-level metrics such as total steps,
 * epochs, final loss, and wall-clock training time.  The loss list is recorded at the
 * {@code logEveryNSteps} frequency configured in
 * {@link org.nd4j.autodiff.samediff.config.RLPipelineConfig}.
 *
 * <p>Example usage:
 * <pre>
 * TrainingResult result = pipeline.trainDPO(pairs);
 * System.out.printf("Final loss: %.4f after %d steps in %d ms%n",
 *     result.getFinalLoss(), result.getTotalSteps(), result.getTrainingTimeMs());
 * </pre>
 *
 * @author Adam Gibson
 */
@Data
@Builder
public class TrainingResult {

    /** Total number of gradient-update steps performed across all epochs. */
    private final int totalSteps;

    /** Number of epochs completed. */
    private final int totalEpochs;

    /**
     * Loss values sampled every {@code logEveryNSteps} steps.
     * May be empty if training ends before the first log point.
     */
    @Singular
    private final List<Double> losses;

    /**
     * Loss value at the last recorded log step.
     * Returns {@link Double#NaN} if no steps were recorded.
     */
    private final double finalLoss;

    /** Wall-clock training time in milliseconds. */
    private final long trainingTimeMs;

    /**
     * Compute the average of all recorded losses.
     *
     * @return mean loss, or {@link Double#NaN} if the loss list is empty
     */
    public double averageLoss() {
        if (losses == null || losses.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (double l : losses) {
            sum += l;
        }
        return sum / losses.size();
    }

    /**
     * Return a concise one-line summary string.
     */
    public String summary() {
        return String.format(
                "TrainingResult[steps=%d, epochs=%d, finalLoss=%.6f, avgLoss=%.6f, timeMs=%d]",
                totalSteps, totalEpochs, finalLoss, averageLoss(), trainingTimeMs);
    }
}
