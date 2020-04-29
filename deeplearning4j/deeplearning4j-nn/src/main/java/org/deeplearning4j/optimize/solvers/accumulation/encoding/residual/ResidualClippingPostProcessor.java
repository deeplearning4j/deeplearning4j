/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.optimize.solvers.accumulation.encoding.residual;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ResidualPostProcessor;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Residual clipping post processor clips the values of a residual every N iterations as follows:<br>
 * For residual vector R, and C = thresholdMultipleClipValue, T is the current encoding threshold<br>
 * {@code R[i] =  C*T} if R[i] > C*T<br>
 * {@code R[i] = -C*T} if R[i] < -C*T<br>
 * {@code R[i]} is unmodified otherwise<br>
 * <br>
 * Note: Regarding the frequency, a value around 5 is suggested as a good balance between applying frequently enough,
 * and minimizing the computational overhead. Very infrequent clipping might allow stale updates to be communicated
 * (if that is a problem at all), whereas very frequent clipping (every iteration) may have a much higher overhead
 * relative to the benefit, compared to less frequent applications.<br>
 * <br>
 * The motivation here for specifying the clipping value C in terms of a multiple of the threshold is simple: if there
 * were no new updates, and the threshold didn't change, it would take C steps to communicate the current residual.
 *
 * @author Alex Black
 */
@Slf4j
public class ResidualClippingPostProcessor implements ResidualPostProcessor {

    private final double thresholdMultipleClipValue;
    private final int frequency;

    /**
     *
     * @param thresholdMultipleClipValue The multiple of the current threshold to use for clipping. A value of C means
     *                                   that the residual vector will be clipped to the range [-C*T, C*T] for the current
     *                                   threshold T
     * @param frequency                  Frequency with which to apply the clipping
     */
    public ResidualClippingPostProcessor(double thresholdMultipleClipValue, int frequency) {
        Preconditions.checkState(thresholdMultipleClipValue >= 1.0, "Threshold multiple must be a positive value and " +
                "greater than 1.0 (1.0 means clip at 1x the current threshold)");
        this.thresholdMultipleClipValue = thresholdMultipleClipValue;
        this.frequency = frequency;
    }

    @Override
    public void processResidual(int iteration, int epoch, double lastThreshold, INDArray residualVector) {
        if(iteration > 0 && iteration % frequency == 0) {
            double currClip = lastThreshold * thresholdMultipleClipValue;
            //TODO replace with single op once we have GPU version
            BooleanIndexing.replaceWhere(residualVector, currClip, Conditions.greaterThan(currClip));
            BooleanIndexing.replaceWhere(residualVector, -currClip, Conditions.lessThan(-currClip));
            log.debug("Applied residual clipping: iter={}, epoch={}, lastThreshold={}, multiple={}, clipValue={}", iteration, epoch, lastThreshold, thresholdMultipleClipValue, currClip);
        }
    }

    @Override
    public ResidualClippingPostProcessor clone() {
        return new ResidualClippingPostProcessor(thresholdMultipleClipValue, frequency);
    }
}
