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

package org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithmReducer;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

@EqualsAndHashCode(exclude = {"lastThreshold", "lastSparsity"})
public class AdaptiveThresholdAlgorithm implements ThresholdAlgorithm {

    private final double initialThreshold;
    private final double minTargetSparsity;
    private final double maxTargetSparsity;
    private final double decayRate;

    private double lastThreshold = Double.NaN;
    private double lastSparsity = Double.NaN;

    public AdaptiveThresholdAlgorithm(double initialThreshold, double minTargetSparsity, double maxTargetSparsity,
                                      double decayRate){
        Preconditions.checkArgument(initialThreshold > 0.0, "Initial threshold must be positive. Got: %s", initialThreshold);
        Preconditions.checkArgument(minTargetSparsity > 0.0 && maxTargetSparsity > 0.0, "Minimum and maximum target " +
                "sparsities must be > 0. Got minTargetSparsity=%s, maxTargetSparsity=%s", minTargetSparsity, maxTargetSparsity );
        Preconditions.checkArgument(minTargetSparsity <= maxTargetSparsity, "Min target sparsity must be less than or equal " +
                "to max target sparsity. Got minTargetSparsity=%s, maxTargetSparsity=%s", minTargetSparsity, maxTargetSparsity );
        Preconditions.checkArgument(decayRate >= 0.5 && decayRate < 1.0, "Decay rate must be a number in range 0.5 (inclusive) to 1.0 (exclusive). " +
                "Usually decay rate is in range 0.95 to 0.999. Got decay rate: %s", decayRate);

        this.initialThreshold = initialThreshold;
        this.minTargetSparsity = minTargetSparsity;
        this.maxTargetSparsity = maxTargetSparsity;
        this.decayRate = decayRate;
    }

    @Override
    public double calculateThreshold(int iteration, int epoch, Double lastThreshold, Boolean lastWasDense,
                                     Double lastSparsityRatio, INDArray updatesPlusResidual) {

        //handle first iteration - use initial threshold
        if(lastThreshold == null && Double.isNaN(this.lastThreshold)){
            return initialThreshold;
        }

        //Check and adapt based on sparsity
        double adaptFromThreshold = (lastThreshold != null ? lastThreshold : this.lastThreshold);
        double prevSparsity;
        if(lastSparsityRatio != null){
            prevSparsity = lastSparsityRatio;
        } else if(lastWasDense != null && lastWasDense){
            prevSparsity = 1.0/16;  //Could be higher, don't know exactly due to dense encoding
        } else if(!Double.isNaN(this.lastSparsity)){
            prevSparsity = this.lastSparsity;
        } else {
            throw new IllegalStateException("Unexpected state: not first iteration but no last sparsity value is available");
        }


        this.lastSparsity = prevSparsity;

        if(prevSparsity >= minTargetSparsity && prevSparsity <= maxTargetSparsity){
            //OK: keep the last threshold unchanged
            return adaptFromThreshold;
        }

        if(prevSparsity < minTargetSparsity ){
            //Sparsity ratio was too small (too sparse) - decrease threshold to increase number of values communicated

            double retThreshold = decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            return retThreshold;
        }

        if(prevSparsity > maxTargetSparsity){
            //Sparsity ratio was too high (too dense) - increase threshold to decrease number of values communicated
            double retThreshold = 1.0/decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            return retThreshold;
        }

        throw new IllegalStateException("Invalid previous sparsity value: " + prevSparsity);        //Should never happen, unless NaN?
    }

    @Override
    public ThresholdAlgorithmReducer newReducer() {
        return new Reducer(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);
    }

    @Override
    public AdaptiveThresholdAlgorithm clone() {
        AdaptiveThresholdAlgorithm ret = new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);
        ret.lastThreshold = lastThreshold;
        ret.lastSparsity = lastSparsity;
        return ret;
    }


    //Reducer stores last threshold between epoch instead of starting adaption from scratch for each epoch
    private static class Reducer implements ThresholdAlgorithmReducer {
        private final double initialThreshold;
        private final double minTargetSparsity;
        private final double maxTargetSparsity;
        private final double decayRate;

        private double lastThresholdSum;
        private double lastSparsitySum;
        private int count;

        private Reducer(double initialThreshold, double minTargetSparsity, double maxTargetSparsity, double decayRate){
            this.initialThreshold = initialThreshold;
            this.minTargetSparsity = minTargetSparsity;
            this.maxTargetSparsity = maxTargetSparsity;
            this.decayRate = decayRate;
        }

        @Override
        public void add(ThresholdAlgorithm instance) {
            AdaptiveThresholdAlgorithm a = (AdaptiveThresholdAlgorithm)instance;
            if(a == null || Double.isNaN(a.lastThreshold))
                return;

            lastThresholdSum += a.lastThreshold;
            lastSparsitySum += a.lastSparsity;
            count++;
        }

        @Override
        public ThresholdAlgorithmReducer merge(ThresholdAlgorithmReducer other) {
            Reducer r = (Reducer) other;
            this.lastThresholdSum += r.lastThresholdSum;
            this.lastSparsitySum += r.lastSparsitySum;
            this.count += r.count;
            return this;
        }

        @Override
        public ThresholdAlgorithm getFinalResult() {
            AdaptiveThresholdAlgorithm ret = new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);
            if(count > 0){
                ret.lastThreshold = lastThresholdSum / count;
                ret.lastSparsity = lastSparsitySum / count;
            }
            return ret;
        }
    }
}
