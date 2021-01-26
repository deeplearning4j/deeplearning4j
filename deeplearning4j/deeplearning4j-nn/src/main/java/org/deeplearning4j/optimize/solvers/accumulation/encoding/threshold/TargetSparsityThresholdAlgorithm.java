/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithmReducer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Targets a specific sparisty throughout training
 *
 * @author Alex Black
 */
@Slf4j
@EqualsAndHashCode(exclude = {"lastThreshold", "lastSparsity"})
public class TargetSparsityThresholdAlgorithm implements ThresholdAlgorithm {
    public static final double DEFAULT_INITIAL_THRESHOLD = 1e-4;
    public static final double DEFAULT_SPARSITY_TARGET = 1e-3;
    public static final double DEFAULT_DECAY_RATE = Math.pow(0.5, (1/20.0));        //Corresponds to increase/decrease by factor of 2 in 20 iterations


    private final double initialThreshold;
    private final double sparsityTarget;
    private final double decayRate;

    @Getter
    private double lastThreshold = Double.NaN;
    @Getter
    private double lastSparsity = Double.NaN;

    /**
     * Create the adaptive threshold algorithm with the default initial threshold {@link #DEFAULT_INITIAL_THRESHOLD},
     * default sparsity target {@link #DEFAULT_SPARSITY_TARGET} and default decay rate {@link #DEFAULT_DECAY_RATE}
     */
    public TargetSparsityThresholdAlgorithm(){
        this(DEFAULT_INITIAL_THRESHOLD, DEFAULT_SPARSITY_TARGET, DEFAULT_DECAY_RATE);
    }
    /**
     *
     * @param initialThreshold  The initial threshold to use
     * @param sparsityTarget    The sparsity target
     * @param decayRate         The decay rate. For example 0.95
     */
    public TargetSparsityThresholdAlgorithm(double initialThreshold, double sparsityTarget, double decayRate){
        Preconditions.checkArgument(initialThreshold > 0.0, "Initial threshold must be positive. Got: %s", initialThreshold);
        Preconditions.checkState(sparsityTarget > 0.0 && sparsityTarget < 1.0/16, "Sparsity target must be between 0 (exclusive) and 1.0/16 (inclusive), got %s", sparsityTarget);
        Preconditions.checkArgument(decayRate >= 0.5 && decayRate < 1.0, "Decay rate must be a number in range 0.5 (inclusive) to 1.0 (exclusive). " +
                "Usually decay rate is in range 0.95 to 0.999. Got decay rate: %s", decayRate);

        this.initialThreshold = initialThreshold;
        this.sparsityTarget = sparsityTarget;
        this.decayRate = decayRate;
    }

    @Override
    public double calculateThreshold(int iteration, int epoch, Double lastThreshold, Boolean lastWasDense,
                                     Double lastSparsityRatio, INDArray updatesPlusResidual) {

        //handle first iteration - use initial threshold
        if(lastThreshold == null && Double.isNaN(this.lastThreshold)){
            this.lastThreshold = initialThreshold;
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
            throw new IllegalStateException("Unexpected state: not first iteration but no last sparsity value is available: iteration=" +
                    iteration + ", epoch=" + epoch + ", lastThreshold=" + lastThreshold + ", lastWasDense=" + lastWasDense +
                    ", lastSparsityRatio=" + lastSparsityRatio + ", this.lastSparsity=" + this.lastSparsity);
        }


        this.lastSparsity = prevSparsity;

        if(prevSparsity < sparsityTarget ){
            //Sparsity ratio was too small (too sparse) - decrease threshold to increase number of values communicated
            double retThreshold = decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            if(log.isDebugEnabled()) {
                log.debug("TargetSparsityThresholdAlgorithm: iter {} epoch {}: prev sparsity {} < target sparsity {}, reducing threshold from {} to  {}",
                        iteration, epoch, prevSparsity, sparsityTarget, adaptFromThreshold, retThreshold);
            }
            return retThreshold;
        }

        if(prevSparsity > sparsityTarget){
            //Sparsity ratio was too high (too dense) - increase threshold to decrease number of values communicated
            double retThreshold = 1.0/decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            if(log.isDebugEnabled()) {
                log.debug("TargetSparsityThresholdAlgorithm: iter {} epoch {}: prev sparsity {} > max sparsity {}, increasing threshold from {} to  {}",
                        iteration, epoch, prevSparsity, sparsityTarget, adaptFromThreshold, retThreshold);
            }
            return retThreshold;
        }

        //Must be exactly equal
        if(log.isDebugEnabled()) {
            log.debug("TargetSparsityThresholdAlgorithm: keeping existing threshold of {}, previous sparsity {}, target sparsity {}", adaptFromThreshold, prevSparsity, sparsityTarget);
        }
        this.lastThreshold = adaptFromThreshold;
        return adaptFromThreshold;
    }

    @Override
    public ThresholdAlgorithmReducer newReducer() {
        return new Reducer(initialThreshold, sparsityTarget, decayRate);
    }

    @Override
    public TargetSparsityThresholdAlgorithm clone() {
        TargetSparsityThresholdAlgorithm ret = new TargetSparsityThresholdAlgorithm(initialThreshold, sparsityTarget, decayRate);
        ret.lastThreshold = lastThreshold;
        ret.lastSparsity = lastSparsity;
        return ret;
    }

    @Override
    public String toString(){
        String s = "TargetSparsityThresholdAlgorithm(initialThreshold=" + initialThreshold + ",targetSparsity=" + sparsityTarget +
                ",decayRate=" + decayRate;
        if(Double.isNaN(lastThreshold)){
            return s + ")";
        }
        return s + ",lastThreshold=" + lastThreshold + ")";
    }


    //Reducer stores last threshold between epoch instead of starting adaption from scratch for each epoch
    private static class Reducer implements ThresholdAlgorithmReducer {
        private final double initialThreshold;
        private final double targetSparsity;
        private final double decayRate;

        private double lastThresholdSum;
        private double lastSparsitySum;
        private int count;

        private Reducer(double initialThreshold, double targetSparsity, double decayRate){
            this.initialThreshold = initialThreshold;
            this.targetSparsity = targetSparsity;
            this.decayRate = decayRate;
        }

        @Override
        public void add(ThresholdAlgorithm instance) {
            TargetSparsityThresholdAlgorithm a = (TargetSparsityThresholdAlgorithm)instance;
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
            TargetSparsityThresholdAlgorithm ret = new TargetSparsityThresholdAlgorithm(initialThreshold, targetSparsity, decayRate);
            if(count > 0){
                ret.lastThreshold = lastThresholdSum / count;
                ret.lastSparsity = lastSparsitySum / count;
            }
            return ret;
        }
    }
}
