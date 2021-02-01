/*
 *  ******************************************************************************
 *  *
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
 * An adaptive threshold algorithm used to determine the encoding threshold for distributed training.<br>
 * The idea: the threshold can be too high or too low for optimal training - both cases are bad.<br>
 * So instead, we'll define a range of "acceptable" sparsity ratio values (default: 1e-4 to 1e-2).<br>
 * The sparsity ratio is defined as numValues(encodedUpdate)/numParameters<br>
 * <br>
 * If the sparsity ratio falls outside of this acceptable range, we'll either increase or decrease the threshold.<br>
 * The threshold changed multiplicatively using the decay rate:<br>
 * To increase threshold: {@code newThreshold = decayRate * threshold}<br>
 * To decrease threshold: {@code newThreshold = (1.0/decayRate) * threshold}<br>
 * The default decay rate used is {@link #DEFAULT_DECAY_RATE}=0.965936 which corresponds to an a maximum increase or
 * decrease of the threshold by a factor of:<br>
 * * 2.0 in 20 iterations<br>
 * * 100 in 132 iterations<br>
 * * 1000 in 200 iterations<br>
 * <br>
 * <br>
 * A high threshold leads to few values being encoded and communicated - a small "sparsity ratio".<br>
 * Too high threshold (too low sparsity ratio): fast network communication but slow training (few parameter updates being communicated).<br>
 * <br>
 * A low threshold leads to many values being encoded and communicated - a large "sparsity ratio".<br>
 * Too low threshold (too high sparsity ratio): slower network communication and maybe slow training (lots of parameter updates
 * being communicated - but they are all very small, changing network's predictions only a tiny amount).<br>
 * <br>
 * A sparsity ratio of 1.0 means all values are present in the encoded update vector.<br>
 * A sparsity ratio of 0.0 means all values were excluded from the encoded update vector.<br>
 *
 * @author Alex Black
 */
@Slf4j
@EqualsAndHashCode(exclude = {"lastThreshold", "lastSparsity"})
public class AdaptiveThresholdAlgorithm implements ThresholdAlgorithm {
    public static final double DEFAULT_INITIAL_THRESHOLD = 1e-4;
    public static final double DEFAULT_MIN_SPARSITY_TARGET = 1e-4;
    public static final double DEFAULT_MAX_SPARSITY_TARGET = 1e-2;
    public static final double DEFAULT_DECAY_RATE = Math.pow(0.5, (1/20.0));        //Corresponds to increase/decrease by factor of 2 in 20 iterations


    private final double initialThreshold;
    private final double minTargetSparsity;
    private final double maxTargetSparsity;
    private final double decayRate;

    @Getter
    private double lastThreshold = Double.NaN;
    @Getter
    private double lastSparsity = Double.NaN;

    /**
     * Create the adaptive threshold algorithm with the default initial threshold {@link #DEFAULT_INITIAL_THRESHOLD},
     * default minimum sparsity target {@link #DEFAULT_MIN_SPARSITY_TARGET}, default maximum sparsity target {@link #DEFAULT_MAX_SPARSITY_TARGET},
     * and default decay rate {@link #DEFAULT_DECAY_RATE}
     */
    public AdaptiveThresholdAlgorithm(){
        this(DEFAULT_INITIAL_THRESHOLD, DEFAULT_MIN_SPARSITY_TARGET, DEFAULT_MAX_SPARSITY_TARGET, DEFAULT_DECAY_RATE);
    }

    /**
     * Create the adaptive threshold algorithm with the specified initial threshold, but defaults for the other values:
     * default minimum sparsity target {@link #DEFAULT_MIN_SPARSITY_TARGET}, default maximum sparsity target {@link #DEFAULT_MAX_SPARSITY_TARGET},
     * and default decay rate {@link #DEFAULT_DECAY_RATE}
     */
    public AdaptiveThresholdAlgorithm(double initialThreshold){
        this(initialThreshold, DEFAULT_MIN_SPARSITY_TARGET, DEFAULT_MAX_SPARSITY_TARGET, DEFAULT_DECAY_RATE);
    }

    /**
     *
     * @param initialThreshold  The initial threshold to use
     * @param minTargetSparsity The minimum target sparsity ratio - for example 1e-4
     * @param maxTargetSparsity The maximum target sparsity ratio - for example 1e-2
     * @param decayRate         The decay rate. For example 0.95
     */
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

        if(prevSparsity >= minTargetSparsity && prevSparsity <= maxTargetSparsity){
            //OK: keep the last threshold unchanged
            if(log.isDebugEnabled()) {
                log.debug("AdaptiveThresholdAlgorithm: iter {} epoch {}: prev sparsity {}, keeping existing threshold of {}", iteration, epoch, prevSparsity, adaptFromThreshold);
            }
            return adaptFromThreshold;
        }

        if(prevSparsity < minTargetSparsity ){
            //Sparsity ratio was too small (too sparse) - decrease threshold to increase number of values communicated
            double retThreshold = decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            if(log.isDebugEnabled()) {
                log.debug("AdaptiveThresholdAlgorithm: iter {} epoch {}: prev sparsity {} < min sparsity {}, reducing threshold from {} to  {}",
                        iteration, epoch, prevSparsity, minTargetSparsity, adaptFromThreshold, retThreshold);
            }
            return retThreshold;
        }

        if(prevSparsity > maxTargetSparsity){
            //Sparsity ratio was too high (too dense) - increase threshold to decrease number of values communicated
            double retThreshold = 1.0/decayRate * adaptFromThreshold;
            this.lastThreshold = retThreshold;
            if(log.isDebugEnabled()) {
                log.debug("AdaptiveThresholdAlgorithm: iter {} epoch {}: prev sparsity {} > max sparsity {}, increasing threshold from {} to  {}",
                        iteration, epoch, prevSparsity, maxTargetSparsity, adaptFromThreshold, retThreshold);
            }
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

    @Override
    public String toString(){
        String s = "AdaptiveThresholdAlgorithm(initialThreshold=" + initialThreshold + ",minTargetSparsity=" + minTargetSparsity +
                ",maxTargetSparsity=" + maxTargetSparsity + ",decayRate=" + decayRate;
        if(Double.isNaN(lastThreshold)){
            return s + ")";
        }
        return s + ",lastThreshold=" + lastThreshold + ")";
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
            if (!Double.isNaN(a.lastSparsity)) {
                lastSparsitySum += a.lastSparsity;
            }
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
