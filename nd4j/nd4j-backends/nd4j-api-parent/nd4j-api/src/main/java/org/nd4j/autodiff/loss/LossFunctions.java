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

package org.nd4j.autodiff.loss;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;

/**
 * SameDiff loss functions
 *
 * @author Alex Black
 */
@Deprecated
public class LossFunctions {

    private static final int[] SCALAR = new int[]{1,1};

    public enum Reduction {
        /**
         * No reduction. Output is the same shape as the predictions/labels.
         * Weights (if any) are applied. Dimension args are ignored.
         * Example: 2d input, MSE.
         * Output: sqDiff(predictions,labels) -> shape same as input/labels
         */
        NONE,
        /**
         * Reduce as normal along the specified dimensions, but don't sum/mean etc the remaining
         * dimensions.
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mean(weights * sqDiff(predictions,labels),1) -> shape [dim0,1]
         */
        SPECIFIED_DIMS,
        /**
         * Sum across the remaining dimensions, returning a scalar
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex)
         */
        SUM,
        /**
         * Weighted mean: sum(weights * loss) / sum(weights)
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex) / sum(weights)
         *
         * NOTE: if weights array is not provided, then weights default to (effectively) 1.0 for all entries - and hence
         * MEAN_BY_WEIGHT is equivalent to SUM (as sum(1.0) = 1.0)
         */
        MEAN_BY_WEIGHT,

        /**
         * Weighted mean: sum(weights * loss) / count(weights != 0)
         * Example: 2d input, MSE loss along dimension 1.
         * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)          *Same as SPECIFIED_DIMS*
         *         output = sum(mse_per_ex) / count(weights != 0)
         *
         * NOTE: if weights array is not provided, then weights default to scalar 1.0 and hence MEAN_BY_COUNT
         * is equivalent to
         */
        MEAN_BY_COUNT

    }

    private LossFunctions(){ }


    private static LossInfo.Builder validate(String lossName, SDVariable predictions, SDVariable label, Reduction reduction){
        Preconditions.checkNotNull(predictions, "Predictions variable cannot be null for loss function - %s", lossName);
        Preconditions.checkNotNull(label, "Label variable cannot be null for loss function - %s", lossName);
        Preconditions.checkNotNull(reduction, "Reduction enumeration cannot be null for loss function - %s", lossName);
        return LossInfo.builder()
                .lossName(lossName)
                .reduction(reduction)
                .label(label)
                .predictions(predictions);
    }


    /**
     * Mean squared error: L = mean( (predicted - label)^2)
     *
     * @param outputName  Name of the output SDVariable
     * @param predictions Predictions variable
     * @param label       Label variable
     * @param weights     Weights array. May be null, or any broadcastable shape (with predictions/label arrays).
     *                    Note that this is also used for masking (weight of 0 = 'masked out')
     * @param reduction   Type of reduction to perform for the loss function
     * @param dimensions  Dimension(s) to apply the loss function on
     * @return LossInfo - bean with variables etc for the loss function
     */
    public static LossInfo mse(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                               Reduction reduction, int... dimensions){
        LossInfo.Builder b = validate("mse", predictions, label, reduction);
        SameDiff sd = predictions.getSameDiff();

        if(weights == null){
            weights = sd.one("mse_loss_weights", SCALAR);
        }

        SDVariable diff = predictions.sub(label);
        String name = (reduction == Reduction.NONE ? outputName : null);
        SDVariable preReduceLoss = sd.square(diff).mul(name, weights);

        return doReduce(sd, outputName, true, b, reduction, preReduceLoss, label, weights, dimensions);
    }


    /**
     * L1 loss - sum of absolute errors. L = sum_i abs(predicted_i - actual_i)
     *
     * @param outputName
     * @param predictions
     * @param label
     * @param weights
     * @param reduction
     * @param dimensions
     * @return
     */
    public static LossInfo l1(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                              Reduction reduction, int... dimensions){
        LossInfo.Builder b = validate("l1", predictions, label, reduction);
        SameDiff sd = predictions.getSameDiff();

        if(weights == null){
            weights = sd.one("l1_loss_weights", SCALAR);
        }

        String name = (reduction == Reduction.NONE ? outputName : null);
        SDVariable preReduceLoss = sd.abs(predictions.sub(label)).mul(name, weights);

        return doReduce(sd, outputName, false, b, reduction, preReduceLoss, label, weights, dimensions);
    }


    /**
     * L2 loss function: i.e., sum of squared errors, L = sum_i (actual_i - predicted)^2
     *
     * @param outputName
     * @param predictions
     * @param label
     * @param weights
     * @param reduction
     * @param dimensions
     * @return
     */
    public static LossInfo l2(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                              Reduction reduction, int... dimensions){
        LossInfo.Builder b = validate("l2", predictions, label, reduction);
        SameDiff sd = predictions.getSameDiff();

        if(weights == null){
            weights = sd.one("l2_loss_weights", SCALAR);
        }

        SDVariable diff = predictions.sub(label);
        String name = (reduction == Reduction.NONE ? outputName : null);
        SDVariable preReduceLoss = sd.square(diff).mul(name, weights);

        return doReduce(sd, outputName, false, b, reduction, preReduceLoss, label, weights, dimensions);
    }

    public static LossInfo negativeLogLikelihood(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                                                 Reduction reduction, int... dimensions){
        return mcxent(outputName, predictions, label, weights, reduction, dimensions);
    }

    /**
     * Multi-Class Cross Entropy loss function:<br>
     * L = sum_i actual_i * log( predicted_i )
     *
     * @param outputName
     * @param predictions
     * @param label
     * @param weights
     * @param reduction
     * @param dimensions
     * @return
     */
    public static LossInfo mcxent(String outputName, SDVariable predictions, SDVariable label, SDVariable weights,
                                  Reduction reduction, int... dimensions){
        LossInfo.Builder b = validate("mcxent", predictions, label, reduction);
        SameDiff sd = predictions.getSameDiff();

        if(weights == null){
            weights = sd.one("mcxent_loss_weights", SCALAR);
        }

        String name = (reduction == Reduction.NONE ? outputName : null);
        SDVariable weightedLogProd = sd.log(predictions).mul(label).mul(name, weights);

        return doReduce(sd, outputName, false, b, reduction, weightedLogProd, label, weights, dimensions);
    }


    /**
     * Determine the number of weight entries that are non-zero, after broadcasting
     *
     * @param weights
     * @param labels
     * @return
     */
    private static SDVariable nonZeroCount(SDVariable weights, SDVariable labels){
        SameDiff sd = weights.getSameDiff();

        SDVariable present = sd.neq(weights, 0.0);
        SDVariable presentBroadcast = sd.zerosLike(labels).add(present);

        return sd.sum(presentBroadcast);
    }

    /**
     * Perform the final reduction on the loss function
     *
     * @param sd
     * @param outputName
     * @param isMean
     * @param b
     * @param reduction
     * @param preReduceLoss
     * @param label
     * @param weights
     * @param dimensions
     * @return
     */
    private static LossInfo doReduce(SameDiff sd, String outputName, boolean isMean, LossInfo.Builder b, Reduction reduction,
                          SDVariable preReduceLoss, SDVariable label, SDVariable weights, int[] dimensions){
        switch (reduction){
            case NONE:
                //Return same shape as predictions/labels
                b.loss(preReduceLoss);
                break;
            case SPECIFIED_DIMS:
                //Reduce along specified dimensions
                if(isMean){
                    //Example: MSE + mean along examples
                    b.loss(sd.mean(outputName, preReduceLoss, dimensions));
                } else {
                    //Example: L1 loss (sum) + mean along examples
                    b.loss(sd.sum(outputName, preReduceLoss, dimensions));
                }
            case SUM:
                if(isMean){
                    //Example: MSE (mean) + sum along examples
                    SDVariable m = sd.mean(preReduceLoss, dimensions);
                    b.loss(sd.sum(outputName, m));
                } else {
                    //Example: L1 loss (sum) + sum along examples -> sum along all dimensions
                    b.loss(sd.sum(outputName, preReduceLoss));
                }
                break;
            case MEAN_BY_WEIGHT:
                SDVariable weightSum = sd.sum(weights);
                if(isMean){
                    //Example: MSE (mean) + mean by weights over examples
                    //reduce along dims + reduce along remaining dims == reduce along *all* dims
                    SDVariable m2 = sd.mean(preReduceLoss);
                    b.loss(m2.div(outputName, weightSum));
                } else {
                    //Example: L1 (sum) + mean by weights over examples
                    SDVariable sum = sd.sum(preReduceLoss, dimensions);
                    b.loss(sum.div(outputName, weightSum));
                }
                break;
            case MEAN_BY_COUNT:
                SDVariable nonZeroWeights = nonZeroCount(weights, label);
                SDVariable r;
                if(isMean){
                    //Example: MSE (mean) + mean by count over examples
                    r = sd.sum(preReduceLoss);
                } else {
                    //Example: L1 (sum) + mean by count over examples
                    SDVariable sum = sd.sum(preReduceLoss, dimensions);
                    r = sd.mean(sum);
                }
                b.loss(r.div(outputName, nonZeroWeights));
                break;
            default:
                throw new RuntimeException("Unknown reduction: " + reduction);
        }

        return b.build();
    }

}
