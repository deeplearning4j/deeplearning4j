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

package org.deeplearning4j.util;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class MaskedReductionUtil {

    private static final int[] CNN_DIM_MASK_H = new int[] {0, 2};
    private static final int[] CNN_DIM_MASK_W = new int[] {0, 3};

    private MaskedReductionUtil(){ }

    public static INDArray maskedPoolingTimeSeries(PoolingType poolingType, INDArray toReduce, INDArray mask,
                                                   int pnorm, DataType dataType) {
        if (toReduce.rank() != 3) {
            throw new IllegalArgumentException("Expect rank 3 array: got " + toReduce.rank());
        }
        if (mask.rank() != 2) {
            throw new IllegalArgumentException("Expect rank 2 array for mask: got " + mask.rank());
        }

        toReduce = toReduce.castTo(dataType);
        mask = mask.castTo(dataType);

        //Sum pooling: easy. Multiply by mask, then sum as normal
        //Average pooling: as above, but do a broadcast element-wise divi by mask.sum(1)
        //Max pooling: set to -inf if mask is 0, then do max as normal

        switch (poolingType) {
            case MAX:
                INDArray negInfMask = mask.castTo(dataType).rsub(1.0);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(dataType, toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(toReduce, negInfMask, withInf, 0, 2));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                return withInf.max(2);
            case AVG:
            case SUM:
                INDArray masked = Nd4j.createUninitialized(dataType, toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked, 0, 2));
                INDArray summed = masked.sum(2);
                if (poolingType == PoolingType.SUM) {
                    return summed;
                }

                INDArray maskCounts = mask.sum(1);
                // Guard against zero-count rows to avoid NaN from division by zero
                // (can occur if mask has all-zero rows, e.g. integer-type masks after truncation)
                BooleanIndexing.replaceWhere(maskCounts, 1.0, Conditions.equals(0.0));
                summed.diviColumnVector(maskCounts);
                return summed;
            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = Nd4j.createUninitialized(dataType, toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked2, 0, 2));

                INDArray abs = Transforms.abs(masked2, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = abs.sum(2);

                return Transforms.pow(pNorm, 1.0 / pnorm);
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }

    public static INDArray maskedPoolingEpsilonTimeSeries(PoolingType poolingType, INDArray input, INDArray mask,
                                                          INDArray epsilon2d, int pnorm) {

        if (input.rank() != 3) {
            throw new IllegalArgumentException("Expect rank 3 input activation array: got " + input.rank());
        }
        if (mask.rank() != 2) {
            throw new IllegalArgumentException("Expect rank 2 array for mask: got " + mask.rank());
        }


        //Mask: [minibatch, tsLength]
        //Epsilon: [minibatch, vectorSize]

        mask = mask.castTo(input.dataType());

        switch (poolingType) {
            case MAX:
                INDArray negInfMask = mask.rsub(1.0);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(input.dataType(), input.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(input, negInfMask, withInf, 0, 2));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                INDArray isMaxBool = Nd4j.createUninitialized(DataType.BOOL, withInf.shape(), withInf.ordering());
                Nd4j.exec(new IsMax(withInf, isMaxBool, 2));
                INDArray isMax = isMaxBool.castTo(withInf.dataType());

                return Nd4j.getExecutioner().exec(new BroadcastMulOp(isMax, epsilon2d, isMax, 0, 1));
            case AVG:
            case SUM:
                //if out = sum(in,dims) then dL/dIn = dL/dOut -> duplicate to each step and mask
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                //With masking: N differs for different time series

                INDArray out = Nd4j.createUninitialized(input.dataType(), input.shape(), 'f');

                //Broadcast copy op, then divide and mask to 0 as appropriate
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(out, epsilon2d, out, 0, 1));
                Nd4j.getExecutioner().exec(new BroadcastMulOp(out, mask, out, 0, 2));

                if (poolingType == PoolingType.SUM) {
                    return out;
                }

                INDArray nEachTimeSeries = mask.sum(1); //[minibatchSize,tsLength] -> [minibatchSize,1]
                Nd4j.getExecutioner().exec(new BroadcastDivOp(out, nEachTimeSeries, out, 0));

                return out;

            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = Nd4j.createUninitialized(input.dataType(), input.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(input, mask, masked2, 0, 2));

                INDArray abs = Transforms.abs(masked2, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = Transforms.pow(abs.sum(2), 1.0 / pnorm);

                INDArray numerator;
                if (pnorm == 2) {
                    numerator = input.dup();
                } else {
                    INDArray absp2 = Transforms.pow(Transforms.abs(input, true), pnorm - 2, false);
                    numerator = input.mul(absp2);
                }

                INDArray denom = Transforms.pow(pNorm, pnorm - 1, false);
                //3d shape with trailing dimension of 1
                if(epsilon2d.rank() != denom.rank() && denom.length() == epsilon2d.length()) {
                    epsilon2d = epsilon2d.reshape(denom.shape());
                }
                denom.rdivi(epsilon2d);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(numerator, denom, numerator, 0, 1));
                Nd4j.getExecutioner().exec(new BroadcastMulOp(numerator, mask, numerator, 0, 2)); //Apply mask

                return numerator;
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }


    public static INDArray maskedPoolingConvolution(PoolingType poolingType, INDArray toReduce, INDArray mask, int pnorm, DataType dataType) {
        if(mask.rank() != 4){
            //TODO BETTER ERROR MESSAGE EXPLAINING FORMAT
            //TODO ALSO HANDLE LEGACY FORMAT WITH WARNING WHERE POSSIBLE
            throw new IllegalStateException("Expected rank 4 mask array: Got array with shape " + Arrays.toString(mask.shape()));
        }

        toReduce = toReduce.castTo(dataType);   //cast input to target dtype (no-op if already correct)
        mask = mask.castTo(dataType);   //no-op if already correct dtype

        // [minibatch, channels, h, w] data with a mask array of shape [minibatch, 1, X, Y]
        // where X=(1 or inH) and Y=(1 or inW)
        // Use element-wise mul with broadcasting (NumPy-style) to correctly handle masks with
        // singleton dimensions (e.g. [mb,1,h,w] broadcast over [mb,c,h,w]).

        switch (poolingType) {
            case MAX:
                //TODO This is ugly - replace it with something better... Need something like a Broadcast CAS op
                INDArray negInfMask;
                if(mask.dataType() == DataType.BOOL){
                    negInfMask = Transforms.not(mask).castTo(dataType);
                } else {
                    negInfMask = mask.rsub(1.0);
                }
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                // Use broadcasting add: negInfMask has shape [mb,1,h,w] or similar, toReduce is [mb,c,h,w]
                INDArray withInf = toReduce.add(negInfMask);
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                return withInf.max(2, 3);
            case AVG:
            case SUM:
                // Element-wise mul with broadcasting handles singleton dims correctly
                INDArray masked = toReduce.mul(mask);

                INDArray summed = masked.sum(2, 3);
                if (poolingType == PoolingType.SUM) {
                    return summed;
                }
                // Compute N = number of valid spatial positions per batch item.
                // mask may have singleton dims (e.g. [mb,1,H,1] or [mb,1,1,W]).
                // mask.sum(1,2,3) would only count the non-singleton positions.
                // Instead, broadcast mask to full spatial shape first.
                long mb0 = toReduce.size(0);
                long inH = toReduce.size(2), inW = toReduce.size(3);
                // Broadcast mask to full spatial extent only (not channels) to get correct N.
                // mask is [mb,1,H,1] or [mb,1,1,W] or [mb,1,H,W] — broadcast to [mb,1,inH,inW]
                // then sum over dims 1,2,3 to get valid spatial position count per minibatch.
                INDArray maskExpanded = mask.broadcast(mb0, 1, inH, inW);
                INDArray maskCounts = maskExpanded.sum(1, 2, 3);
                // Guard against zero-count rows to avoid NaN from division by zero
                BooleanIndexing.replaceWhere(maskCounts, 1.0, Conditions.equals(0.0));
                summed.diviColumnVector(maskCounts);
                return summed;

            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = toReduce.mul(mask);

                INDArray abs = Transforms.abs(masked2, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = abs.sum(2, 3);

                return Transforms.pow(pNorm, 1.0 / pnorm);
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }


    public static INDArray maskedPoolingEpsilonCnn(PoolingType poolingType, INDArray input, INDArray mask,
                                                   INDArray epsilon2d, int pnorm, DataType dataType) {

        // input: [minibatch, channels, H, W]
        // mask:  [minibatch, 1, H, 1] or [minibatch, 1, 1, W] (4D, broadcastable over channels)
        // epsilon2d: [minibatch, channels] (2D gradient from downstream)

        mask = mask.castTo(dataType);   //No-op if correct type

        // epsilon2d is [mb, ch]; expand to [mb, ch, 1, 1] so it broadcasts over H and W.
        long mb = input.size(0);
        long ch = input.size(1);
        INDArray epsilonExpanded = epsilon2d.reshape(mb, ch, 1, 1);

        switch (poolingType) {
            case MAX:
                // Build withInf: set masked positions to -inf so they can't be the max
                INDArray negInfMask;
                if(mask.dataType() == DataType.BOOL){
                    negInfMask = Transforms.not(mask).castTo(dataType);
                } else {
                    negInfMask = mask.rsub(1.0);
                }
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                // NumPy-style broadcast: input[mb,ch,H,W] + negInfMask[mb,1,H,W] -> [mb,ch,H,W]
                INDArray withInf = input.add(negInfMask);
                //At this point: all the masked out positions have value -inf, hence can't be the max

                INDArray isMaxBool2 = Nd4j.createUninitialized(DataType.BOOL, withInf.shape(), withInf.ordering());
                Nd4j.exec(new IsMax(withInf, isMaxBool2, 2, 3));
                INDArray isMax2 = isMaxBool2.castTo(withInf.dataType());

                // Multiply isMax by epsilon: broadcast epsilon[mb,ch,1,1] over H,W
                return isMax2.muli(epsilonExpanded);

            case AVG:
            case SUM:
                //if out = sum(in,dims) then dL/dIn = dL/dOut -> duplicate to each spatial position and mask
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                //With masking: N differs for different examples

                // Broadcast epsilon[mb,ch,1,1] to all H,W positions, then zero out masked positions
                INDArray out = epsilonExpanded.broadcast(input.shape()).dup();
                // Apply mask via NumPy-style broadcast: out[mb,ch,H,W] * mask[mb,1,H,W]
                out.muli(mask);

                if (poolingType == PoolingType.SUM) {
                    return out;
                }

                // N per batch item: count of valid spatial positions, accounting for
                // broadcast expansion of singleton dims in mask (e.g. [mb,1,H,1] or [mb,1,1,W]).
                // mask.sum(1,2,3) only counts the non-broadcast positions; we must broadcast first.
                long inH2 = input.size(2), inW2 = input.size(3);
                // Broadcast mask to full spatial extent only (not channels) to get correct N.
                INDArray maskExpanded2 = mask.broadcast(mb, 1, inH2, inW2);
                INDArray nEachBatch = maskExpanded2.sum(1, 2, 3).reshape(mb, 1, 1, 1);
                BooleanIndexing.replaceWhere(nEachBatch, 1.0, Conditions.equals(0.0));
                out.divi(nEachBatch);

                return out;

            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                // Apply mask via NumPy-style broadcast
                INDArray masked2 = input.mul(mask);

                INDArray abs = Transforms.abs(masked2, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = Transforms.pow(abs.sum(2, 3), 1.0 / pnorm);

                INDArray numerator;
                if (pnorm == 2) {
                    numerator = input.dup();
                } else {
                    INDArray absp2 = Transforms.pow(Transforms.abs(input, true), pnorm - 2, false);
                    numerator = input.mul(absp2);
                }

                // denom: pNorm^(p-1), shape [mb,ch]; expand to [mb,ch,1,1]
                INDArray denom = Transforms.pow(pNorm, pnorm - 1, false).reshape(mb, ch, 1, 1);
                // epsilonExpanded / denom -> [mb,ch,1,1]
                INDArray scaledEpsilon = epsilonExpanded.div(denom);
                // Multiply numerator[mb,ch,H,W] by scaledEpsilon[mb,ch,1,1]
                numerator.muli(scaledEpsilon);
                // Apply mask: zero out invalid positions
                numerator.muli(mask);

                return numerator;
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);

        }
    }
}
