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

package org.deeplearning4j.util;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 *
 * This is a TEMPORARY class for implementing global pooling with masking. Note that it may be removed in a future release,
 * if and when these approaches are formally implemented as native operations in ND4J. Consequently, this should not
 * be considered part of the public API.
 *
 * @author Alex Black
 */
public class MaskedReductionUtil {

    private static final int[] CNN_DIM_MASK_H = new int[] {0, 2};
    private static final int[] CNN_DIM_MASK_W = new int[] {0, 3};

    private MaskedReductionUtil(){ }

    public static INDArray maskedPoolingTimeSeries(PoolingType poolingType, INDArray toReduce, INDArray mask,
                    int pnorm) {
        if (toReduce.rank() != 3) {
            throw new IllegalArgumentException("Expect rank 3 array: got " + toReduce.rank());
        }
        if (mask.rank() != 2) {
            throw new IllegalArgumentException("Expect rank 2 array for mask: got " + mask.rank());
        }

        //Sum pooling: easy. Multiply by mask, then sum as normal
        //Average pooling: as above, but do a broadcast element-wise divi by mask.sum(1)
        //Max pooling: set to -inf if mask is 0, then do max as normal

        switch (poolingType) {
            case MAX:
                //TODO This is ugly - replace it with something better... Need something like a Broadcast CAS op
                INDArray negInfMask = Transforms.not(mask);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(toReduce, negInfMask, withInf, 0, 2));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                return withInf.max(2);
            case AVG:
            case SUM:
                INDArray masked = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked, 0, 2));
                INDArray summed = masked.sum(2);
                if (poolingType == PoolingType.SUM) {
                    return summed;
                }

                INDArray maskCounts = mask.sum(1);
                summed.diviColumnVector(maskCounts);
                return summed;
            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = Nd4j.createUninitialized(toReduce.shape());
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
        if (epsilon2d.rank() != 2) {
            throw new IllegalArgumentException("Expected rank 2 array for errors: got " + epsilon2d.rank());
        }

        //Mask: [minibatch, tsLength]
        //Epsilon: [minibatch, vectorSize]

        switch (poolingType) {
            case MAX:
                //TODO This is ugly - replace it with something better... Need something like a Broadcast CAS op
                INDArray negInfMask = Transforms.not(mask);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(input.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(input, negInfMask, withInf, 0, 2));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(withInf, 2));

                return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(isMax, epsilon2d, isMax, 0, 1));
            case AVG:
            case SUM:
                //if out = sum(in,dims) then dL/dIn = dL/dOut -> duplicate to each step and mask
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                //With masking: N differs for different time series

                INDArray out = Nd4j.createUninitialized(input.shape(), 'f');

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
                INDArray masked2 = Nd4j.createUninitialized(input.shape());
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
                denom.rdivi(epsilon2d);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(numerator, denom, numerator, 0, 1));
                Nd4j.getExecutioner().exec(new BroadcastMulOp(numerator, mask, numerator, 0, 2)); //Apply mask

                return numerator;
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }


    public static INDArray maskedPoolingConvolution(PoolingType poolingType, INDArray toReduce, INDArray mask, int pnorm) {
        if(mask.rank() != 4){
            //TODO BETTER ERROR MESSAGE EXPLAINING FORMAT
            //TODO ALSO HANDLE LEGACY FORMAT WITH WARNING WHERE POSSIBLE
            throw new IllegalStateException("Expected rank 4 mask array: Got array with shape " + Arrays.toString(mask.shape()));
        }

        // [minibatch, channels, h, w] data with a mask array of shape [minibatch, 1, X, Y]
        // where X=(1 or inH) and Y=(1 or inW)

        //General case: must be equal or 1 on each dimension
        int[] dimensions = new int[4];
        int count = 0;
        for(int i=0; i<4; i++ ){
            if(toReduce.size(i) == mask.size(i)){
                dimensions[count++] = i;
            }
        }
        if(count < 4){
            dimensions = Arrays.copyOfRange(dimensions, 0, count);
        }

        switch (poolingType) {
            case MAX:
                //TODO This is ugly - replace it with something better... Need something like a Broadcast CAS op
                INDArray negInfMask = Transforms.not(mask);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(toReduce, negInfMask, withInf, dimensions));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                return withInf.max(2, 3);
            case AVG:
            case SUM:
                INDArray masked = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked, dimensions));

                INDArray summed = masked.sum(2, 3);
                if (poolingType == PoolingType.SUM) {
                    return summed;
                }
                INDArray maskCounts = mask.sum(1,2,3);
                summed.diviColumnVector(maskCounts);
                return summed;

            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked2, dimensions));

                INDArray abs = Transforms.abs(masked2, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = abs.sum(2, 3);

                return Transforms.pow(pNorm, 1.0 / pnorm);
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }


    public static INDArray maskedPoolingEpsilonCnn(PoolingType poolingType, INDArray input, INDArray mask,
                    INDArray epsilon2d, boolean alongHeight, int pnorm) {

        // [minibatch, channels, h=1, w=X] or [minibatch, channels, h=X, w=1] data
        // with a mask array of shape [minibatch, X]

        //If masking along height: broadcast dimensions are [0,2]
        //If masking along width: broadcast dimensions are [0,3]

        int[] dimensions = (alongHeight ? CNN_DIM_MASK_H : CNN_DIM_MASK_W);

        switch (poolingType) {
            case MAX:
                //TODO This is ugly - replace it with something better... Need something like a Broadcast CAS op
                INDArray negInfMask = Transforms.not(mask);
                BooleanIndexing.replaceWhere(negInfMask, Double.NEGATIVE_INFINITY, Conditions.equals(1.0));

                INDArray withInf = Nd4j.createUninitialized(input.shape());
                Nd4j.getExecutioner().exec(new BroadcastAddOp(input, negInfMask, withInf, dimensions));
                //At this point: all the masked out steps have value -inf, hence can't be the output of the MAX op

                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(withInf, 2, 3));

                return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(isMax, epsilon2d, isMax, 0, 1));
            case AVG:
            case SUM:
                //if out = sum(in,dims) then dL/dIn = dL/dOut -> duplicate to each step and mask
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                //With masking: N differs for different time series

                INDArray out = Nd4j.createUninitialized(input.shape(), 'f');

                //Broadcast copy op, then divide and mask to 0 as appropriate
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(out, epsilon2d, out, 0, 1));
                Nd4j.getExecutioner().exec(new BroadcastMulOp(out, mask, out, dimensions));

                if (poolingType == PoolingType.SUM) {
                    return out;
                }

                //Note that with CNNs, current design is restricted to [minibatch, channels, 1, W] ot [minibatch, channels, H, 1]
                INDArray nEachTimeSeries = mask.sum(1); //[minibatchSize,tsLength] -> [minibatchSize,1]
                Nd4j.getExecutioner().exec(new BroadcastDivOp(out, nEachTimeSeries, out, 0));

                return out;

            case PNORM:
                //Similar to average and sum pooling: there's no N term here, so we can just set the masked values to 0
                INDArray masked2 = Nd4j.createUninitialized(input.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(input, mask, masked2, dimensions));

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

                INDArray denom = Transforms.pow(pNorm, pnorm - 1, false);
                denom.rdivi(epsilon2d);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(numerator, denom, numerator, 0, 1));
                Nd4j.getExecutioner().exec(new BroadcastMulOp(numerator, mask, numerator, dimensions)); //Apply mask

                return numerator;
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);

        }
    }
}
