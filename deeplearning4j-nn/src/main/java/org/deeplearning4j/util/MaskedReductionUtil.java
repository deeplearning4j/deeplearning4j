package org.deeplearning4j.util;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * This is a TEMPORARY class that will be removed in a future release (once these approaches are formally implemented
 * in nd4j).
 *
 * @author Alex Black
 */
public class MaskedReductionUtil {

    private static final int[] CNN_DIM_MASK_H = new int[]{0,2};
    private static final int[] CNN_DIM_MASK_W = new int[]{0,3};


    public static INDArray maskedPoolingTimeSeries(PoolingType poolingType, INDArray toReduce, INDArray mask){
        if(toReduce.rank() != 3){
            throw new IllegalArgumentException("Expect rank 3 array: got " + toReduce.rank());
        }
        if(mask.rank() != 2){
            throw new IllegalArgumentException("Expect rank 2 array for mask: got " + toReduce.rank());
        }

        //Sum pooling: easy. Multiply by mask, then sum as normal
        //Average pooling: as above, but do a broadcast element-wise divi by mask.sum(1)
        //Max pooling: set to -inf if mask is 0, then do max as normal

        switch (poolingType){
            case MAX:
                INDArray withInf = Nd4j.createUninitialized(toReduce.shape());
                //Need something like a Broadcast CAS op

                throw new UnsupportedOperationException("Not yet implemented");
            case AVG:
            case SUM:
                INDArray masked = Nd4j.createUninitialized(toReduce.shape());
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked, 0,2));
                INDArray summed = masked.sum(2);
//                System.out.println(summed);
                if(poolingType == PoolingType.SUM){
                    return summed;
                }

                INDArray maskCounts = mask.sum(1);
                summed.diviColumnVector(maskCounts);
                return summed;
            case PNORM:
                throw new UnsupportedOperationException("Not yet implemented");
            case NONE:
                throw new UnsupportedOperationException("NONE pooling type not supported");
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }
    }


    public static INDArray maskedPoolingConvolution(PoolingType poolingType, INDArray toReduce, INDArray mask, boolean alongHeight){

        switch (poolingType){
            case MAX:
                throw new UnsupportedOperationException("Not yet implemented");
            case AVG:
            case SUM:
                // [minibatch, depth, h=1, w=X] or [minibatch, depth, h=X, w=1] data
                // with a mask array of shape [minibatch, X]

                //If masking along height: broadcast dimensions are [0,2]
                //If masking along width: broadcast dimensions are [0,3]

                INDArray masked = Nd4j.createUninitialized(toReduce.shape());
                int[] dimensions = (alongHeight ? CNN_DIM_MASK_H : CNN_DIM_MASK_W);
                Nd4j.getExecutioner().exec(new BroadcastMulOp(toReduce, mask, masked, dimensions));

                INDArray summed = masked.sum(2,3);
                if(poolingType == PoolingType.SUM){
                    return summed;
                }
                INDArray maskCounts = mask.sum(1);
                summed.diviColumnVector(maskCounts);
                return summed;

            case PNORM:
                throw new UnsupportedOperationException("Not yet implemented");
            case NONE:
                throw new UnsupportedOperationException("NONE pooling type not supported");
            default:
                throw new UnsupportedOperationException("Unknown or not supported pooling type: " + poolingType);
        }

    }
}
