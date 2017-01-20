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
