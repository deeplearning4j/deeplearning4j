package org.nd4j.autodiff.validation.functions;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Function;

import java.util.Arrays;

@AllArgsConstructor
@Data
public class RelErrorFn implements Function<INDArray,String> {

    private final INDArray expected;
    private final double maxRelativeError;
    private final double minAbsoluteError;
    

    @Override
    public String apply(INDArray actual) {
        //TODO switch to binary relative error ops
        if(!Arrays.equals(expected.shape(), actual.shape())){
            throw new IllegalStateException("Shapes differ! " + Arrays.toString(expected.shape()) + " vs " + Arrays.toString(actual.shape()));
        }

        NdIndexIterator iter = new NdIndexIterator(expected.shape());
        while(iter.hasNext()){
            long[] next = iter.next();
            double d1 = expected.getDouble(next);
            double d2 = actual.getDouble(next);
            if(d1 == 0.0 && d2 == 0){
                continue;
            }
            if(Math.abs(d1-d2) < minAbsoluteError){
                continue;
            }
            double re = Math.abs(d1-d2) / (Math.abs(d1) + Math.abs(d2));
            if(re > maxRelativeError){
                return "Failed on relative error at position " + Arrays.toString(next) + ": relativeError=" + re + ", maxRE=" + maxRelativeError + ", absError=" +
                        Math.abs(d1-d2) + ", minAbsError=" + minAbsoluteError + " - values (" + d1 + "," + d2 + ")";
            }
        }
        return null;
    }
}
