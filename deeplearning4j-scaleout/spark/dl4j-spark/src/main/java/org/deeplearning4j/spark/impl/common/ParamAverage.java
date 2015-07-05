package org.deeplearning4j.spark.impl.common;

import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class ParamAverage implements VoidFunction<INDArray> {
    private ParamAccumulator accum = new ParamAccumulator();


    @Override
    public void call(INDArray indArray) throws Exception {
         accum.add(indArray,indArray);
    }

    public ParamAccumulator getAccum() {
        return accum;
    }

    public void setAccum(ParamAccumulator accum) {
        this.accum = accum;
    }
}
