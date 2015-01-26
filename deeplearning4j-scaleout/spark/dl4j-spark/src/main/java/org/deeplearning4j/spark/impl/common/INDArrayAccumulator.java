package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.api.ndarray.INDArray;
import scala.Option;

/**
 * Created by agibsonccc on 1/25/15.
 */
public class INDArrayAccumulator extends Accumulator<INDArray> {
    public INDArrayAccumulator(INDArray initialValue, AccumulatorParam<INDArray> param, Option<String> name) {
        super(initialValue, param, name);
    }

    public INDArrayAccumulator(INDArray initialValue, AccumulatorParam<INDArray> param) {
        super(initialValue, param);
    }
}
