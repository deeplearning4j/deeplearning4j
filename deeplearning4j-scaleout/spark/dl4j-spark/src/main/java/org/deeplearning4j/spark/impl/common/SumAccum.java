package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Option;

/**
 * @author Adam Gibson
 */
public class SumAccum extends Accumulator<INDArray> {

    public SumAccum(int length) {
        super(Nd4j.zeros(length), new ParamAccumulator());
    }

    @Override
    public Option<String> name() {
        return Option.empty();
    }
}
