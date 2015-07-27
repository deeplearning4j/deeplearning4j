package org.deeplearning4j.nn.conf.preprocessor.output;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public abstract  class BaseOutputPreProcessor implements OutputPreProcessor {


    @Override
    public boolean equals(Object obj) {
        return obj.getClass().toString().equals(getClass().toString());
    }
}
