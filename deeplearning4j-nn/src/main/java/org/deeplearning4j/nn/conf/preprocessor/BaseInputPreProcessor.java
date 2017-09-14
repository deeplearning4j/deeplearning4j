package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * @author Adam Gibson
 */

public abstract class BaseInputPreProcessor implements InputPreProcessor {
    @Override
    public BaseInputPreProcessor clone() {
        try {
            BaseInputPreProcessor clone = (BaseInputPreProcessor) super.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Default: pass-through, unmodified
        return new Pair<>(maskArray, currentMaskState);
    }
}
