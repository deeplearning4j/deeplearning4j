package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.InputPreProcessor;

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
}
