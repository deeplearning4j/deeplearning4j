package org.deeplearning4j.nn.conf.preprocessor.input;

import org.deeplearning4j.nn.conf.InputPreProcessor;

/**
 * @author Adam Gibson
 */
public abstract class BaseInputPreProcessor implements InputPreProcessor {
    @Override
    public boolean equals(Object obj) {
        return getClass().toString().equals(obj.getClass().toString());
    }
}
