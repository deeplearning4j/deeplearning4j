package org.deeplearning4j.nn.conf.preprocessor.output;

import org.deeplearning4j.nn.conf.OutputPostProcessor;

/**
 * @author Adam Gibson
 */
@Deprecated
public abstract  class BaseOutputPostProcessor implements OutputPostProcessor {


    @Override
    public boolean equals(Object obj) {
        return obj.getClass().toString().equals(getClass().toString());
    }
}
