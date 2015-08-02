package org.deeplearning4j.nn.conf.preprocessor.processor;

import org.deeplearning4j.nn.conf.Processor;

/**
 * @author Adam Gibson
 */
public abstract class BaseProcessor implements Processor {


    @Override
    public boolean equals(Object obj) {
        return getClass().toString().equals(obj.getClass().toString());
    }
}
