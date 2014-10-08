package org.nd4j.linalg.indexing.functions;

import com.google.common.base.Function;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class Identity implements Function<Number,Number> {
    @Override
    public Number apply(Number input) {
        return input;
    }
}
