package org.nd4j.linalg.indexing.functions;

import com.google.common.base.Function;
import com.google.common.base.Functions;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class Zero implements Function<Number,Number> {
    @Override
    public Number apply(Number input) {
        return 0;
    }
}
