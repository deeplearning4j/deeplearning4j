package org.nd4j.linalg.indexing.functions;

import java.util.function.Function;

/**
 * Created by agibsonccc on 10/8/14.
 */
public class Value implements Function<Number,Number> {
    private Number number;

    public Value(Number number) {
        this.number = number;
    }

    @Override
    public Number apply(Number number) {
        return this.number;
    }
}
