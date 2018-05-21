package org.nd4j.linalg.indexing.functions;


import com.google.common.base.Function;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Returns a stable number based on infinity
 * or nan
 */
public class StableNumber implements Function<Number, Number> {
    private Type type;

    public enum Type {
        DOUBLE, FLOAT
    }

    public StableNumber(Type type) {
        this.type = type;
    }

    @Override
    public Number apply(Number number) {
        switch (type) {
            case DOUBLE:
                if (Double.isInfinite(number.doubleValue()))
                    return -Double.MAX_VALUE;
                if (Double.isNaN(number.doubleValue()))
                    return Nd4j.EPS_THRESHOLD;
            case FLOAT:
                if (Float.isInfinite(number.floatValue()))
                    return -Float.MAX_VALUE;
                if (Float.isNaN(number.floatValue()))
                    return Nd4j.EPS_THRESHOLD;
            default:
                throw new IllegalStateException("Illegal opType");

        }

    }
}
