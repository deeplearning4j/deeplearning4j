package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNumber;

/**
 * Condition for boolean indexing
 */
public interface Condition  extends Function<Number,Boolean> {

    @Override
    public Boolean apply(Number input);

     public Boolean apply(IComplexNumber input);
}
