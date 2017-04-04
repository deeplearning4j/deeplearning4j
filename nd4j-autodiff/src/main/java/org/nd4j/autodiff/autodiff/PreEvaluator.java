package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.Field;

public interface PreEvaluator<X extends Field<X>> {
    public void update(Variable<X> v);

}
