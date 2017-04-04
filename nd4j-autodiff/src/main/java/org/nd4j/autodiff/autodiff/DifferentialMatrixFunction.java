package org.nd4j.autodiff.autodiff;

import org.nd4j.autodiff.Field;
import org.nd4j.autodiff.Ring;

public interface DifferentialMatrixFunction<X extends Field<X>> extends
        Ring<DifferentialMatrixFunction<X>>, Differential<X, DifferentialMatrixFunction<X>> {

}
