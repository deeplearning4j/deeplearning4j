package org.nd4j.autodiff.tensorgrad.impl;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.autodiff.DifferentialFunction;
import org.nd4j.autodiff.autodiff.Variable;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 4/9/17.
 */
@Data
@Builder
public class TensorGradVariable {
    private INDArray arr;
    private Variable<ArrayField> arrayField;
    private DifferentialFunction<ArrayField> differentialFunction;
    private String varName;
    private TensorGrad tensorGrad;


}
