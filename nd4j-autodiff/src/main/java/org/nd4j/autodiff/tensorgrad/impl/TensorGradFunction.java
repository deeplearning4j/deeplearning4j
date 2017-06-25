package org.nd4j.autodiff.tensorgrad.impl;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.io.Serializable;

/**
 * Created by agibsonccc on 4/17/17.
 */
@Data
@Builder
@AllArgsConstructor
public class TensorGradFunction implements Serializable {
    protected DifferentialFunction<ArrayField> differentialFunction;


    public TensorGradVariable exec(TensorGradVariable[] inputs, Object[] extraArgs) {
          return null;
    }



}
