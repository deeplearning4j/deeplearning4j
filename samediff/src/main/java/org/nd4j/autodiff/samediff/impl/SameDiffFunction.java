package org.nd4j.autodiff.samediff.impl;

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
public class SameDiffFunction implements Serializable {
    protected DifferentialFunction<ArrayField> differentialFunction;


    public SameDiffVariable exec(SameDiffVariable[] inputs, Object[] extraArgs) {
          return null;
    }



}
