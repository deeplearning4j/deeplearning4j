package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class Pooling2DDerivative extends Pooling2D {
    @Builder(builderMethodName = "derivativeBuilder")
    public Pooling2DDerivative(SameDiff sameDiff, SDVariable[] inputs, INDArray[] arrayInputs, INDArray[] arrayOutputs, Pooling2DConfig config) {
        super(sameDiff, inputs, arrayInputs, arrayOutputs, config);
    }

    public Pooling2DDerivative() {}


    @Override
    public String opName() {
         return super.opName() + "_bp";
    }

   @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
       throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
