package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;


/**
 * Pooling2DDerivative operation
 */
@Slf4j
public class Pooling2DDerivative extends Pooling2D {


    @Builder(builderMethodName = "sameDiffDerivativeBuilder")
    public Pooling2DDerivative(SameDiff sameDiff, DifferentialFunction[] inputs, boolean inPlace, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, int virtualHeight, int virtualWidth, double extra, Pooling2DType type, boolean isSameMode) {
        super(sameDiff, inputs, inPlace, kh, kw, sy, sx, ph, pw, dh, dw, virtualHeight, virtualWidth, extra, type, isSameMode);
    }

    @Builder(builderMethodName = "execDerivativeBuilder")
    public Pooling2DDerivative(INDArray[] inputs, INDArray[] outputs, int kh, int kw, int sy, int sx, int ph, int pw, int dh, int dw, int virtualHeight, int virtualWidth, double extra, Pooling2DType type, boolean isSameMode) {
        super(inputs, outputs, kh, kw, sy, sx, ph, pw, dh, dw, virtualHeight, virtualWidth, extra, type, isSameMode);
    }

    public Pooling2DDerivative() {}


    @Override
    public String opName() {
         return super.opName() + "_bp";
    }

   @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
       throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
