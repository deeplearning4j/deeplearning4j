package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * UpsamplingDerivative operation
 */
@Slf4j
public class Upsampling2dDerivative extends DynamicCustomOp {

    protected boolean nchw;
    protected int scaleH;
    protected int scaleW;

    public Upsampling2dDerivative() {}

    public Upsampling2dDerivative(SameDiff sameDiff, SDVariable input, SDVariable gradient, boolean nchw, int scaleH, int scaleW) {
        super(null, sameDiff, new SDVariable[]{input, gradient});

        this.nchw = nchw;
        this.scaleH = scaleH;
        this.scaleW = scaleW;

        addIArgument(scaleH);
        addIArgument(scaleW);
        addIArgument(nchw ? 1 : 0);
    }

    @Override
    public String opName() {
        return "upsampling2d_bp";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Unable to take derivative of derivative.");
    }

}
