package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Upsampling operation
 */
@Slf4j
@Getter
public class Upsampling2d extends DynamicCustomOp {


    protected boolean nchw;
    protected int scaleH;
    protected int scaleW;

    public Upsampling2d(SameDiff sameDiff, SDVariable input, boolean nchw, int scaleH, int scaleW) {
        super(null,sameDiff, new SDVariable[]{input});
        this.nchw = nchw;
        this.scaleH = scaleH;
        this.scaleW = scaleW;

        addIArgument(scaleH);
        addIArgument(scaleW);
        addIArgument(nchw ? 1 : 0);
    }


    public Upsampling2d() {}


    @Override
    public String opName() {
        return "upsampling2d";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().upsampling2dBp(arg(), f1.get(0), nchw, scaleH, scaleW));
    }

}
