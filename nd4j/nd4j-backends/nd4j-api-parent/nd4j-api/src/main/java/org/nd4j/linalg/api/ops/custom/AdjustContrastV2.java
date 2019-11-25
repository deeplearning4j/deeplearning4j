package org.nd4j.linalg.api.ops.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

public class AdjustContrastV2 extends BaseAdjustContrast {

    public AdjustContrastV2() {super();}

    public AdjustContrastV2(INDArray in, double factor, INDArray out) {
        super(in, factor, out);
    }

    public AdjustContrastV2(SameDiff sameDiff, SDVariable in, SDVariable factor) {
        super( sameDiff,new SDVariable[]{in,factor});
    }

    @Override
    public String opName() {
        return "adjust_contrast_v2";
    }

    @Override
    public String tensorflowName() {
        return "AdjustContrastv2";
    }
}