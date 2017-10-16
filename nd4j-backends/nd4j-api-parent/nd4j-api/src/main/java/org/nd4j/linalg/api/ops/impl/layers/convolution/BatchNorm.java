package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * BatchNorm operation
 */
@Slf4j
@Getter
public class BatchNorm extends DynamicCustomOp {

    private boolean training;
    private boolean isLockGammaBeta;
    private boolean isMiniBatch;

    @Builder(builderMethodName = "sameDiffBuilder")
    public BatchNorm(SameDiff sameDiff, DifferentialFunction[] inputs, boolean inPlace, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(null,sameDiff, inputs, inPlace);
        this.training = training;
        this.isLockGammaBeta = isLockGammaBeta;
        this.isMiniBatch = isMiniBatch;
        getIArguments().add(fromBoolean(training));
        getIArguments().add(fromBoolean(isLockGammaBeta));
        getIArguments().add(fromBoolean(isMiniBatch));
    }

    @Builder(builderMethodName = "execBuilder")
    public BatchNorm(INDArray[] inputs, INDArray[] outputs, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(null,inputs,outputs);
        this.training = training;
        this.isLockGammaBeta = isLockGammaBeta;
        this.isMiniBatch = isMiniBatch;
        getIArguments().add(fromBoolean(training));
        getIArguments().add(fromBoolean(isLockGammaBeta));
        getIArguments().add(fromBoolean(isMiniBatch));
    }

    public BatchNorm() {}


    @Override
    public String opName() {
        return "batchnorm";
    }





    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

}
