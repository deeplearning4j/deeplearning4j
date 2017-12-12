package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
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

    @Builder(builderMethodName = "builder")
    public BatchNorm(SameDiff sameDiff, SDVariable[] inputFunctions, INDArray[] inputArrays, INDArray[] outputArrays,boolean inPlace, boolean training, boolean isLockGammaBeta, boolean isMiniBatch) {
        super(null,sameDiff, inputFunctions, inPlace);
        this.training = training;
        this.isLockGammaBeta = isLockGammaBeta;
        this.isMiniBatch = isMiniBatch;
        if(inputArrays != null) {
            addInputArgument(inputArrays);
        }

        if(outputArrays != null) {
            addOutputArgument(outputArrays);
        }

        addIArgument(fromBoolean(training));
        addIArgument(fromBoolean(isLockGammaBeta));
        addIArgument(fromBoolean(isMiniBatch));
    }



    public BatchNorm() {}


    @Override
    public String opName() {
        return "batchnorm";
    }

    @Override
    public String onnxName() {
        return "BatchNormalization";
    }

    @Override
    public String tensorflowName() {
        return "batch_norm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        BatchNormDerivative batchNormDerivative = BatchNormDerivative.derivativeBuilder()
                .isLockGammaBeta(isLockGammaBeta)
                .isMiniBatch(isMiniBatch)
                .training(training)
                .build();
        ret.addAll(Arrays.asList(batchNormDerivative.outputVariables()));
        return ret;
    }

}
