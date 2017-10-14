package org.nd4j.linalg.api.ops.impl.transforms.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Upsampling operation
 */
@Slf4j
@Getter
public class Upsampling extends DynamicCustomOp {


    private int scaleFactor;

    @Builder(builderMethodName = "sameDiffBuilder")
    public Upsampling(SameDiff sameDiff, DifferentialFunction[] inputs,boolean inPlace, int scaleFactor) {
        super(null,sameDiff, inputs, inPlace);
        this.scaleFactor = scaleFactor;
        getIArguments().add(scaleFactor);
    }

    @Builder(builderMethodName = "execBuilder")
    public Upsampling(INDArray[] inputs, INDArray[] outputs, int scaleFactor) {
        super(null,inputs,outputs);
        this.scaleFactor = scaleFactor;
        getIArguments().add(scaleFactor);
    }

    public Upsampling() {}


    @Override
    public String opName() {
        return "upsampling";
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        UpsamplingDerivative upsamplingDerivative = UpsamplingDerivative.sameDiffDerivativeBuilder()
                .scaleFactor(scaleFactor)
                .inputs(f1.toArray(new DifferentialFunction[f1.size()]))
                .build();
        List<DifferentialFunction> ret = new ArrayList<>();
        ret.addAll(Arrays.asList(upsamplingDerivative.getOutputFunctions()));
        return ret;
    }

}
