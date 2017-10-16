package org.nd4j.linalg.api.ops.impl.layers;

import lombok.Builder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseModule;
import org.nd4j.linalg.api.ops.Module;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.ops.impl.transforms.Variable;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;

import java.util.ArrayList;
import java.util.List;

/**
 * Linear:
 * a * bT
 *
 * @author Adam Gibson
 */
public class Linear extends BaseModule {
    private DifferentialFunction forward;
    private int nIn,nOut;
    private WeightInitScheme weightInitScheme,biasWeightInitScheme;

    @Builder(builderMethodName = "execBuilder")
    public Linear(INDArray input,
                  int nIn,
                  int nOut,
                  WeightInitScheme weightInitScheme,
                  WeightInitScheme biasWeightInitScheme) {
        super(null,
                getParams(nIn,nOut,weightInitScheme,biasWeightInitScheme),
                new INDArray[]{Nd4j.create(Shape.getMatrixMultiplyShape(input.shape(),
                        new int[]{nOut,nIn}))},
                new ArrayList<Double>(), new ArrayList<Integer>(),new ArrayList<Module>());
        this.weightInitScheme = weightInitScheme;
        this.biasWeightInitScheme = biasWeightInitScheme;
        this.nIn = nIn;
        this.nOut = nOut;
    }

    @Builder(builderMethodName = "sameDiffBuilder")
    public Linear(SameDiff sameDiff,
                  int nIn,
                  int nOut,
                  DifferentialFunction input,
                  WeightInitScheme weightInitScheme,
                  WeightInitScheme biasWeightInitScheme) {
        super(null, sameDiff, new DifferentialFunction[]{input}, false, new ArrayList<Module>());
        this.weightInitScheme = weightInitScheme;
        this.biasWeightInitScheme = biasWeightInitScheme;

        this.args = getFunctionParams(nIn,nOut);
        this.nIn = nIn;
        this.nOut = nOut;

    }

    @Override
    public String opName() {
        return "linear";
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        execSameDiff();
        return forward.doDiff(f1);
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret = new ArrayList<>();
        ret.add(Shape.getMatrixMultiplyShape(getInputArguments().get(0).shape(),new int[]{nOut,nIn}));

        ret.add(Shape.getMatrixMultiplyShape(getInputArguments().get(0).shape(),getInputArguments().get(1).transpose().shape()));
        if(biasWeightInitScheme != null) {
            ret.add(new int[]{nOut,1});
        }
        return ret;
    }

    @Override
    public void exec() {
        if(this.getInputArguments().isEmpty()) {
            throw new IllegalStateException("No arguments found.");
        }

        INDArray input = getInputArguments().get(0);
        INDArray right = getInputArguments().get(1);
        if(getOutputArguments().isEmpty()) {
            if(getInputArguments().size() == 1)
                getOutputArguments().add(input.mmul(right.transpose()));
            else
                getOutputArguments().add(input.mmul(right.transpose()).addiRowVector(getInputArguments().get(2)));

        }
        else {
            input.mmul(right.transpose(),getOutputArguments().get(0));
        }

    }

    @Override
    public void execSameDiff() {
        if(args == null || args.length == 0) {
            throw new IllegalStateException("No arguments found");
        }

        if(forward == null) {
            //bias needs to be added yet
            if(args.length > 1)
                forward =  f().add(new Mmul(sameDiff, args()[0], args()[1],
                        MMulTranspose.builder().transposeA(false).transposeB(true).build()),args()[1]);
            else {
                forward = new Mmul(sameDiff, args()[0], args()[1],
                        MMulTranspose.builder().transposeA(false).transposeB(true).build());
            }

            this.outputFunctions = forward.outputFunctions();
        }


    }

    private static INDArray[] getParams(int nIn,
                                        int nOut,
                                        WeightInitScheme paramsScheme,
                                        WeightInitScheme biasInitScheme) {
        if(biasInitScheme != null) {
            return new INDArray[] {paramsScheme.create(new int[]{nOut,nIn}),biasInitScheme.create(new int[]{nOut,1})};
        }
        else {
            return new INDArray[] {paramsScheme.create(new int[]{nOut,nIn})};

        }
    }

    private DifferentialFunction[] getFunctionParams(int nIn,
                                                     int nOut) {
        if(biasWeightInitScheme != null) {
            return new DifferentialFunction[] {
                    new Variable(sameDiff,"w", NDArrayInformation.newInfo(new int[]{nOut,nIn}
                            ,weightInitScheme)),
                    new Variable(sameDiff,"b", NDArrayInformation.newInfo(new int[]{nOut,1}
                            ,biasWeightInitScheme))
            };
        }
        else {
            return new DifferentialFunction[] {
                    new Variable(sameDiff,"w", NDArrayInformation.newInfo(new int[]{nOut,nIn}))
            };
        }
    }
}
