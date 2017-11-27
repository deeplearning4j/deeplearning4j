package org.nd4j.linalg.api.ops.impl.layers;

import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseModule;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Linear:
 * a * bT
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class Linear extends BaseModule {
    private DifferentialFunction forward;
    private int nIn,nOut;
    private WeightInitScheme weightInitScheme,biasWeightInitScheme;

    @Builder(builderMethodName = "execBuilder")
    public Linear(int nIn,
                  int nOut,
                  WeightInitScheme weightInitScheme,
                  WeightInitScheme biasWeightInitScheme) {
        super(null,
                getParams(nIn,nOut,weightInitScheme,biasWeightInitScheme),
                new INDArray[]{},
                new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        this.weightInitScheme = weightInitScheme;
        this.biasWeightInitScheme = biasWeightInitScheme;
        this.nIn = nIn;
        this.nOut = nOut;
    }

    @Builder(builderMethodName = "sameDiffBuilder")
    public Linear(SameDiff sameDiff,
                  int nIn,
                  int nOut,
                  WeightInitScheme weightInitScheme,
                  WeightInitScheme biasWeightInitScheme) {
        super(null, sameDiff, null, false, new ArrayList<>());
        this.weightInitScheme = weightInitScheme;
        this.biasWeightInitScheme = biasWeightInitScheme;

        sameDiff.associateFunctionsAsArgs(getFunctionParams(nIn,nOut),this);
        this.nIn = nIn;
        this.nOut = nOut;

    }

    @Override
    public String opName() {
        return "linear";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

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
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }



    @Override
    public void exec(INDArray... inputs) {
        if(this.getInputArguments().isEmpty()) {
            throw new IllegalStateException("No arguments found.");
        }

        INDArray weights = getInputArguments().get(0);
        INDArray right = getInputArguments().get(1);
        if(getOutputArguments().isEmpty()) {
            if(getInputArguments().size() == 1)
                getOutputArguments().add(inputs[0].mmul(weights.transpose()));
            else
                getOutputArguments().add(inputs[0].mmul(weights.transpose()).addiColumnVector(right));

        }
        else {
            inputs[0].mmul(weights.transpose(),getOutputArguments().get(0));
        }

    }

    @Override
    public void execSameDiff(DifferentialFunction... input) {
        val args = args();
        if(args == null || args.length == 0) {
            throw new IllegalStateException("No arguments found");
        }

        if(forward == null) {
            //bias needs to be added yet
            if(args.length > 1)
                forward =  f().add(new Mmul(sameDiff, input[0],args()[0],
                        MMulTranspose.builder().transposeA(false).transposeB(true).build()),args()[1]);
            else {
                forward = new Mmul(sameDiff, input[0],args()[0],
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
                    SDVariable.builder().sameDiff(sameDiff).varName("w")
                            .shape(new int[]{nOut,nIn}).weightInitScheme(
                            weightInitScheme).build(),
                    SDVariable.builder().sameDiff(sameDiff)
                            .varName("b")
                            .shape(new int[]{nOut,1})
                            .weightInitScheme(biasWeightInitScheme).build()
            };
        }
        else {
            return new DifferentialFunction[] {
                    SDVariable.builder().sameDiff(sameDiff)
                            .varName("w")
                            .shape(new int[]{nOut,nIn})
                            .weightInitScheme(new ZeroInitScheme('f'))
                            .build()
            };
        }
    }
}
