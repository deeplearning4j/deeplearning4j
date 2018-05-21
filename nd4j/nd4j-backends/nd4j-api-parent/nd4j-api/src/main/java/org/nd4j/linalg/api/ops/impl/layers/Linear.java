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
import org.nd4j.linalg.api.ops.Module;
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
                new ArrayList<Double>(), new ArrayList<Integer>(), new ArrayList<Module>());
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
        super(null, sameDiff, null, false, new ArrayList<Module>());
        this.weightInitScheme = weightInitScheme;
        this.biasWeightInitScheme = biasWeightInitScheme;

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
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        execSameDiff();
        return forward.doDiff(f1);
    }

    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>();
        ret.add(Shape.getMatrixMultiplyShape(inputArguments()[0].shape(),new long[]{nOut,nIn}));

        ret.add(Shape.getMatrixMultiplyShape(inputArguments()[0].shape(),inputArguments()[1].transpose().shape()));
        if(biasWeightInitScheme != null) {
            ret.add(new long[]{nOut,1});
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
        val inputArguments = inputArguments();
        if(inputArguments == null || inputArguments.length < 1) {
            throw new IllegalStateException("No arguments found.");
        }

        INDArray weights = inputArguments[0];
        INDArray right = inputArguments[1];

        val outputArguments = outputArguments();

        if(outputArguments == null || outputArguments.length < 1) {
            if(inputArguments.length == 1)
              addOutputArgument(inputs[0].mmul(weights.transpose()));
            else
                addOutputArgument(inputs[0].mmul(weights.transpose()).addiColumnVector(right));

        }
        else {
            inputs[0].mmul(weights.transpose(),outputArguments[0]);
        }

    }

    @Override
    public void execSameDiff(SDVariable... input) {
        val args = args();
        if(args == null || args.length == 0) {
            throw new IllegalStateException("No arguments found");
        }

        if(forward == null) {
            //bias needs to be added yet
            if(args.length > 1)
                forward =  f().add(new Mmul(sameDiff, input[0],args()[0],
                        MMulTranspose.builder()
                                .transposeA(false)
                                .transposeB(true)
                                .build()).outputVariables()[0],args()[1]);
            else {
                forward = new Mmul(sameDiff, input[0],args()[0],
                        MMulTranspose.builder().transposeA(false).transposeB(true).build());
            }

            this.outputVariables = forward.outputVariables();
        }


    }

    private static INDArray[] getParams(int nIn,
                                        int nOut,
                                        WeightInitScheme paramsScheme,
                                        WeightInitScheme biasInitScheme) {
        if(biasInitScheme != null) {
            return new INDArray[] {paramsScheme.create(new long[]{nOut,nIn}),biasInitScheme.create(new long[]{nOut,1})};
        }
        else {
            return new INDArray[] {paramsScheme.create(new long[]{nOut,nIn})};

        }
    }

    private DifferentialFunction[] getFunctionParams(int nIn,
                                                     int nOut) {
        if(biasWeightInitScheme != null) {
            return new DifferentialFunction[] {
                    SDVariable.builder().sameDiff(sameDiff).varName("w")
                            .shape(new long[]{nOut,nIn}).weightInitScheme(
                            weightInitScheme).build(),
                    SDVariable.builder().sameDiff(sameDiff)
                            .varName("b")
                            .shape(new long[]{nOut,1})
                            .weightInitScheme(biasWeightInitScheme).build()
            };
        }
        else {
            return new DifferentialFunction[] {
                    SDVariable.builder().sameDiff(sameDiff)
                            .varName("w")
                            .shape(new long[]{nOut,nIn})
                            .weightInitScheme(new ZeroInitScheme('f'))
                            .build()
            };
        }
    }
}
