package org.nd4j.linalg.api.ops.impl.transforms.temp;

import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ExternalErrorsFunction extends DifferentialFunction {

    private Map<String,INDArray> gradients;
    private Map<String,SDVariable> gradVariables;
    private SDVariable out;


    public ExternalErrorsFunction(SameDiff sd, List<SDVariable> inputs, Map<String,INDArray> gradients){
        super(sd, inputs.toArray(new SDVariable[inputs.size()]));
        if(gradients == null)
            gradients = new HashMap<>();
        this.gradients = gradients;
    }

    public ExternalErrorsFunction(){ }

    public void updateVariable(String str, INDArray gradient){
        gradients.put(str, gradient);
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        if(out == null){
            out = sameDiff.zero("dummyOutput", new long[]{1});
        }
        return new SDVariable[]{out};
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> out = new ArrayList<>();
        if (gradVariables == null) {
            gradVariables = new HashMap<>();
            for(SDVariable arg : args()){
                INDArray gradArr = gradients.get(arg.getVarName());
                SDVariable grad;
                if(gradArr != null){
                    grad = sameDiff.var(arg.getVarName() + "-externalGrad", gradArr);
                } else {
                    grad = sameDiff.var(arg.getVarName() + "-externalGrad", arg.getShape());
                }
                gradVariables.put(arg.getVarName(), grad);
                out.add(grad);
            }
        }
        return out;
    }


    public void updateBeforeExecution(){
        Preconditions.checkState(gradVariables != null, "Variables list is null - doDiff has not been called?");

        //Update external gradients ready for execution
        for(Map.Entry<String,SDVariable> e : gradVariables.entrySet()){
            INDArray extGradArray = gradients.get(e.getKey());
            if(extGradArray == null){
                throw new IllegalStateException("Cannot execute SameDiff instance with external errors: external gradient " +
                        "for variable " + e.getKey() + " has not been defined");
            }
            gradVariables.get(e.getKey()).setArray(extGradArray);
        }
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("Not supported: " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Not supported: " +  opName());
    }

    @Override
    public String opName(){
        return "ExternalErrorFnPrototype";
    }

    @Override
    public String toString(){
        return "ExternalErrorsFunction(" + (gradVariables != null ? gradVariables.keySet() : "") + ")";
    }
}
