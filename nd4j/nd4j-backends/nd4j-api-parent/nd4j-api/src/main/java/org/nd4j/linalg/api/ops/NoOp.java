package org.nd4j.linalg.api.ops;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class NoOp extends DynamicCustomOp {

    public NoOp(){ }

    public NoOp(SameDiff sd, SDVariable in){
        super("noop", sd, new SDVariable[]{in});
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f1.get(0));
    }



    @Override
    public String opName() {
        return "noop";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        return "NoOp";
    }

    @Override
    public String tensorflowName() {
        return "NoOp";
    }

    @Override
    public int getNumOutputs(){
        return 1;
    }

    @Override
    public List<long[]> calculateOutputShape(){
        if(inputArguments != null && !inputArguments.isEmpty()){
            return Collections.singletonList(inputArguments.get(0).shape());
        }
        return Collections.emptyList();
    }
}
