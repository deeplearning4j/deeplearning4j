package org.nd4j.linalg.api.ops.impl.transforms;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Assign op: x = y, with broadcast as required
 */
public class Assign extends DynamicCustomOp {

    public Assign(){

    }

    public Assign(INDArray[] inputs, INDArray[] outputs) {
        super(null,inputs, outputs);
    }

    @Override
    public void addIArgument(int... arg) {
        super.addIArgument(arg);
    }

    public Assign(SameDiff sameDiff, SDVariable x, SDVariable y){
        super(null, sameDiff, new SDVariable[]{x,y});
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String opName() {
        return "assign";
    }

    @Override
    public String onnxName() {
        return "GivenTensorFill";
    }

    @Override
    public String tensorflowName() {
        return "Assign";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1){
        //TODO replace with assign backprop op from libnd4j (that handles the broadcast case properly)
        return Arrays.asList(f().zerosLike(larg()), f1.get(0));
    }
}
