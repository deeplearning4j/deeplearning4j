package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;

import java.util.Map;

public class TensorArrayConcatV3 extends BaseTensorOp {

    public TensorArrayConcatV3(String name, SameDiff sameDiff, SDVariable[] args){
        super(name, sameDiff, args);
    }
    public TensorArrayConcatV3(SameDiff sameDiff, SDVariable[] args){
        super(null, sameDiff, args);
    }

    public TensorArrayConcatV3(){}
   @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "TensorArrayConcatV3";
    }


    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "tensorarrayconcatv3";
    }

    @Override
    public TensorList execute(SameDiff sameDiff) {
       val list = getList(sameDiff);
       val array = list.concat();
       val name = this.getOwnName();
       sameDiff.updateVariable(name, array);

       return list;
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
