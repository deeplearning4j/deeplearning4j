package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Backprop op for concat
 *
 * @author Alex Black
 */
@Slf4j
public class ConcatBp extends DynamicCustomOp {
    private int concatDimension;

    public ConcatBp(){

    }

    /**
     *
     * @param sameDiff
     * @param concatDimension
     * @param inputsAndGrad     Original inputs, followed by output gradient
     */
    public ConcatBp(SameDiff sameDiff, int concatDimension, SDVariable... inputsAndGrad){
        super(null, sameDiff, inputsAndGrad);
        addIArgument(concatDimension);
        this.concatDimension = concatDimension;
    }

    @Override
    public String opName() {
        return "concat_bp";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        //No op
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        //No op
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
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public int getNumOutputs(){
        return args().length - 1;
    }
}
