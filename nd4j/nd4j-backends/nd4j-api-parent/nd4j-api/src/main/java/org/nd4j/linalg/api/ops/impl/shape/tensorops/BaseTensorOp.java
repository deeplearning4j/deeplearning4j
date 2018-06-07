package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public abstract  class BaseTensorOp extends DynamicCustomOp {


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val inputOne = nodeDef.getInput(1);
        val varFor = initWith.getVariable(inputOne);
        val nodeWithIndex = TFGraphMapper.getInstance().getNodeWithNameFromGraph(graph,inputOne);
        val var = TFGraphMapper.getInstance().getArrayFrom(nodeWithIndex,graph);
        if(var != null) {
            val idx = var.getInt(0);
            addIArgument(idx);
        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Differentiation not supported yet.");

    }

    public abstract TensorList execute(SameDiff sameDiff);

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String toString() {
        return opName();
    }


}
