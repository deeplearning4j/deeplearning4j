package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.list.compat.TensorList;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public abstract  class BaseTensorOp extends DynamicCustomOp {

    public BaseTensorOp(String name, SameDiff sameDiff, SDVariable[] args){
        super(name, sameDiff, args);
    }
    public BaseTensorOp(SameDiff sameDiff, SDVariable[] args){
        super(null, sameDiff, args);
    }

    public BaseTensorOp(){}

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

    protected INDArray getArgumentArray(int index) {
        val arg = this.arg(index);
        val array = this.sameDiff.getArrForVarName(arg.getVarName());

        return array;
    }

    protected TensorList getList(SameDiff sameDiff) {
        /**
         * First argument is TensorList.
         */
        val arg0 = this.arg(0);
        val tName = arg0.getVarName();

        val list = sameDiff.getListByName(tName);

        if (list == null)
            throw new ND4JIllegalStateException("There's no TensorList with name [" + tName + "] registered");

        return list;
    }

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
