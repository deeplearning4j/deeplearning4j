package org.nd4j.linalg.api.ops.impl.broadcast;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public class BiasAdd extends DynamicCustomOp {


    public BiasAdd() {}

    @Override
    public String opName() {
        return "biasadd";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

    }

    @Override
    public INDArray[] inputArguments() {
        val originalRet = super.inputArguments();
        val ret = new INDArray[2];
        if(!originalRet[0].isVector()) {
            ret[0] = originalRet[0];
            ret[1] = originalRet[1];
        }
        else {
            ret[0] = originalRet[1];
            ret[1] = originalRet[0];
        }
        return ret;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

    @Override
    public String onnxName() {
        return "BiasAdd";
    }

    @Override
    public String tensorflowName() {
        return "BiasAdd";
    }
}
