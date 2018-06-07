package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

public class TensorArrayV3 extends  BaseTensorOp {

    @Override
    public String tensorflowName() {
        return "TensorArrayV3";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

        val idd = nodeDef.getInput(nodeDef.getInputCount() - 1);
        NodeDef iddNode = null;
        for(int i = 0; i < graph.getNodeCount(); i++) {
            if(graph.getNode(i).getName().equals(idd)) {
                iddNode = graph.getNode(i);
            }
        }


        val arr = TFGraphMapper.getInstance().getNDArrayFromTensor("value",iddNode,graph);

        if (arr != null) {
            int idx = arr.getInt(0);
            addIArgument(idx);
        }

    }

    @Override
    public TensorList execute(SameDiff sameDiff) {
        val list = new TensorList(this.getOwnName());

        return list;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "tensorarrayv3";
    }


    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }
}
