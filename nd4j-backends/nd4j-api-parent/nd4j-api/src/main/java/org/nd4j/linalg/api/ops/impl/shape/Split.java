package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Map;

public class Split extends DynamicCustomOp {

    private int numSplit;

    @Override
    public String opName() {
        return "split";
    }

    @Override
    public String tensorflowName() {
        return "Split";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
       val numSplits = (int) attributesForNode.get("num_split").getI();
       this.numSplit = numSplits;
       addIArgument(numSplits);

    }
}
