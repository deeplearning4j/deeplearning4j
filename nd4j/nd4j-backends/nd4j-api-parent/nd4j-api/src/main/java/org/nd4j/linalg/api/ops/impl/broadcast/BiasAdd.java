package org.nd4j.linalg.api.ops.impl.broadcast;

import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
public class BiasAdd extends DynamicCustomOp {


    public BiasAdd(SameDiff sameDiff, SDVariable input, SDVariable bias) {
        super(null, sameDiff, new SDVariable[] {input, bias}, false);
    }

    @Override
    public String opName() {
        return "biasadd";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);

    }

    @Override
    public List<long[]> calculateOutputShape() {
        val args = args();
        for(int i = 0; i < args.length; i++)
            if(args[i].getShape() == null)
                return Collections.emptyList();
        val firstShape = ArrayUtil.prod(args[0].getShape());
        val secondShape = ArrayUtil.prod(args[1].getShape());

        if(firstShape > secondShape)
            return Arrays.asList(args[0].getShape());
        else
            return Arrays.asList(args[1].getShape());
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
