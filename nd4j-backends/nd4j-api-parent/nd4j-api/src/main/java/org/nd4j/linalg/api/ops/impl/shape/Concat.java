package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Slf4j
public class Concat extends DynamicCustomOp {
    private int concatDimension;

    @Override
    public String opName() {
        return "concat";
    }




    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        int idx = -1;
        int cnt = 0;
        int concatDimension = 0;
        for (int i = 0; i < nodeDef.getInputCount(); i++) {
            val input = nodeDef.getInput(i);
            val variable = initWith.getVariable(input);
            // concat dimension is only possible
            if (variable != null && variable.getArr() == null) {
                idx = cnt;
                if(variable.getShape() != null)
                    concatDimension = variable.getShape()[0];
                break;
            } else if (variable != null) {
                val arr = variable.getArr();
                if (arr.length() == 1) {
                    concatDimension = arr.getInt(0);
                    idx = cnt;
                    break;
                }
            }
            cnt++;
        }

        if (idx < 0)
            throw new ND4JIllegalStateException("Can't find dimension for concatenatiion");

        // if that's convolution graph, we should swap dimensions
        if (concatDimension == 3)
            concatDimension = 1;

        this.concatDimension = concatDimension;
        log.debug("Concat dimension: {}", concatDimension);

    }

    @Override
    public List<Integer> getIArguments() {
        return Collections.singletonList(concatDimension);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String onnxName() {
        return "Concat";
    }

    @Override
    public String tensorflowName() {
        return "Concat";
    }



    @Override
    public Op.Type opType() {
        return Op.Type.SHAPE;
    }
}
