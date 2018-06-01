package org.nd4j.linalg.api.ops.impl.shape;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
public class MergeSum extends DynamicCustomOp {

    public MergeSum(SameDiff sameDiff, SDVariable... inputs) {
        super(null, sameDiff, inputs);
    }

    public MergeSum(){ }

    @Override
    public String opName() {
        return "mergesum";
    }



    @Override
    public List<long[]> calculateOutputShape() {
        List<long[]> ret = new ArrayList<>(1);
        ret.add(arg().getShape());
        return ret;
    }


    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        // noop
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "AddN";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable gradient = sameDiff.setupFunction(i_v.get(0));
        List<SDVariable> ret = new ArrayList<>();
        for (int i = 0; i < args().length; i++)
            ret.add(gradient);
        return ret;
    }

}
