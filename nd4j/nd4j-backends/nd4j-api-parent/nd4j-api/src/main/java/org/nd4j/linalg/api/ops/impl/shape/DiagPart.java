package org.nd4j.linalg.api.ops.impl.shape;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;


/**
 * Return the diagonal part of a tensor. The input tensor has to
 * have dimensions [d1,..., dk, d1,..., dk], so that the diagonal
 * blocks have shape [d1,..., dk].
 * <p>
 * A simple special case of this is returning the diagonal of a
 * matrix as vector.
 *
 * @author Max Pumperla
 */
public class DiagPart extends DynamicCustomOp {

    public DiagPart() {
    }

    public DiagPart(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(null, sameDiff, args, inPlace);
    }

    public DiagPart(INDArray in, INDArray out){
        super(null, in, out, null, null);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable grad = i_v.get(0);
        SDVariable ret = sameDiff.diag(grad);
        return Collections.singletonList(ret);
    }

    @Override
    public String opName() {
        return "diag_part";
    }


    @Override
    public String tensorflowName() {
        return "DiagPart";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        super.initFromOnnx(node, initWith, attributesForNode, graph);
    }

}
