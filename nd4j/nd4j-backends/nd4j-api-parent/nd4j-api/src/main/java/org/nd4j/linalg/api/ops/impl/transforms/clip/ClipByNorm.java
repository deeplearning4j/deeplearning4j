package org.nd4j.linalg.api.ops.impl.transforms.clip;

import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ClipByNorm extends DynamicCustomOp {

    private double clipValue;

    public ClipByNorm() {

    }

    public ClipByNorm(SameDiff sameDiff, SDVariable x, double clipValue, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{x});
        this.clipValue = clipValue;
        this.dimensions = dimensions;
        addIArgument(dimensions);
        addTArgument(clipValue);
    }

    @Override
    public String opName() {
        return "clipbynorm";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //dOut/dIn is ??? if clipped, 1 otherwise
        int origRank = Shape.rankFromShape(arg().getShape());
        SDVariable l2norm = f().norm2(arg(), dimensions);
        SDVariable broadcastableNorm = f().reductionBroadcastableWithOrigShape(origRank, dimensions, l2norm);
        SDVariable isClippedBC = f().gte(broadcastableNorm, clipValue);
        SDVariable notClippedBC = isClippedBC.rsub(1.0);

//        SDVariable dnormdx = arg().div(broadcastableNorm);
//        SDVariable sqNorm = f().square(broadcastableNorm);
//        SDVariable dOutdInClipped = sqNorm.rdiv(-1).mul(dnormdx).mul(arg()) //-1/(norm2(x))^2 * x/norm2(x)
//                .add(broadcastableNorm.rdiv(1.0))
//                .mul(clipValue);

        SDVariable dOutdInClipped = f().neg(f().square(arg()).div(f().cube(broadcastableNorm))) //-x^2/(norm2(x))^3
                .add(broadcastableNorm.rdiv(1.0))   //+ 1/norm(x)
                .mul(clipValue).mul(isClippedBC);


        SDVariable ret = notClippedBC.add(dOutdInClipped).mul(grad.get(0));
        return Arrays.asList(ret);
    }
}
