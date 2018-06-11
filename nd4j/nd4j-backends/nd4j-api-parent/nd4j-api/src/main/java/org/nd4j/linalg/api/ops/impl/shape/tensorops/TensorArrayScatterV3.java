package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.list.compat.TensorList;

import java.util.Map;

public class TensorArrayScatterV3 extends BaseTensorOp {

    @Override
    public String tensorflowName() {
        return "TensorArrayScatterV3";
    }

    @Override
    public TensorList execute(SameDiff sameDiff) {
        val list = getList(sameDiff);

        val indices = this.getArgumentArray(1);
        val source = this.getArgumentArray(2);

        val axis = ArrayUtil.range(1, source.rank());
        val numTads = source.tensorssAlongDimension(axis);

        for (int e = 0; e < indices.length(); e++) {
            val cIdx = indices.getInt(e);

            val array = source.tensorAlongDimension(cIdx, axis).dup(source.ordering());
            list.put(cIdx, array);
        }

        return list;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "tensorarrayscatterv3";
    }


    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
    }
}
