package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.HashMap;
import java.util.Map;

/**
 * GatherND op
 */
@NoArgsConstructor
public class GatherNd extends DynamicCustomOp {


    public GatherNd(SameDiff sameDiff, SDVariable input, SDVariable indices, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
    }

    @Override
    public String opName() {
        return "gather_nd";
    }

    @Override
    public String onnxName() {
        return "GatherND";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"GatherNd"};
    }
}
