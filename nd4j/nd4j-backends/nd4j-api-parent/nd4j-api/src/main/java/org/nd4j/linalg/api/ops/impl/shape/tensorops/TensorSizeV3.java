package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.list.compat.TensorList;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TensorSizeV3 extends BaseTensorOp {
   @Override
   public String tensorflowName() {
      return "TensorArraySizeV3";
   }


   @Override
   public String opName() {
      return "tensorarraysizev3";
   }


   @Override
   public TensorList execute(SameDiff sameDiff) {
      val list = getList(sameDiff);

      val result = Nd4j.trueScalar(list.size());

      // storing our size
      sameDiff.putArrayForVarName(this.getOwnName(), result);

      return list;
   }

   @Override
   public List<long[]> calculateOutputShape() {
      // output is scalar only
      return Collections.singletonList(new long[]{});
   }

   @Override
   public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
      super.initFromTensorFlow(nodeDef, initWith, attributesForNode, graph);
   }

   @Override
   public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
      return super.mappingsForFunction();
   }

   @Override
   public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
   }
}
