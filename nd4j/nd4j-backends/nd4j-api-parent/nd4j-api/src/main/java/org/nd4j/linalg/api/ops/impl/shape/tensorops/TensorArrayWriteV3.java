package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import org.nd4j.linalg.api.ops.Op;

public class TensorArrayWriteV3 extends BaseTensorOp {

   @Override
   public String tensorflowName() {
      return "TensorArrayWriteV3";
   }



   @Override
   public String opName() {
      return "tensorarraywritev3";
   }

   @Override
   public Op.Type opType() {
      return Op.Type.CUSTOM;
   }
}
