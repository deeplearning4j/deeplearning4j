package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;

public class TensorArrayWriteV3 extends BaseTensorOp {

   @Override
   public String tensorflowName() {
      return "TensorArrayWriteV3";
   }

   @Override
   public TensorList execute(SameDiff sameDiff) {
      return null;
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
