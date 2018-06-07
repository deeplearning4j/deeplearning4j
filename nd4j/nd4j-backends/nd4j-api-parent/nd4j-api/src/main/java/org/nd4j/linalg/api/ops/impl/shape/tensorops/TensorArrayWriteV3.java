package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
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
      val list = getList(sameDiff);

      // we know that arg 0
      val args = this.args();

      val varIdx = args[1];
      val varArray = args[2];

      val ids = (int) sameDiff.getArrForVarName(varIdx.getVarName()).getDouble(0);
      val array = sameDiff.getArrForVarName(varArray.getVarName());

      list.put(ids, array);

      return list;
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
