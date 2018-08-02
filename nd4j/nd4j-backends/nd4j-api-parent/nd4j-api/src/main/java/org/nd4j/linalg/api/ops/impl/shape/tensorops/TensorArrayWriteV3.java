/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.list.compat.TensorList;

public class TensorArrayWriteV3 extends BaseTensorOp {

   public TensorArrayWriteV3(String name, SameDiff sameDiff, SDVariable[] args){
      super(name, sameDiff, args);
   }
   public TensorArrayWriteV3(SameDiff sameDiff, SDVariable[] args){
      super(null, sameDiff, args);
   }

   public TensorArrayWriteV3(){}
   @Override
   public String tensorflowName() {
      return "TensorArrayWriteV3";
   }

   @Override
   public TensorList execute(SameDiff sameDiff) {
      val list = getList(sameDiff);

      val ids =getArgumentArray(1).getInt(0);
      val array = getArgumentArray(2);

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
