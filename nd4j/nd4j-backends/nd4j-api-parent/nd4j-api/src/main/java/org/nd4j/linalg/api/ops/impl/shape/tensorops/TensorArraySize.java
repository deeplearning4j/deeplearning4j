/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.shape.tensorops;

import onnx.Onnx;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TensorArraySize extends BaseTensorOp {
   @Override
   public String[] tensorflowNames() {
      return new String[]{"TensorArraySize", "TensorArraySizeV2", "TensorArraySizeV3"};
   }


   @Override
   public String opName() {
      return "size_list";
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
   public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
   }

   @Override
   public List<DataType> calculateOutputDataTypes(List<DataType> inputDataType){
      //Size is always int32
      return Collections.singletonList(DataType.INT);
   }
}
