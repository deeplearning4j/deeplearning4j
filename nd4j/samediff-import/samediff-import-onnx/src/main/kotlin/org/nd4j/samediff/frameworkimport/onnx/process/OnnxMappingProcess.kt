/* Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.samediff.frameworkimport.onnx.process

import onnx.Onnx

import org.nd4j.samediff.frameworkimport.onnx.attributeValueTypeForOnnxAttribute
import org.nd4j.samediff.frameworkimport.process.AbstractMappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule

open class OnnxMappingProcess(inputFramework: String = "onnx",
                              frameworkVersion: String = "1.4",
                              inputFrameworkOpName: String,
                              opName: String,
                              opMappingRegistry: OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto,
                                      Onnx.NodeProto,
                                      Onnx.TensorProto,
                                      Onnx.TensorProto.DataType,
                                      Onnx.AttributeProto,
                                      Onnx.AttributeProto>,
                              tensorMappingRules: List<TensorMappingRule<Onnx.GraphProto,
                                      Onnx.NodeProto, Onnx.NodeProto,
                                      Onnx.AttributeProto, Onnx.AttributeProto,
                                      Onnx.TensorProto, Onnx.TensorProto.DataType>> = emptyList(),
                              inputIndexOverrides: Map<Int,Int> = emptyMap(),
                              attributeMappingRules: List<out AttributeMappingRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto,
                                      Onnx.TensorProto, Onnx.TensorProto.DataType>> = emptyList())
    : AbstractMappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>(
    inputFramework,
    frameworkVersion,
    inputFrameworkOpName,
    inputIndexOverrides,
    opName,
    opMappingRegistry,
    tensorMappingRules,
    attributeMappingRules) {
    override fun inputOpDefValueTypes(): Map<String, AttributeValueType> {
        val opDef = opMappingRegistry.lookupInputFrameworkOpDef(inputFrameworkOpName)
        val ret = HashMap<String,AttributeValueType>()
        opDef.attributeList.forEach { attributeProto ->
              ret[attributeProto.name] = attributeValueTypeForOnnxAttribute(attributeProto)
        }

        return ret
    }

}

