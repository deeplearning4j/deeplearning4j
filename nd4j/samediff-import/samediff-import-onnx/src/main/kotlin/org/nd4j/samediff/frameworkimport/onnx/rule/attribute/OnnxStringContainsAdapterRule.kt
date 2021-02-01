/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.samediff.frameworkimport.onnx.rule.attribute

import onnx.Onnx
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.*
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRAttr
import org.nd4j.samediff.frameworkimport.onnx.isOnnxAttributeName
import org.nd4j.samediff.frameworkimport.onnx.isOnnxTensorName
import org.nd4j.samediff.frameworkimport.onnx.onnxAttributeTypeFor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.rule.MappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.rule.attribute.StringContainsAdapterRule

@MappingRule("onnx","stringcontains","attribute")
class OnnxStringContainsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    StringContainsAdapterRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
      val onnxOp = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")[mappingProcess.inputFrameworkOpName()]!!

        return isOnnxTensorName(name, onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
      val onnxOp = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")[mappingProcess.inputFrameworkOpName()]!!

        return isOnnxAttributeName(name, onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
      val onnxOp = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")[mappingProcess.inputFrameworkOpName()]!!

        return onnxAttributeTypeFor(name, onnxOp)
    }

}