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
package org.nd4j.samediff.frameworkimport.tensorflow.rule.attribute

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.argDescriptorType
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.isNd4jTensorName
import org.nd4j.samediff.frameworkimport.isOutputFrameworkAttributeName
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.rule.MappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeScalarNDArrayAttribute
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRAttr
import org.nd4j.samediff.frameworkimport.tensorflow.ir.isTensorflowAttributeName
import org.nd4j.samediff.frameworkimport.tensorflow.ir.isTensorflowTensorName
import org.nd4j.samediff.frameworkimport.tensorflow.ir.tensorflowAttributeValueTypeFor
import org.tensorflow.framework.*

@MappingRule("tensorflow","attributescalarndarrayattribute","attribute")
class TensorflowAttributeScalarNDArrayAttribute(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                                transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    AttributeScalarNDArrayAttribute<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow")[mappingProcess.inputFrameworkOpName()]!!

        return isTensorflowTensorName(name, opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow")[mappingProcess.inputFrameworkOpName()]!!

        return isTensorflowAttributeName(name, opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow")[mappingProcess.inputFrameworkOpName()]!!

        return tensorflowAttributeValueTypeFor(attributeName = name, opDef = opDef)
    }
}