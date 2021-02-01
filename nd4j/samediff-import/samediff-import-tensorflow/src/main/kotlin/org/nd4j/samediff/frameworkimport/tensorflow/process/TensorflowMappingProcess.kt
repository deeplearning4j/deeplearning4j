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
package org.nd4j.samediff.frameworkimport.tensorflow.process


import org.nd4j.common.base.Preconditions
import org.nd4j.samediff.frameworkimport.process.AbstractMappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.nd4j.samediff.frameworkimport.tensorflow.ir.attributeValueTypeForTensorflowAttribute
import org.tensorflow.framework.*

open class TensorflowMappingProcess(inputFramework: String = "tensorflow",
                                    frameworkVersion: String = "2.3",
                                    inputFrameworkOpName: String,
                                    opName: String,
                                    opMappingRegistry: OpMappingRegistry<GraphDef,
                                            NodeDef, OpDef,
                                            TensorProto, DataType, OpDef.AttrDef, AttrValue>,
                                    tensorMappingRules: List<TensorMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue, TensorProto, DataType>> = emptyList(),
                                    attributeMappingRules: List<AttributeMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue,
                                            TensorProto, DataType>> = emptyList(),
                                    inputIndexOverrides: Map<Int,Int> = emptyMap())
    : AbstractMappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef,
        AttrValue, DataType>(
    inputFramework,
    frameworkVersion,
    inputFrameworkOpName,
    inputIndexOverrides,
    opName,
    opMappingRegistry,
    tensorMappingRules,
    attributeMappingRules) {
    override fun inputOpDefValueTypes(): Map<String, AttributeValueType> {
        Preconditions.checkNotNull(inputFrameworkOpName,"No input framework op def name found!")
        val opDef = opMappingRegistry.lookupInputFrameworkOpDef(inputFrameworkOpName)
        val retMap = HashMap<String,AttributeValueType>()
        opDef.attrList.forEach { attrDef ->
            retMap[attrDef.name] = attributeValueTypeForTensorflowAttribute(attrDef)
        }

        return retMap

    }

}


