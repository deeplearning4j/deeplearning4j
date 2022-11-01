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
package org.nd4j.samediff.frameworkimport.ir

import org.nd4j.ir.TensorNamespace.DataType
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Neutral attribute conversion class for handling interop between different frameworks.
 *
 * @author Adam Gibson
 */
interface IRAttribute<ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum> {

    fun name(): String

    fun floatValue(): Float

    fun listFloatValue(): List<Float>

    fun tensorValue(): IRTensor<TENSOR_TYPE, DATA_TYPE>

    fun listTensorValue(): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    fun intValue(): Long

    fun listIntValue(): List<Long>

    fun boolValue(): Boolean

    fun listBoolValue(): List<Boolean>

    fun shapeValue(): List<Long>

    fun listDataTypes(): List<DataType>

    fun stringValue(): String

    fun listStringValue(): List<String>

    fun attributeValueType(): AttributeValueType

    fun dataTataTypeValue(): IRDataType<DATA_TYPE>

    fun internalAttributeDef(): ATTRIBUTE_TYPE

    /**
     * Returns the graph value. This value, when used should be cast to the proper type.
     */
    fun graphValue(registry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>): IRGraph<GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,ProtocolMessageEnum>


    fun listGraphValue(registry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>): List<IRGraph<GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,ProtocolMessageEnum>>


    fun internalAttributeValue(): ATTRIBUTE_VALUE_TYPE
}