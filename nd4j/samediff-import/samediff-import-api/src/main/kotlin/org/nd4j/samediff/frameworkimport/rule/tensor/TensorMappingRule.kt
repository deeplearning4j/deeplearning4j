/* ******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.nd4j.samediff.frameworkimport.rule.tensor

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface TensorMappingRule<GRAPH_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {


    fun initWithMappingProcess(mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)


    fun name(): String


    fun serialize(): MapperNamespace.MappingRule


    fun mappingNamesToPerform(): Map<String,String>

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convertInput(mappingContext: MappingContext<GRAPH_TYPE, NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor>


    fun inputArgumentMappings(): Map<String, String>

    fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE>

    fun isInputTensorName(inputName: String): Boolean

    fun isOutputTensorName(outputName: String): Boolean

    fun inputFrameworkOpName(): String

    fun modifyInputFrameworkOpName(inputFrameworkOpName: String)

}