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
package org.nd4j.samediff.frameworkimport.ir

import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface IROpDef<
        GRAPH_DEF: GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ARG_DEF_TYPE : GeneratedMessageV3, DATA_TYPE,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3>
        where DATA_TYPE: ProtocolMessageEnum {
    fun opName(): String

    fun internalValue(): OP_DEF_TYPE

    fun inputArgs(): List<IRArgDef<ARG_DEF_TYPE, DATA_TYPE>>

    fun outputArgs(): List<IRArgDef<ARG_DEF_TYPE, DATA_TYPE>>

    fun attributes(): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

}