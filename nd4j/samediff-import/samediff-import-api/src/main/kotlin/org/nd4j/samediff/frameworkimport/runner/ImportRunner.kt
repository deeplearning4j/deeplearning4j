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
package org.nd4j.samediff.frameworkimport.runner

import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface ImportRunner<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {
    fun <GRAPH_TYPE: GeneratedMessageV3,
            NODE_TYPE : GeneratedMessageV3,
            OP_DEF_TYPE : GeneratedMessageV3,
            TENSOR_TYPE : GeneratedMessageV3,
            ATTR_DEF_TYPE : GeneratedMessageV3,
            ATTR_VALUE_TYPE : GeneratedMessageV3,
            DATA_TYPE: ProtocolMessageEnum> initAttributes(
        df: DifferentialFunction,
        sd: SameDiff,
        descriptorAndContext: Pair<MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor>
    )
}