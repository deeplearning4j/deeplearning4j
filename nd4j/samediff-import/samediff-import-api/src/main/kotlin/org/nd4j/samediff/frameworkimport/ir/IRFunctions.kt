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

package org.nd4j.samediff.frameworkimport.ir

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

fun <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE : ProtocolMessageEnum> importInfoForEachNodeInGraph (
    graph: IRGraph<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>,
    dynamicVariables: MutableMap<String, TENSOR_TYPE>)
        :  Map<String,Pair<MappingContext<GRAPH_TYPE,
        NODE_TYPE,OP_DEF_TYPE,
        TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,
        DATA_TYPE>, OpNamespace.OpDescriptor>> {

    val opMappingRegistry = graph.opMappingRegistry()

    val ret = HashMap<String,Pair<MappingContext<GRAPH_TYPE,
            NODE_TYPE, OP_DEF_TYPE,
            TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>, OpNamespace.OpDescriptor>>()

    graph.nodeList().forEach { node ->
        val name = node.nodeName()
        val opMappingProcess =  opMappingRegistry.lookupOpMappingProcess(node.opName())
        val opDefLookup = opMappingRegistry.lookupInputFrameworkOpDef(node.opName())
        val mappingContext = graph.createMappingContext(
            opDef = opDefLookup,
            node = graph.nodeByName(node.nodeName()),
            dynamicVariables = dynamicVariables
        )

        val applied = opMappingProcess.applyProcess(mappingContext)
        ret[name] = applied
    }

    return ret
}
