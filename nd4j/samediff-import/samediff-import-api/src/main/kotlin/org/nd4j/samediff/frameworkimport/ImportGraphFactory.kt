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
package org.nd4j.samediff.frameworkimport

import org.nd4j.common.config.ND4JClassLoading
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.util.*

class ImportGraphFactory {

    val frameworkImportGraphInstances = HashMap<String,ImportGraphHolder>()

    init {
        val loaded = ServiceLoader.load(ImportGraphHolder::class.java,ND4JClassLoading.getNd4jClassloader())
        val iter = loaded.iterator()
        while(iter.hasNext()) {
            val next = iter.next()
            frameworkImportGraphInstances[next.frameworkName()] = next
        }
    }

    fun  <GRAPH_TYPE: GeneratedMessageV3,
            NODE_TYPE : GeneratedMessageV3,
            OP_DEF_TYPE : GeneratedMessageV3,
            TENSOR_TYPE : GeneratedMessageV3,
            ATTR_DEF_TYPE : GeneratedMessageV3,
            ATTR_VALUE_TYPE : GeneratedMessageV3,
            DATA_TYPE: ProtocolMessageEnum> createImportGraph(frameworkName: String): ImportGraph<
            GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTR_DEF_TYPE,
            ATTR_VALUE_TYPE,
            DATA_TYPE> {
        return frameworkImportGraphInstances[frameworkName]!!.createImportGraph()
    }


}