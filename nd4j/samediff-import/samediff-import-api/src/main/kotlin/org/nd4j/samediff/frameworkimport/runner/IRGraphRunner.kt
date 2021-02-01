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

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface IRGraphRunner<
        GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE : ProtocolMessageEnum> {

    fun graph(): IRGraph<GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>

    fun run(inputs: Map<String, INDArray>): Map<String, INDArray>
}