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

import org.nd4j.samediff.frameworkimport.process.AbstractMappingProcessLoader
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.tensorflow.framework.*

class TensorflowMappingProcessLoader(opMappingRegistry: OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>):
    AbstractMappingProcessLoader<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>(opMappingRegistry) {


    override fun frameworkName(): String {
        return "tensorflow"
    }

    override fun instantiateMappingProcess(
        inputFrameworkOpName: String,
        opName: String,
        attributeMappingRules: List<AttributeMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>>,
        tensorMappingRules: List<TensorMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>>,
        opMappingRegistry: OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>,
        indexOverrides: Map<Int, Int>
    ): MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {
        return TensorflowMappingProcess(
            inputFrameworkOpName = inputFrameworkOpName,
            opName = opName,
            attributeMappingRules = attributeMappingRules,
            tensorMappingRules = tensorMappingRules,
            opMappingRegistry = opMappingRegistry,
            inputIndexOverrides = indexOverrides)

    }
}