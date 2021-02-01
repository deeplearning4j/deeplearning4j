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
package org.nd4j.samediff.frameworkimport.onnx.process

import onnx.Onnx
import org.nd4j.ir.MapperNamespace
import org.nd4j.samediff.frameworkimport.process.AbstractMappingProcessLoader
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule

class OnnxMappingProcessLoader(opMappingRegistry:
                               OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,
                                       Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,
                                       Onnx.AttributeProto>): AbstractMappingProcessLoader<Onnx.GraphProto,
        Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto,
        Onnx.TensorProto.DataType>(opMappingRegistry) {


    override fun frameworkName(): String {
       return "onnx"
    }

    override fun instantiateMappingProcess(
        inputFrameworkOpName: String,
        opName: String,
        attributeMappingRules: List<AttributeMappingRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>>,
        tensorMappingRules: List<TensorMappingRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>>,
        opMappingRegistry: OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>,
        indexOverrides: Map<Int, Int>
    ): MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
       return OnnxMappingProcess(
           inputFrameworkOpName = inputFrameworkOpName,
           opName =  opName,
           attributeMappingRules = attributeMappingRules,
           tensorMappingRules = tensorMappingRules,
           opMappingRegistry = opMappingRegistry,
           inputIndexOverrides = indexOverrides
       )
    }

}