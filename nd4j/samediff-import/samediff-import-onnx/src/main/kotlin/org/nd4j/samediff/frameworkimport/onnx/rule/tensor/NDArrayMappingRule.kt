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
package org.nd4j.samediff.frameworkimport.onnx.rule.tensor

import onnx.Onnx
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRTensor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.rule.MappingRule
import org.nd4j.samediff.frameworkimport.rule.tensor.BaseNDArrayMappingRule

@MappingRule("onnx","ndarraymapping","tensor")
class NDArrayMappingRule(mappingNamesToPerform: MutableMap<String,String>,
                         transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
    BaseNDArrayMappingRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto,
            Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {



    override fun createTensorProto(input: Onnx.TensorProto): TensorNamespace.TensorProto {
        return OnnxIRTensor(input).toArgTensor()
    }

    override fun isInputTensorName(inputName: String): Boolean {
        val onnxOp = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")[mappingProcess!!.inputFrameworkOpName()]!!
        return onnxOp.inputList.contains(inputName)
    }

    override fun isOutputTensorName(outputName: String): Boolean {
        val nd4jOpDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess!!.opName())
        return nd4jOpDescriptor.argDescriptorList.filter { inputDescriptor -> inputDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
            .map {inputDescriptor -> inputDescriptor.name }.contains(outputName)
    }
}