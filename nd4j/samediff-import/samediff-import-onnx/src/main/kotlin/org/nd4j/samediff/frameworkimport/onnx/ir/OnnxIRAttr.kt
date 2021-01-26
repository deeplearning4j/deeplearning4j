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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType

class OnnxIRAttr(inputAttributeDef: Onnx.AttributeProto, inputAttributeValue: Onnx.AttributeProto):
    IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.floatsList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        return attributeValue.intsList
    }

    override fun boolValue(): Boolean {
        return attributeValue.i > 0
    }

    override fun listBoolValue(): List<Boolean> {
        TODO("Implement")
    }

    override fun attributeValueType(): AttributeValueType {
        when(attributeDef.type) {
            Onnx.AttributeProto.AttributeType.STRING -> return AttributeValueType.STRING
            Onnx.AttributeProto.AttributeType.STRINGS -> return AttributeValueType.LIST_STRING
            Onnx.AttributeProto.AttributeType.INT -> return AttributeValueType.INT
            Onnx.AttributeProto.AttributeType.INTS -> return AttributeValueType.LIST_INT
            Onnx.AttributeProto.AttributeType.FLOAT -> return AttributeValueType.FLOAT
            Onnx.AttributeProto.AttributeType.FLOATS -> return AttributeValueType.LIST_FLOAT
            Onnx.AttributeProto.AttributeType.TENSOR -> return AttributeValueType.TENSOR
            Onnx.AttributeProto.AttributeType.TENSORS -> return AttributeValueType.LIST_TENSOR
        }

        return AttributeValueType.INVALID
    }



    override fun internalAttributeDef(): Onnx.AttributeProto {
        return attributeDef
    }

    override fun internalAttributeValue(): Onnx.AttributeProto {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return attributeValue.tensorsList.map {
                input ->
            OnnxIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRTensor(attributeValue.t)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.stringsList.map { it.toStringUtf8() }
    }

    override fun dataTataTypeValue(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[attributeDef.t.dataType.ordinal])
    }

}