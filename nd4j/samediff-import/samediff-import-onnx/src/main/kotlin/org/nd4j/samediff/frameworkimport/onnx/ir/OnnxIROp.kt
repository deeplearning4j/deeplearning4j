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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.samediff.frameworkimport.ir.IRArgDef
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IROpDef

class OnnxIROp(input: Onnx.NodeProto):
    IROpDef<Onnx.GraphProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.NodeProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto> {

    val opDef = input

    override fun attributes(): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return opDef.attributeList.map {
            OnnxIRAttr(it, Onnx.AttributeProto.getDefaultInstance())
        }
    }

    override fun opName(): String {
        return opDef.name
    }

    override fun internalValue(): Onnx.NodeProto {
        return opDef
    }

    override fun inputArgs(): List<IRArgDef<Onnx.NodeProto, Onnx.TensorProto.DataType>> {
        return opDef.inputList.map {
            OnnxIRArgDef(opDef)
        }
    }

    override fun outputArgs(): List<IRArgDef<Onnx.NodeProto, Onnx.TensorProto.DataType>> {
        return opDef.outputList.map {
            OnnxIRArgDef(opDef)
        }
    }

}