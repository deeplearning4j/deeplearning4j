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
import org.nd4j.samediff.frameworkimport.ir.IRArgDef
import org.nd4j.samediff.frameworkimport.ir.IRDataType

class OnnxIRArgDef(input: Onnx.NodeProto): IRArgDef<Onnx.NodeProto, Onnx.TensorProto.DataType> {
    private val argDefValue = input

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRArgDef(argDefValue).dataType()
    }

    override fun name(): String {
        return argDefValue.name
    }

    override fun description(): String {
        return argDefValue.docString
    }

    override fun internalValue(): Onnx.NodeProto {
        return argDefValue
    }

    override fun indexOf(): Integer {
        TODO("Not yet implemented")
    }

}