/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.mixins

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.api.Exactly
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

val broadcastingDoc = Mixin("broadcastingDoc"){
    Doc(Language.ANY, DocScope.ALL){
        //TODO: finalize content for this broadcasting mixin doc.
        """
                Note: supports broadcasting if x and y have different shapes and are broadcastable.
                For example, if X has shape [1,10] and Y has shape [5,10] then op(X,Y) has output shape [5,10]
                Broadcast rules are the same as NumPy: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
            """.trimIndent()
    }
}

val transform = Mixin("transform"){
    legacy = true
    Input(DataType.NUMERIC, "x") { description = "Input variable" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val transformArithmetic = Mixin("transformArithmetic"){
    useMixin(transform)
    legacy = false
    Input(DataType.NUMERIC, "y") { description = "Input variable" }
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"
}

val transformCustom2 = Mixin("transformCustom2"){
    Input(DataType.NUMERIC, "x") { description = "First input variable, x" }
    Input(DataType.NUMERIC, "y") { description = "Second input variable, y" }
    Output(DataType.NUMERIC, "out"){ description = "Output"}
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
}

val transformStrict = Mixin("transformStrict"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
}

val transformSame = Mixin("transformSame"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
}

val transformBool = Mixin("transformBool"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
}

val transformAny = Mixin("transformAny"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.any"
}

val transformFloating = Mixin("transformFloating"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.floating"
}

val scalar = Mixin("scalar"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
    Input(DataType.NUMERIC, "x") { description = "Input variable" }
    Arg(DataType.NUMERIC, "value") { description = "Scalar value for op" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val reduce = Mixin("reduce"){
    legacy = true
    Input(DataType.NUMERIC, "in") { description = "Input variable" }
    Arg(DataType.INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(DataType.NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
}

val reduceFloating = Mixin("reduceFloating"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
}

val reduceSame = Mixin("reduceSame"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
}

val reduceLong = Mixin("reduceLong"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
}

val reduce3 = Mixin("reduce3"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
    Input(DataType.NUMERIC, "x") { description = "Input variable x" }
    Input(DataType.NUMERIC, "y") { description = "Input variable y" }
    Arg(DataType.INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to calculate %OPNAME% over" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val indexAccum = Mixin("indexAccum"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
    val input = Input(DataType.NUMERIC, "in") { description = "Input variable" }
    val keepDims = Arg(DataType.BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions"; defaultValue = false }
    val dims = Arg(DataType.INT, "dimensions"){ count = AtLeast(1); isVargarg = true; description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(DataType.NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }

    Signature(input, dims)
    AllParamSignature(withOutput = false)
}

val indexAccumCustom = Mixin("indexAccumCustom"){
    javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum.custom"
    val input = Input(DataType.NUMERIC, "in") { description = "Input variable" }
    val keepDims = Arg(DataType.BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions"; defaultValue = false }
    val dims = Arg(DataType.INT, "dimensions"){ count = AtLeast(1); isVargarg = true; description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(DataType.NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }

    Signature(input, dims)
    AllParamSignature(withOutput = false)
}