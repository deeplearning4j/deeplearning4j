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

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.Range


fun Linalg() =  Namespace("Linalg") {
    //val namespaceJavaPackage = "org.nd4j.linalg"

    Op("Cholesky") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms"
        javaOpClass = "Cholesky"
        Input(DataType.NUMERIC, "input") { description = "Input tensor with inner-most 2 dimensions forming square matrices" }
        Output(DataType.NUMERIC, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes the Cholesky decomposition of one or more square matrices.
            """.trimIndent()
        }
    }

    Op("Lstsq") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lstsq"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.FLOATING_POINT, "l2_reguralizer") {description = "regularizer"}
        Arg(DataType.BOOL, "fast") {description = "fast mode, defaults to True"; defaultValue = true}
        Output(DataType.FLOATING_POINT, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for linear squares problems.
            """.trimIndent()
        }
    }

    Op("Solve") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "LinearSolve"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.BOOL, "adjoint") {description = "adjoint mode, defaults to False"; defaultValue = false}
        Output(FLOATING_POINT, "output"){ description = "Output tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for systems of linear equations.
            """.trimIndent()
        }
    }

    Op("TriangularSolve") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "TriangularSolve"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.BOOL, "lower") {description = "defines whether innermost matrices in matrix are lower or upper triangular"}
        Arg(DataType.BOOL, "adjoint") {description = "adjoint mode"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for systems of linear questions.
            """.trimIndent()
        }
    }

    Op("Lu") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lu"

        Input(DataType.NUMERIC, "input") {description = "input tensor"}
        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes LU decomposition.
            """.trimIndent()
        }
    }

    Op("Matmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        javaOpClass = "Mmul"

        Input(DataType.NUMERIC, "a") {description = "input tensor"}
        Input(DataType.NUMERIC, "b") {description = "input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Performs matrix mutiplication on input tensors.
            """.trimIndent()
        }
    }

    Op("Qr") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Qr"

        Input(DataType.NUMERIC, "input") {description = "input tensor"}
        Arg(DataType.BOOL, "full") {description = "full matrices mode"; defaultValue = false}
        Output(FLOATING_POINT, "outputQ")
        Output(FLOATING_POINT, "outputR")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes the QR decompositions of input matrix.
            """.trimIndent()
        }
    }

    Op("MatrixBandPart") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "MatrixBandPart"

        Input(DataType.NUMERIC, "input") { description = "input tensor" }
        Arg(DataType.INT, "minLower") { description = "lower diagonal count" }
        Arg(DataType.INT, "maxUpper") { description = "upper diagonal count" }
        Output(DataType.FLOATING_POINT, "output1")
        Output(DataType.FLOATING_POINT, "output2")

        Doc(Language.ANY, DocScope.ALL){
            """
             Copy a tensor setting outside a central band in each innermost matrix.
            """.trimIndent()
        }
    }

    Op("cross") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Cross"

        Input(DataType.NUMERIC, "a") {"Input tensor a"}
        Input(DataType.NUMERIC, "b") {"Input tensor b"}
        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes pairwise cross product.
            """.trimIndent()
        }
    }

    Op("diag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Diag"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates diagonal tensor.
            """.trimIndent()
        }
    }

    Op("diag_part") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "DiagPart"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates diagonal tensor.
            """.trimIndent()
        }
    }

    Op("logdet") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Logdet"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates log of determinant.
            """.trimIndent()
        }
    }

    Op("svd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Svd"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Arg(DataType.BOOL, "fullUV") {"Full matrices mode"}
        Arg(DataType.BOOL, "computeUV") {"Compute U and V"}
        Arg(DataType.INT, "switchNum") {"Switch number"; defaultValue = 16}
        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates singular value decomposition.
            """.trimIndent()
        }
    }

    Op("tri") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Tri"

        Arg(DATA_TYPE, "dataType") { description = "Data type"; defaultValue = org.nd4j.linalg.api.buffer.DataType.FLOAT }
        Arg(INT, "row") {"Number of rows in the array"; }
        Arg(INT, "column") {"Number of columns in the array";  }
        Arg(INT, "diagonal") {"The sub-diagonal at and below which the array is filled. k = 0 is the main diagonal, while k < 0 is below it, and k > 0 is above. The default is 0."; defaultValue =  0}


        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             An array with ones at and below the given diagonal and zeros elsewhere.
            """.trimIndent()
        }
    }

    Op("triu") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Triu"
        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Arg(DataType.INT, "diag") {"diagonal"; defaultValue = 0}

        Output(FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Upper triangle of an array. Return a copy of a input tensor with the elements below the k-th diagonal zeroed.
            """.trimIndent()
        }
    }
    
    Alias(SDBaseOps(), "mmul")
}