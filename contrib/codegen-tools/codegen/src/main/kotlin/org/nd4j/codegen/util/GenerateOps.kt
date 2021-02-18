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

package org.nd4j.codegen.util

import org.nd4j.codegen.impl.java.JavaPoetGenerator
import org.nd4j.codegen.ops.Bitwise
import org.nd4j.codegen.ops.Random
import java.io.File

fun main() {
    val outDir = File("F:\\dl4j-builds\\deeplearning4j\\nd4j\\nd4j-backends\\nd4j-api-parent\\nd4j-api\\src\\main\\java\\")
    outDir.mkdirs()

    listOf(Bitwise(), Random()).forEach {
        val generator = JavaPoetGenerator()
        generator.generateNamespaceNd4j(it, null, outDir, it.name + ".java")
    }
}