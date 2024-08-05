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
package org.nd4j.samediff.frameworkimport

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum


interface FrameworkImporter {

    /**
     * Run the import process based on the given file.
     * Optionally, a user may pass in dynamicVariables
     * which are considered to be dynamic variables
     * that may be needed to compute shapes.
     *
     * If dynamicVariables is specified and a variable is specified,
     * the dynamicVariables are a way of passing in dynamic variables
     * that maybe needed for things like shape calculations.
     *
     * Usually, this will be the placeholders as inputs in to a graph.
     * A user may use [suggestDynamicVariables] to pass in variables.
     *
     * A user may also pass in an optional boolean of suggestDynamicVariables of true or false
     * which will handle automatically creating the dynamic variables that maybe needed by the graph
     * for import.
     */
    fun runImport(
        fileName: String,
        dynamicVariables: Map<String, INDArray> = emptyMap(),
        suggestDynamicVariables: Boolean = false,
        trackVariableChanges: Boolean = false
    ): SameDiff

    /**
     * Parses the model and looks for inputs or placeholders that maybe needed in the graph.
     * This maybe required for importing certain models where placeholders are specified
     * and shapes need to be computed. This will return a map of ones with the appropriate shapes
     * and names as a map that the user may pass in to [runImport]
     *
     */
    fun suggestDynamicVariables(fileName: String): Map<String,INDArray>

    /**
     * Suggests dynamic variables but on an in memory graph instead of a file.
     * For more information see [suggestDynamicVariables]
     */
    fun suggestDynamicVariables(irGraph: IRGraph<GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,
            GeneratedMessageV3,GeneratedMessageV3,GeneratedMessageV3,ProtocolMessageEnum>): Map<String,INDArray>


}