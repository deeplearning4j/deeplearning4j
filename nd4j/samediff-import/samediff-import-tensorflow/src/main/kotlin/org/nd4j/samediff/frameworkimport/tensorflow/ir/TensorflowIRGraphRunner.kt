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
package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.runner.IRGraphRunner
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner
import org.tensorflow.framework.*

class TensorflowIRGraphRunner(irGraph: TensorflowIRGraph, inputNames: List<String>, outputNames: List<String>):
    IRGraphRunner<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {

    val irGraph = irGraph
    val graphRunner: GraphRunner
    init {
        graphRunner = GraphRunner.builder()
            .graphBytes(irGraph.graphDef.toByteArray())
            .inputNames(inputNames)
            .outputNames(outputNames)
            .build()
    }


    override fun graph(): IRGraph<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {
        return irGraph
    }

    override fun run(inputs: Map<String, INDArray>): Map<String, INDArray> {
        return graphRunner.run(inputs)
    }

}