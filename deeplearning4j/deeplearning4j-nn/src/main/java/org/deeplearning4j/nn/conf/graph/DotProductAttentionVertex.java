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
package org.deeplearning4j.nn.conf.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = false)
public class DotProductAttentionVertex extends SameDiffVertex {

    private double scaleFactor;
    private double dropoutProbability;
    private boolean useCausalMask;
    private boolean training;

    @Override
    public GraphVertex clone() {
        return null;
    }

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable, Map<String, SDVariable> maskVars) {
        final SDVariable queries = layerInput.get("queries");
        final SDVariable keys = layerInput.get("keys");
        final SDVariable values = layerInput.get("values");
        final SDVariable qMask = maskVars  != null && maskVars.containsKey("qMask") ? maskVars.get("qMask") : null;
        final SDVariable vMask = maskVars != null  && maskVars.containsKey("vMask")? maskVars.get("vMask") : null;
        return sameDiff.nn.dotProductAttentionV2(queries,values,keys,qMask,vMask,scaleFactor,dropoutProbability,useCausalMask,training);

    }

    @Override
    public void defineParametersAndInputs(SDVertexParams params) {

    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {

    }
}
