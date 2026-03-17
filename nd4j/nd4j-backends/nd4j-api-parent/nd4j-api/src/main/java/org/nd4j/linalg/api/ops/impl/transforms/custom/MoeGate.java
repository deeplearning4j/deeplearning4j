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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Mixture-of-Experts (MoE) gating operation.
 * <p>
 * Computes top-K expert routing with load-balancing auxiliary loss:
 * <pre>
 *   logits         = hiddenStates @ gateWeight
 *   expertIndices  = topK(softmax(logits))
 *   gateWeights    = normalized weights for selected experts
 *   auxLoss        = load balancing loss
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: hiddenStates [T, H] - token hidden states</li>
 *   <li>1: gateWeight [H, E] - gating projection (H=hidden, E=numExperts)</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: topK (default 2) - number of experts per token</li>
 *   <li>1: numExperts (default 8) - total number of experts</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: auxLossCoeff (default 0.01) - load balancing loss coefficient</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: expertIndices [T, K] INT64 - selected expert indices</li>
 *   <li>1: gateWeights [T, K] - routing weights for selected experts</li>
 *   <li>2: auxLoss [1] - auxiliary load balancing loss scalar</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class MoeGate extends DynamicCustomOp {

    @Getter private int topK = 2;
    @Getter private int numExperts = 8;
    @Getter private float auxLossCoeff = 0.01f;

    /**
     * SameDiff constructor with default options.
     */
    public MoeGate(SameDiff sameDiff, SDVariable hiddenStates, SDVariable gateWeight) {
        super(null, sameDiff, new SDVariable[]{hiddenStates, gateWeight}, false);
        addIArgument((long) topK, (long) numExperts);
        addTArgument((double) auxLossCoeff);
    }

    /**
     * SameDiff constructor with double auxLossCoeff.
     */
    public MoeGate(SameDiff sameDiff, SDVariable hiddenStates, SDVariable gateWeight,
                   int topK, int numExperts, double auxLossCoeff) {
        this(sameDiff, hiddenStates, gateWeight, topK, numExperts, (float) auxLossCoeff);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public MoeGate(SameDiff sameDiff, SDVariable hiddenStates, SDVariable gateWeight,
                   int topK, int numExperts, float auxLossCoeff) {
        super(null, sameDiff, new SDVariable[]{hiddenStates, gateWeight}, false);
        this.topK = topK;
        this.numExperts = numExperts;
        this.auxLossCoeff = auxLossCoeff;
        addIArgument((long) topK, (long) numExperts);
        addTArgument((double) auxLossCoeff);
    }

    /**
     * INDArray constructor.
     */
    public MoeGate(INDArray hiddenStates, INDArray gateWeight,
                   INDArray expertIndices, INDArray gateWeights, INDArray auxLoss,
                   int topK, int numExperts, float auxLossCoeff) {
        super(null, new INDArray[]{hiddenStates, gateWeight},
                expertIndices != null ? new INDArray[]{expertIndices, gateWeights, auxLoss} : null);
        this.topK = topK;
        this.numExperts = numExperts;
        this.auxLossCoeff = auxLossCoeff;
        addIArgument((long) topK, (long) numExperts);
        addTArgument((double) auxLossCoeff);
    }

    public MoeGate(SameDiff sd, SDVariable input, SDVariable gateWeights, int numExperts, int topK) {
        super(null, sd, new SDVariable[]{input, gateWeights}, false);
        this.topK = topK;
        this.numExperts = numExperts;
        addIArgument((long) topK, (long) numExperts);
        addTArgument((double) this.auxLossCoeff);
    }

    public MoeGate(INDArray input, INDArray gateWeights, int numExperts, int topK) {
        super(new INDArray[]{input, gateWeights}, null);
        this.topK = topK;
        this.numExperts = numExperts;
        addIArgument((long) topK, (long) numExperts);
        addTArgument((double) this.auxLossCoeff);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.topK = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.numExperts = iArguments.get(1).intValue();
        if (tArguments.size() > 0) this.auxLossCoeff = tArguments.get(0).floatValue();
    }

    @Override
    public String opName() {
        return "moe_gate";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(DataType.INT64, dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }
}
