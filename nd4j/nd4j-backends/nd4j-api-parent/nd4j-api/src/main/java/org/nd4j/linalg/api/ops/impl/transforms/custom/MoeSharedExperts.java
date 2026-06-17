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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern)
 *
 * Combines an always-on shared expert (SwiGLU) with top-K routed experts:
 * output = shared_expert(input) + weighted_sum(routed_experts(input))
 *
 * Inputs:
 *   0: input [batch, seq_len, hidden_size]
 *   1: router_weights [hidden_size, num_routed_experts]
 *   2: routed_expert_weights [num_routed_experts, hidden_size, expert_hidden]
 *   3: shared_gate_proj [hidden_size, shared_intermediate_size]
 *   4: shared_up_proj [hidden_size, shared_intermediate_size]
 *   5: shared_down_proj [shared_intermediate_size, hidden_size]
 *   6: routed_expert_bias (optional)
 *
 * Outputs:
 *   0: output [batch, seq_len, hidden_size]
 *   1: router_probs [batch, seq_len, num_routed_experts]
 *   2: expert_indices [batch, seq_len, top_k]
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class MoeSharedExperts extends DynamicCustomOp {

    private int numRoutedExperts;
    private int topK = 2;
    private boolean normalizeProbs = true;
    private double capacityFactor = 1.0;

    public MoeSharedExperts(@NonNull INDArray input, @NonNull INDArray routerWeights,
                            @NonNull INDArray routedExpertWeights, @NonNull INDArray sharedGateProj,
                            @NonNull INDArray sharedUpProj, @NonNull INDArray sharedDownProj,
                            int numRoutedExperts, int topK) {
        super(new INDArray[]{input, routerWeights, routedExpertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj}, null);
        this.numRoutedExperts = numRoutedExperts;
        this.topK = topK;
        addArgs();
    }

    public MoeSharedExperts(@NonNull INDArray input, @NonNull INDArray routerWeights,
                            @NonNull INDArray routedExpertWeights, @NonNull INDArray sharedGateProj,
                            @NonNull INDArray sharedUpProj, @NonNull INDArray sharedDownProj,
                            INDArray routedExpertBias, int numRoutedExperts, int topK,
                            boolean normalizeProbs, double capacityFactor) {
        super(routedExpertBias != null ?
                new INDArray[]{input, routerWeights, routedExpertWeights,
                        sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias} :
                new INDArray[]{input, routerWeights, routedExpertWeights,
                        sharedGateProj, sharedUpProj, sharedDownProj}, null);
        this.numRoutedExperts = numRoutedExperts;
        this.topK = topK;
        this.normalizeProbs = normalizeProbs;
        this.capacityFactor = capacityFactor;
        addArgs();
    }

    public MoeSharedExperts(@NonNull SameDiff sd, @NonNull SDVariable input,
                            @NonNull SDVariable routerWeights, @NonNull SDVariable routedExpertWeights,
                            @NonNull SDVariable sharedGateProj, @NonNull SDVariable sharedUpProj,
                            @NonNull SDVariable sharedDownProj,
                            int numRoutedExperts, int topK) {
        super(null, sd, new SDVariable[]{input, routerWeights, routedExpertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj}, false);
        this.numRoutedExperts = numRoutedExperts;
        this.topK = topK;
        addArgs();
    }

    public MoeSharedExperts(@NonNull SameDiff sd, @NonNull SDVariable input,
                            @NonNull SDVariable routerWeights, @NonNull SDVariable routedExpertWeights,
                            @NonNull SDVariable sharedGateProj, @NonNull SDVariable sharedUpProj,
                            @NonNull SDVariable sharedDownProj, SDVariable routedExpertBias,
                            int numRoutedExperts, int topK, boolean normalizeProbs,
                            double capacityFactor) {
        super(null, sd, routedExpertBias != null ?
                new SDVariable[]{input, routerWeights, routedExpertWeights,
                        sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias} :
                new SDVariable[]{input, routerWeights, routedExpertWeights,
                        sharedGateProj, sharedUpProj, sharedDownProj}, false);
        this.numRoutedExperts = numRoutedExperts;
        this.topK = topK;
        this.normalizeProbs = normalizeProbs;
        this.capacityFactor = capacityFactor;
        addArgs();
    }

    private void addArgs() {
        addIArgument(numRoutedExperts, topK, normalizeProbs ? 1 : 0);
        addTArgument(capacityFactor);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.numRoutedExperts = iArguments.get(0).intValue();
        }
        if (iArguments.size() > 1) {
            this.topK = iArguments.get(1).intValue();
        }
        if (iArguments.size() > 2) {
            this.normalizeProbs = iArguments.get(2) != 0;
        }
        if (tArguments.size() > 0) {
            this.capacityFactor = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "moe_shared_experts";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 6,
                "Expected at least 6 input data types for %s, got %s", getClass(), inputDataTypes);
        DataType inputType = inputDataTypes.get(0);
        return Arrays.asList(inputType, inputType, DataType.INT64);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }

    @Override
    public String onnxName() {
        return "MoeSharedExperts";
    }
}
