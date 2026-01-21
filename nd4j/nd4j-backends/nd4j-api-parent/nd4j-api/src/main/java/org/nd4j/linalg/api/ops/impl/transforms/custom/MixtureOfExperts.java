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
 * Mixture of Experts (MoE) Layer
 *
 * Implements sparse MoE routing where each token is processed by top-k
 * selected experts. Used in models like DeepSeek, Mixtral, etc.
 *
 * Inputs:
 *   0: input [batch, seq_len, hidden_size] - input embeddings
 *   1: router_weights [hidden_size, num_experts] - router projection
 *   2: expert_weights [num_experts, hidden_size, expert_hidden] - expert weight matrices
 *   3: expert_bias [num_experts, expert_hidden] (optional) - expert biases
 *
 * Outputs:
 *   0: output [batch, seq_len, expert_hidden] - combined expert outputs
 *   1: router_probs [batch, seq_len, num_experts] - router probabilities
 *   2: expert_indices [batch, seq_len, top_k] - selected expert indices
 *
 * @author Eclipse Deeplearning4j
 */
@NoArgsConstructor
@Getter
public class MixtureOfExperts extends DynamicCustomOp {

    private int numExperts;
    private int topK = 2;
    private boolean normalizeProbs = true;
    private double capacityFactor = 1.0;

    /**
     * Create a Mixture of Experts operation.
     *
     * @param input Input embeddings [batch, seq_len, hidden_size]
     * @param routerWeights Router projection [hidden_size, num_experts]
     * @param expertWeights Expert weight matrices [num_experts, hidden_size, expert_hidden]
     * @param numExperts Total number of experts
     * @param topK Number of experts to route to per token
     */
    public MixtureOfExperts(@NonNull INDArray input, @NonNull INDArray routerWeights,
                           @NonNull INDArray expertWeights, int numExperts, int topK) {
        super(new INDArray[]{input, routerWeights, expertWeights}, null);
        this.numExperts = numExperts;
        this.topK = topK;
        addArgs();
    }

    /**
     * Create a Mixture of Experts operation with expert bias.
     *
     * @param input Input embeddings [batch, seq_len, hidden_size]
     * @param routerWeights Router projection [hidden_size, num_experts]
     * @param expertWeights Expert weight matrices [num_experts, hidden_size, expert_hidden]
     * @param expertBias Expert biases [num_experts, expert_hidden]
     * @param numExperts Total number of experts
     * @param topK Number of experts to route to per token
     * @param normalizeProbs Whether to normalize router probabilities
     * @param capacityFactor Expert capacity factor
     */
    public MixtureOfExperts(@NonNull INDArray input, @NonNull INDArray routerWeights,
                           @NonNull INDArray expertWeights, INDArray expertBias,
                           int numExperts, int topK, boolean normalizeProbs, double capacityFactor) {
        super(expertBias != null ?
                new INDArray[]{input, routerWeights, expertWeights, expertBias} :
                new INDArray[]{input, routerWeights, expertWeights}, null);
        this.numExperts = numExperts;
        this.topK = topK;
        this.normalizeProbs = normalizeProbs;
        this.capacityFactor = capacityFactor;
        addArgs();
    }

    /**
     * Create a Mixture of Experts for SameDiff.
     *
     * @param sd SameDiff instance
     * @param input Input embeddings [batch, seq_len, hidden_size]
     * @param routerWeights Router projection [hidden_size, num_experts]
     * @param expertWeights Expert weight matrices [num_experts, hidden_size, expert_hidden]
     * @param numExperts Total number of experts
     * @param topK Number of experts to route to per token
     */
    public MixtureOfExperts(@NonNull SameDiff sd, @NonNull SDVariable input,
                           @NonNull SDVariable routerWeights, @NonNull SDVariable expertWeights,
                           int numExperts, int topK) {
        super(null, sd, new SDVariable[]{input, routerWeights, expertWeights}, false);
        this.numExperts = numExperts;
        this.topK = topK;
        addArgs();
    }

    /**
     * Create a Mixture of Experts for SameDiff with all options.
     *
     * @param sd SameDiff instance
     * @param input Input embeddings
     * @param routerWeights Router projection
     * @param expertWeights Expert weight matrices
     * @param expertBias Expert biases (optional, can be null)
     * @param numExperts Total number of experts
     * @param topK Number of experts to route to per token
     * @param normalizeProbs Whether to normalize router probabilities
     * @param capacityFactor Expert capacity factor
     */
    public MixtureOfExperts(@NonNull SameDiff sd, @NonNull SDVariable input,
                           @NonNull SDVariable routerWeights, @NonNull SDVariable expertWeights,
                           SDVariable expertBias, int numExperts, int topK,
                           boolean normalizeProbs, double capacityFactor) {
        super(null, sd, expertBias != null ?
                new SDVariable[]{input, routerWeights, expertWeights, expertBias} :
                new SDVariable[]{input, routerWeights, expertWeights}, false);
        this.numExperts = numExperts;
        this.topK = topK;
        this.normalizeProbs = normalizeProbs;
        this.capacityFactor = capacityFactor;
        addArgs();
    }

    private void addArgs() {
        addIArgument(numExperts, topK, normalizeProbs ? 1 : 0);
        addTArgument(capacityFactor);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) {
            this.numExperts = iArguments.get(0).intValue();
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
        return "mixture_of_experts";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 3,
                "Expected at least 3 input data types for %s, got %s", getClass(), inputDataTypes);
        DataType inputType = inputDataTypes.get(0);
        // Output 0: output [batch, seq, expert_hidden] - same as input
        // Output 1: router_probs [batch, seq, num_experts] - same as input
        // Output 2: expert_indices [batch, seq, top_k] - INT64
        return Arrays.asList(inputType, inputType, DataType.INT64);
    }

    @Override
    public int getNumOutputs() {
        return 3;
    }

    @Override
    public String onnxName() {
        return "MixtureOfExperts";
    }
}
