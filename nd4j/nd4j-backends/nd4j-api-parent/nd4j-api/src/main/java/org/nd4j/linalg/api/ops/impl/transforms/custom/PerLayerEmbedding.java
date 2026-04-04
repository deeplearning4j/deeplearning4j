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

import java.util.Collections;
import java.util.List;

/**
 * Per-Layer Embedding (PLE) for Gemma 4.
 *
 * Applies a per-layer embedding correction to hidden states. Each transformer
 * layer has its own embedding table that adjusts the token representations
 * based on the input token IDs.
 *
 * <p>Computation: output = hiddenStates + scale * pleWeight[tokenIds]</p>
 *
 * Inputs:
 *   0: hiddenStates [batch, seq_len, hidden_size] - current hidden states
 *   1: pleWeight    [vocab_size, hidden_size]      - per-layer embedding table
 *   2: tokenIds     [batch, seq_len]               - input token indices
 *
 * Outputs:
 *   0: output [batch, seq_len, hidden_size] - adjusted hidden states
 *
 * tArgs:
 *   0: scale (optional, default 1.0) - scaling factor for the embedding correction
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class PerLayerEmbedding extends DynamicCustomOp {

    @Getter private double scale = 1.0;

    public PerLayerEmbedding(INDArray hiddenStates, INDArray pleWeight, INDArray tokenIds) {
        this(hiddenStates, pleWeight, tokenIds, 1.0);
    }

    public PerLayerEmbedding(INDArray hiddenStates, INDArray pleWeight, INDArray tokenIds, double scale) {
        super(new INDArray[]{hiddenStates, pleWeight, tokenIds}, null);
        this.scale = scale;
        addTArgument(scale);
    }

    public PerLayerEmbedding(SameDiff sd, SDVariable hiddenStates, SDVariable pleWeight, SDVariable tokenIds) {
        this(sd, hiddenStates, pleWeight, tokenIds, 1.0);
    }

    public PerLayerEmbedding(SameDiff sd, SDVariable hiddenStates, SDVariable pleWeight, SDVariable tokenIds,
                             double scale) {
        super(null, sd, new SDVariable[]{hiddenStates, pleWeight, tokenIds}, false);
        this.scale = scale;
        addTArgument(scale);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) {
            this.scale = tArguments.get(0);
        }
    }

    @Override
    public String opName() {
        return "per_layer_embedding";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
