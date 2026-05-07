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
 * Apply sampling penalties and min-p filtering to logits in-place.
 *
 * This op modifies logits before the sampling step to control repetition
 * and token diversity. Supports three penalty types:
 *
 * <ul>
 *   <li><b>Repetition penalty</b>: Multiplicative, direction-aware.
 *       logit = logit > 0 ? logit / repPenalty : logit * repPenalty.
 *       Value of 1.0 = off, >1.0 penalizes repetition.</li>
 *   <li><b>Frequency penalty</b>: Subtracts count(token) * freqPenalty.
 *       Proportional to how often a token has appeared.</li>
 *   <li><b>Presence penalty</b>: Subtracts presPenalty for any token that
 *       appeared at least once (regardless of count).</li>
 *   <li><b>Min-P filtering</b>: Adaptive threshold — masks tokens whose
 *       softmax probability is less than minP * max_probability.
 *       Keeps more tokens when the model is uncertain, fewer when confident.</li>
 * </ul>
 *
 * Inputs:
 *   0: logits    [batch, vocabSize] or [vocabSize] — modified in-place
 *   1: inputIds  [batch, seqLen] or [seqLen] INT64 — prior token IDs
 *
 * Output:
 *   0: modified logits (same shape as input 0)
 *
 * Float args:
 *   0: repetitionPenalty (default: 1.0 = off)
 *   1: frequencyPenalty  (default: 0.0 = off)
 *   2: presencePenalty   (default: 0.0 = off)
 *   3: minP              (default: 0.0 = off, typical: 0.05-0.1)
 */
@NoArgsConstructor
public class SamplingPenalties extends DynamicCustomOp {

    @Getter private double repetitionPenalty = 1.0;
    @Getter private double frequencyPenalty = 0.0;
    @Getter private double presencePenalty = 0.0;
    @Getter private double minP = 0.0;

    public SamplingPenalties(INDArray logits, INDArray inputIds,
                              double repetitionPenalty, double frequencyPenalty,
                              double presencePenalty, double minP) {
        super(new INDArray[]{logits, inputIds}, null);
        this.repetitionPenalty = repetitionPenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        this.minP = minP;
        addTArgument(repetitionPenalty, frequencyPenalty, presencePenalty, minP);
    }

    public SamplingPenalties(SameDiff sameDiff, SDVariable logits, SDVariable inputIds,
                              double repetitionPenalty, double frequencyPenalty,
                              double presencePenalty, double minP) {
        super(null, sameDiff, new SDVariable[]{logits, inputIds}, false);
        this.repetitionPenalty = repetitionPenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        this.minP = minP;
        addTArgument(repetitionPenalty, frequencyPenalty, presencePenalty, minP);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.repetitionPenalty = tArguments.get(0);
        if (tArguments.size() > 1) this.frequencyPenalty = tArguments.get(1);
        if (tArguments.size() > 2) this.presencePenalty = tArguments.get(2);
        if (tArguments.size() > 3) this.minP = tArguments.get(3);
    }

    @Override
    public String opName() {
        return "sampling_penalties";
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
