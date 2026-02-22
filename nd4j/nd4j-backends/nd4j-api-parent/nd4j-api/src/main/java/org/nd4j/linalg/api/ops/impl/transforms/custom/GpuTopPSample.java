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
 * GPU-accelerated Top-P (nucleus) sampling for LLM token generation.
 * <p>
 * Performs nucleus sampling entirely on the GPU: sorts tokens by probability,
 * computes cumulative sums to find the smallest set of tokens whose cumulative
 * probability exceeds p, then samples from that nucleus.
 * <p>
 * Pipeline: logits -> temperature scaling -> softmax -> sort -> cumsum -> nucleus filter -> sample
 * <p>
 * Supports additional penalty parameters for controlling repetition:
 * <ul>
 *   <li>repetition_penalty: multiplicative penalty for previously generated tokens</li>
 *   <li>frequency_penalty: additive penalty proportional to token frequency</li>
 *   <li>presence_penalty: flat additive penalty for any previously seen token</li>
 * </ul>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: logits [batch, vocab_size]</li>
 *   <li>1: random values [batch] (uniform [0,1), optional)</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: sampled token IDs [batch] (INT64)</li>
 *   <li>1: sampled probabilities [batch] (FLOAT, optional)</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: seed (RNG seed, default: 0)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: p (nucleus probability threshold, default: 0.9)</li>
 *   <li>1: temperature (default: 1.0)</li>
 *   <li>2: repetition_penalty (default: 1.0, no penalty)</li>
 *   <li>3: frequency_penalty (default: 0.0)</li>
 *   <li>4: presence_penalty (default: 0.0)</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see GpuTopKSample
 * @see TokenSample
 */
@NoArgsConstructor
public class GpuTopPSample extends DynamicCustomOp {

    @Getter private double p = 0.9;
    @Getter private double temperature = 1.0;
    @Getter private double repetitionPenalty = 1.0;
    @Getter private double frequencyPenalty = 0.0;
    @Getter private double presencePenalty = 0.0;
    @Getter private long seed = 0;

    /**
     * INDArray constructor with logits only (default p=0.9, temperature=1.0).
     */
    public GpuTopPSample(INDArray logits, double p) {
        this(logits, p, 1.0, 0);
    }

    /**
     * INDArray constructor with core parameters.
     *
     * @param logits      input logits [batch, vocab_size]
     * @param p           nucleus probability threshold (0.0 to 1.0)
     * @param temperature temperature for scaling
     * @param seed        RNG seed (0 for random)
     */
    public GpuTopPSample(INDArray logits, double p, double temperature, long seed) {
        super(new INDArray[]{logits}, null);
        this.p = p;
        this.temperature = temperature;
        this.seed = seed;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * Full INDArray constructor with all penalty parameters.
     */
    public GpuTopPSample(INDArray logits, double p, double temperature, long seed,
                         double repetitionPenalty, double frequencyPenalty, double presencePenalty) {
        super(new INDArray[]{logits}, null);
        this.p = p;
        this.temperature = temperature;
        this.seed = seed;
        this.repetitionPenalty = repetitionPenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * INDArray constructor with explicit random values.
     */
    public GpuTopPSample(INDArray logits, INDArray randomValues, double p, double temperature) {
        super(new INDArray[]{logits, randomValues}, null);
        this.p = p;
        this.temperature = temperature;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * SameDiff constructor.
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, double p, double temperature, long seed) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.p = p;
        this.temperature = temperature;
        this.seed = seed;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * SameDiff constructor with random values input.
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, SDVariable randomValues,
                         double p, double temperature) {
        super(null, sameDiff, new SDVariable[]{logits, randomValues}, false);
        this.p = p;
        this.temperature = temperature;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * SameDiff constructor with int seed and reordered parameters (seed, p, temperature, ...).
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, int seed, double p,
                         double temperature, double repetitionPenalty, double frequencyPenalty,
                         double presencePenalty) {
        this(sameDiff, logits, p, temperature, (long) seed, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * Full SameDiff constructor with all parameters.
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, double p, double temperature,
                         long seed, double repetitionPenalty, double frequencyPenalty,
                         double presencePenalty) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.p = p;
        this.temperature = temperature;
        this.seed = seed;
        this.repetitionPenalty = repetitionPenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.presencePenalty = presencePenalty;
        addIArgument(seed);
        addTArgument(p, temperature, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.seed = iArguments.get(0);
        if (tArguments.size() > 0) this.p = tArguments.get(0);
        if (tArguments.size() > 1) this.temperature = tArguments.get(1);
        if (tArguments.size() > 2) this.repetitionPenalty = tArguments.get(2);
        if (tArguments.size() > 3) this.frequencyPenalty = tArguments.get(3);
        if (tArguments.size() > 4) this.presencePenalty = tArguments.get(4);
    }

    @Override
    public String opName() {
        return "gpu_top_p_sample";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // Output 0: token IDs (INT64), Output 1: probabilities (FLOAT)
        return Arrays.asList(DataType.INT64, DataType.FLOAT);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
