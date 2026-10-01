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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Nucleus (top-p) token sampling for LLM token generation.
 * <p>
 * Draws one token per row from the smallest set of the most likely tokens that holds p of the softmax of the logits
 * scaled by 1 / temperature, and reports the probability of each drawn token under the kept, renormalized
 * distribution. Tokens of equal weight are kept or dropped together. With temperature &lt;= 0 and p &lt;= 0 the
 * selection is greedy: the argmax of each row, lowest index on ties, with probability 1. Otherwise a temperature
 * &lt;= 0 leaves the logits unscaled.
 * <p>
 * Given the token history, the penalties first rewrite the logits of the sampled position (the logits input itself
 * stays unchanged):
 * <ul>
 *   <li>repetition_penalty: the positive logit of a seen token divides by it, a negative one multiplies by it</li>
 *   <li>frequency_penalty: subtracted from the logit of a token once per occurrence in the history</li>
 *   <li>presence_penalty: subtracted once from the logit of every token in the history</li>
 * </ul>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: logits (floating) [vocab], [batch, vocab] or [batch, seqLen, vocab] (the last position is sampled)</li>
 *   <li>1: uniforms (optional, floating) [batch] in [0, 1), the draw of each row; an empty array stands for absent
 *   uniforms</li>
 *   <li>2: token history (optional, integer ids) [seqLen], shared by every row, or [batch, seqLen]; an empty array
 *   stands for an absent history</li>
 * </ul>
 * <p>
 * Outputs:
 * <ul>
 *   <li>0: sampled token IDs (INT64) [batch], a scalar for rank-1 logits</li>
 *   <li>1: probabilities of the sampled tokens [batch], a scalar for rank-1 logits, in the type of the logits</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: seed (without uniforms, a positive seed makes the draws reproducible, otherwise they take fresh entropy;
 *   default: 0)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: p (the probability mass to keep; p &lt;= 0 or p &gt;= 1 keeps every token; default: 0.9)</li>
 *   <li>1: temperature (default: 1.0)</li>
 *   <li>2: repetition_penalty (default: 1.0, no penalty)</li>
 *   <li>3: frequency_penalty (default: 0.0)</li>
 *   <li>4: presence_penalty (default: 0.0)</li>
 * </ul>
 *
 * Adam Gibson
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
     * @param seed        RNG seed (a positive seed is reproducible, otherwise the draws take fresh entropy)
     */
    public GpuTopPSample(INDArray logits, double p, double temperature, long seed) {
        this(logits, null, null, p, temperature, seed, 1.0, 0.0, 0.0);
    }

    /**
     * INDArray constructor with all penalty parameters. The penalties take effect only with a token history, which
     * the full constructor takes.
     */
    public GpuTopPSample(INDArray logits, double p, double temperature, long seed,
                         double repetitionPenalty, double frequencyPenalty, double presencePenalty) {
        this(logits, null, null, p, temperature, seed, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * INDArray constructor with explicit random values (null draws from fresh entropy).
     */
    public GpuTopPSample(INDArray logits, INDArray randomValues, double p, double temperature) {
        this(logits, randomValues, null, p, temperature, 0L, 1.0, 0.0, 0.0);
    }

    /**
     * Full INDArray constructor.
     *
     * @param logits            input logits [vocab], [batch, vocab] or [batch, seqLen, vocab]
     * @param randomValues      uniforms in [0, 1), one per row, or null to draw from the seed
     * @param history           token ids seen so far, [seqLen] or [batch, seqLen], or null for no penalties
     * @param p                 nucleus probability threshold
     * @param temperature       temperature for scaling
     * @param seed              RNG seed used without random values
     * @param repetitionPenalty multiplicative penalty of seen tokens (1 = none)
     * @param frequencyPenalty  penalty per occurrence of a seen token
     * @param presencePenalty   penalty of any seen token
     */
    public GpuTopPSample(INDArray logits, INDArray randomValues, INDArray history, double p, double temperature,
                         long seed, double repetitionPenalty, double frequencyPenalty, double presencePenalty) {
        super(inputs(logits, randomValues, history), null);
        configure(p, temperature, seed, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    /**
     * SameDiff constructor.
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, double p, double temperature, long seed) {
        this(sameDiff, logits, null, null, p, temperature, seed, 1.0, 0.0, 0.0);
    }

    /**
     * SameDiff constructor with random values input (null draws from fresh entropy).
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, SDVariable randomValues,
                         double p, double temperature) {
        this(sameDiff, logits, randomValues, null, p, temperature, 0L, 1.0, 0.0, 0.0);
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
     * SameDiff constructor with all penalty parameters. The penalties take effect only with a token history, which
     * the full constructor takes.
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, double p, double temperature,
                         long seed, double repetitionPenalty, double frequencyPenalty,
                         double presencePenalty) {
        this(sameDiff, logits, null, null, p, temperature, seed, repetitionPenalty, frequencyPenalty,
                presencePenalty);
    }

    /**
     * Full SameDiff constructor.
     *
     * @param logits            input logits [vocab], [batch, vocab] or [batch, seqLen, vocab]
     * @param randomValues      uniforms in [0, 1), one per row, or null to draw from the seed
     * @param history           token ids seen so far, [seqLen] or [batch, seqLen], or null for no penalties
     * @param p                 nucleus probability threshold
     * @param temperature       temperature for scaling
     * @param seed              RNG seed used without random values
     * @param repetitionPenalty multiplicative penalty of seen tokens (1 = none)
     * @param frequencyPenalty  penalty per occurrence of a seen token
     * @param presencePenalty   penalty of any seen token
     */
    public GpuTopPSample(SameDiff sameDiff, SDVariable logits, SDVariable randomValues, SDVariable history,
                         double p, double temperature, long seed, double repetitionPenalty,
                         double frequencyPenalty, double presencePenalty) {
        super(null, sameDiff, inputs(sameDiff, logits, randomValues, history), false);
        configure(p, temperature, seed, repetitionPenalty, frequencyPenalty, presencePenalty);
    }

    public GpuTopPSample(SameDiff sd, SDVariable logits, double p, double temperature) {
        this(sd, logits, p, temperature, 0L);
    }

    public GpuTopPSample(INDArray logits, double p, double temperature) {
        this(logits, p, temperature, 0L);
    }

    // Absent uniforms ahead of a history take an empty placeholder, which the op reads as absent.
    private static INDArray[] inputs(INDArray logits, INDArray randomValues, INDArray history) {
        if (history == null) {
            return randomValues == null ? new INDArray[]{logits} : new INDArray[]{logits, randomValues};
        }
        return new INDArray[]{logits, randomValues == null ? Nd4j.empty(DataType.FLOAT) : randomValues, history};
    }

    private static SDVariable[] inputs(SameDiff sameDiff, SDVariable logits, SDVariable randomValues,
                                       SDVariable history) {
        if (history == null) {
            return randomValues == null ? new SDVariable[]{logits} : new SDVariable[]{logits, randomValues};
        }
        return new SDVariable[]{logits,
                randomValues == null ? sameDiff.constant(Nd4j.empty(DataType.FLOAT)) : randomValues, history};
    }

    private void configure(double p, double temperature, long seed, double repetitionPenalty,
                           double frequencyPenalty, double presencePenalty) {
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
        Preconditions.checkState(inputDataTypes != null && !inputDataTypes.isEmpty(),
                "Expected the logits data type for gpu_top_p_sample, got %s", inputDataTypes);
        // Output 0: token IDs (INT64), Output 1: probabilities in the type of the logits
        return Arrays.asList(DataType.INT64, inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}
