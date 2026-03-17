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
import java.util.Collections;
import java.util.List;

/**
 * GPU-accelerated Top-K sampling for LLM token generation.
 * <p>
 * Performs top-K filtering, softmax, and multinomial sampling entirely on the GPU,
 * eliminating the device-to-host transfer overhead for logits. Uses CUB radix sort
 * for efficient top-K selection.
 * <p>
 * Pipeline: logits -> temperature scaling -> top-K filter -> softmax -> multinomial sample
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: logits [batch, vocab_size]</li>
 *   <li>1: random values [batch] (uniform [0,1) for multinomial, optional)</li>
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
 *   <li>0: k (number of top tokens to consider, default: 50)</li>
 *   <li>1: seed (RNG seed, used if no random values input, default: 0)</li>
 * </ul>
 * <p>
 * Float arguments:
 * <ul>
 *   <li>0: temperature (default: 1.0)</li>
 * </ul>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see GpuTopPSample
 * @see TokenSample
 */
@NoArgsConstructor
public class GpuTopKSample extends DynamicCustomOp {

    @Getter private int k = 50;
    @Getter private long seed = 0;
    @Getter private double temperature = 1.0;

    /**
     * INDArray constructor with logits only (default k=50, temperature=1.0).
     */
    public GpuTopKSample(INDArray logits, int k) {
        this(logits, k, 1.0, 0);
    }

    /**
     * Full INDArray constructor.
     *
     * @param logits      input logits [batch, vocab_size]
     * @param k           number of top tokens to consider
     * @param temperature temperature for scaling
     * @param seed        RNG seed (0 for random)
     */
    public GpuTopKSample(INDArray logits, int k, double temperature, long seed) {
        super(new INDArray[]{logits}, null);
        this.k = k;
        this.temperature = temperature;
        this.seed = seed;
        addIArgument((long) k, seed);
        addTArgument(temperature);
    }

    /**
     * INDArray constructor with explicit random values.
     */
    public GpuTopKSample(INDArray logits, INDArray randomValues, int k, double temperature) {
        super(new INDArray[]{logits, randomValues}, null);
        this.k = k;
        this.temperature = temperature;
        addIArgument((long) k, seed);
        addTArgument(temperature);
    }

    /**
     * SameDiff constructor with int seed and reordered parameters (k, seed, temperature).
     */
    public GpuTopKSample(SameDiff sameDiff, SDVariable logits, int k, int seed, double temperature) {
        this(sameDiff, logits, k, temperature, (long) seed);
    }

    /**
     * SameDiff constructor.
     */
    public GpuTopKSample(SameDiff sameDiff, SDVariable logits, int k, double temperature, long seed) {
        super(null, sameDiff, new SDVariable[]{logits}, false);
        this.k = k;
        this.temperature = temperature;
        this.seed = seed;
        addIArgument((long) k, seed);
        addTArgument(temperature);
    }

    /**
     * SameDiff constructor with random values input.
     */
    public GpuTopKSample(SameDiff sameDiff, SDVariable logits, SDVariable randomValues,
                         int k, double temperature) {
        super(null, sameDiff, new SDVariable[]{logits, randomValues}, false);
        this.k = k;
        this.temperature = temperature;
        addIArgument((long) k, seed);
        addTArgument(temperature);
    }

    public GpuTopKSample(SameDiff sd, SDVariable logits, int k, double temperature) {
        this(sd, logits, k, temperature, 0L);
    }

    public GpuTopKSample(INDArray logits, int k, double temperature) {
        this(logits, k, temperature, 0L);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.k = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.seed = iArguments.get(1);
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "gpu_top_k_sample";
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
