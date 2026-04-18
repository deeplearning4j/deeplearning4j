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

package org.nd4j.autodiff.samediff.peft;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.config.LoftQConfig;
import org.nd4j.autodiff.samediff.config.LoraConfig;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 * LoRA (Low-Rank Adaptation) layer implementation.
 * <p>
 * Implements the low-rank decomposition: W = W₀ + BA
 * where B ∈ R^{d×r} and A ∈ R^{r×k}.
 * <p>
 * The original weight W₀ is frozen, and only the low-rank matrices A and B
 * are trained. The output is scaled by α/r (or α/√r for rsLoRA).
 *
 * <p>Implementation details:</p>
 * <pre>
 * # During forward pass:
 * base_output = x @ W₀
 * lora_output = (x @ A.T @ B.T) * scaling
 * output = base_output + dropout(lora_output)
 *
 * where scaling = α/r (or α/√r for rsLoRA)
 * </pre>
 *
 * @author Adam Gibson
 * @see LoraConfig
 */
@Data
@Builder
@Slf4j
public class LoraLayer {

    /**
     * Name prefix for this LoRA layer.
     */
    private final String namePrefix;

    /**
     * The LoRA configuration.
     */
    private final LoraConfig config;

    /**
     * Input dimension of the original weight matrix.
     */
    private final int inFeatures;

    /**
     * Output dimension of the original weight matrix.
     */
    private final int outFeatures;

    /**
     * The down-projection matrix A ∈ R^{r×k}.
     * Initialized with Kaiming/Xavier initialization.
     */
    private SDVariable loraA;

    /**
     * The up-projection matrix B ∈ R^{d×r}.
     * Initialized to zeros so initial output is unchanged.
     */
    private SDVariable loraB;

    /**
     * Create the LoRA variables in the SameDiff graph using standard initialization.
     *
     * @param sd The SameDiff instance
     */
    public void createVariables(SameDiff sd) {
        createVariables(sd, null);
    }

    /**
     * Create the LoRA variables in the SameDiff graph.
     * <p>
     * When {@code pretrainedWeight} is non-null and the config is a {@link LoftQConfig},
     * both A and B are initialized using the LoftQ algorithm (SVD of the quantization
     * residual), providing a better starting point than random initialization.
     * For all other initialization methods, {@code pretrainedWeight} is ignored.
     *
     * @param sd               The SameDiff instance
     * @param pretrainedWeight The original pretrained weight matrix [outFeatures, inFeatures],
     *                         required for LoftQ initialization; may be null otherwise
     */
    public void createVariables(SameDiff sd, INDArray pretrainedWeight) {
        int r = config.getR();
        DataType dtype = DataType.FLOAT;

        String initMethod = config.getInitLoraWeights();

        if ("loftq".equalsIgnoreCase(initMethod) && config instanceof LoftQConfig && pretrainedWeight != null) {
            // LoftQ initialization: derive both A and B from SVD of quantization residual
            LoftQConfig loftQConfig = (LoftQConfig) config;
            Pair<INDArray, INDArray> abPair = LoftQInitializer.initialize(pretrainedWeight, r, loftQConfig);
            INDArray bInit = abPair.getFirst();   // [outFeatures, r]
            INDArray aInit = abPair.getSecond();  // [r, inFeatures]

            loraA = sd.var(namePrefix + "_lora_A", aInit);
            loraB = sd.var(namePrefix + "_lora_B", bInit);

            log.info("LoftQ-initialized LoRA layer '{}': A[{},{}], B[{},{}], rank={}, iters={}",
                namePrefix, r, inFeatures, outFeatures, r, r, loftQConfig.getNumIterations());
        } else {
            // Standard initialization
            if ("loftq".equalsIgnoreCase(initMethod) && pretrainedWeight == null) {
                log.warn("LoftQ requested for '{}' but no pretrained weight provided; " +
                    "falling back to kaiming_uniform initialization", namePrefix);
            }

            // A matrix: [r, inFeatures] - initialized with configured method
            INDArray aInit = initializeA(r, inFeatures, dtype);
            loraA = sd.var(namePrefix + "_lora_A", aInit);

            // B matrix: [outFeatures, r] - initialized to zeros
            INDArray bInit = Nd4j.zeros(dtype, outFeatures, r);
            loraB = sd.var(namePrefix + "_lora_B", bInit);

            log.debug("Created LoRA layer '{}': A[{},{}], B[{},{}], rank={}, scaling={}",
                namePrefix, r, inFeatures, outFeatures, r, r, config.getScaling());
        }
    }

    /**
     * Initialize matrix A using the configured initialization method.
     */
    private INDArray initializeA(int r, int inFeatures, DataType dtype) {
        String initMethod = config.getInitLoraWeights();

        switch (initMethod.toLowerCase()) {
            case "zeros":
                return Nd4j.zeros(dtype, r, inFeatures);

            case "gaussian":
                return Nd4j.randn(dtype, r, inFeatures).muli(config.getInitStd());

            case "kaiming_uniform":
            case "loftq":  // LoftQ fallback (no pretrained weight) uses kaiming_uniform
            default:
                // Kaiming uniform initialization
                double bound = Math.sqrt(6.0 / inFeatures);
                return Nd4j.rand(dtype, r, inFeatures).subi(0.5).muli(2 * bound);
        }
    }

    /**
     * Apply the LoRA transformation to an input.
     *
     * @param sd    The SameDiff instance
     * @param input The input tensor
     * @return The LoRA output (to be added to the base output)
     */
    public SDVariable apply(SameDiff sd, SDVariable input) {
        // x @ A.T = x @ A^T (input: [batch, inFeatures], A: [r, inFeatures])
        // Result: [batch, r]
        SDVariable afterA = sd.mmul(input, sd.transpose(loraA));

        // (x @ A.T) @ B.T (B: [outFeatures, r], B.T: [r, outFeatures])
        // Result: [batch, outFeatures]
        SDVariable afterB = sd.mmul(afterA, sd.transpose(loraB));

        // Apply scaling
        double scaling = config.getScaling();
        SDVariable scaled = sd.math.mul(afterB, scaling);

        // Apply dropout if configured
        double dropout = config.getLoraDropout();
        if (dropout > 0) {
            // During training, apply dropout
            // dropout(input, inverted, probabilityValue) - inverted=false means standard dropout
            scaled = sd.nn.dropout(scaled, false, dropout);
        }

        return scaled;
    }

    /**
     * Get the merged weight update (B @ A) scaled appropriately.
     * This can be added to the original weights to merge LoRA.
     *
     * @return The weight update matrix
     */
    public INDArray getMergedWeightUpdate() {
        if (loraA == null || loraB == null) {
            throw new IllegalStateException("LoRA variables not initialized");
        }

        INDArray a = loraA.getArr();
        INDArray b = loraB.getArr();

        // Compute B @ A and scale
        // B: [outFeatures, r], A: [r, inFeatures]
        // Result: [outFeatures, inFeatures]
        INDArray merged = b.mmul(a);
        merged.muli(config.getScaling());

        return merged;
    }

    /**
     * Get the number of trainable parameters in this LoRA layer.
     */
    public long getTrainableParameters() {
        int r = config.getR();
        return (long) r * inFeatures + (long) outFeatures * r;
    }

    /**
     * Create a LoRA layer for a given weight matrix using standard initialization.
     *
     * @param sd          SameDiff instance
     * @param namePrefix  Name prefix for the LoRA variables
     * @param config      LoRA configuration
     * @param inFeatures  Input dimension
     * @param outFeatures Output dimension
     * @return The created LoRA layer
     */
    public static LoraLayer create(SameDiff sd, String namePrefix, LoraConfig config,
                                   int inFeatures, int outFeatures) {
        return create(sd, namePrefix, config, inFeatures, outFeatures, null);
    }

    /**
     * Create a LoRA layer for a given weight matrix, optionally using LoftQ initialization.
     *
     * @param sd               SameDiff instance
     * @param namePrefix       Name prefix for the LoRA variables
     * @param config           LoRA configuration (use {@link LoftQConfig} to enable LoftQ init)
     * @param inFeatures       Input dimension
     * @param outFeatures      Output dimension
     * @param pretrainedWeight The original weight matrix [outFeatures, inFeatures] for LoftQ;
     *                         pass null to use standard initialization
     * @return The created LoRA layer
     */
    public static LoraLayer create(SameDiff sd, String namePrefix, LoraConfig config,
                                   int inFeatures, int outFeatures, INDArray pretrainedWeight) {
        LoraLayer layer = LoraLayer.builder()
            .namePrefix(namePrefix)
            .config(config)
            .inFeatures(inFeatures)
            .outFeatures(outFeatures)
            .build();

        layer.createVariables(sd, pretrainedWeight);
        return layer;
    }
}
