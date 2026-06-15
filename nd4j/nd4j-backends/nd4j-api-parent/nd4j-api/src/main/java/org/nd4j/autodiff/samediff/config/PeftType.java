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

package org.nd4j.autodiff.samediff.config;

/**
 * Types of Parameter-Efficient Fine-Tuning (PEFT) methods.
 * <p>
 * Based on the Hugging Face PEFT library, these methods allow fine-tuning
 * large models with minimal parameter updates while maintaining performance
 * comparable to full fine-tuning.
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see <a href="https://github.com/huggingface/peft">Hugging Face PEFT</a>
 */
public enum PeftType {

    /**
     * Low-Rank Adaptation (LoRA).
     * Decomposes weight updates into low-rank matrices: W = W0 + BA
     * where B ∈ R^{d×r} and A ∈ R^{r×k} with r << min(d, k).
     * <p>
     * Key parameters: rank (r), alpha, target_modules, dropout
     */
    LORA,

    /**
     * Prefix Tuning.
     * Prepends trainable prefix vectors to all transformer layers.
     * The prefix parameters are optimized through a separate feed-forward network.
     * <p>
     * Key parameters: num_virtual_tokens, prefix_projection
     */
    PREFIX_TUNING,

    /**
     * Prompt Tuning.
     * Adds trainable soft prompt tokens to the input embeddings only.
     * Simpler than prefix tuning - only modifies the input layer.
     * <p>
     * Key parameters: num_virtual_tokens, prompt_tuning_init
     */
    PROMPT_TUNING,

    /**
     * P-Tuning.
     * Uses a prompt encoder (LSTM) to generate trainable prompt embeddings.
     * Prompt tokens can be inserted anywhere in the input sequence.
     * <p>
     * Key parameters: num_virtual_tokens, encoder_hidden_size
     */
    P_TUNING,

    /**
     * IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
     * Learns vectors that rescale activation values.
     * Extremely parameter-efficient with fewer trainable parameters than LoRA.
     * <p>
     * Key parameters: target_modules, feedforward_modules
     */
    IA3,

    /**
     * Adapter layers.
     * Inserts small bottleneck layers between transformer layers.
     * <p>
     * Key parameters: adapter_size, adapter_activation
     */
    ADAPTERS,

    /**
     * BitFit - Bias-only fine-tuning.
     * Only updates bias terms in the model.
     * Very parameter-efficient but limited expressiveness.
     */
    BITFIT,

    /**
     * QLoRA - Quantized LoRA.
     * Combines 4-bit quantization with LoRA for memory efficiency.
     * <p>
     * Key parameters: same as LORA plus quantization settings
     */
    QLORA,

    /**
     * DoRA - Weight-Decomposed Low-Rank Adaptation.
     * Decomposes updates into magnitude and direction components.
     * <p>
     * Key parameters: same as LORA plus magnitude initialization
     */
    DORA,

    /**
     * AdaLoRA - Adaptive Low-Rank Adaptation.
     * Adaptively allocates rank based on importance scoring.
     * <p>
     * Key parameters: initRank, targetRank, pruning schedule
     */
    ADALORA,

    /**
     * LoHa - Low-Rank Hadamard Product.
     * Uses Hadamard products of two low-rank decompositions.
     * Effective rank is dim², so dim=8 → effective rank up to 64.
     * <p>
     * Key parameters: dim, alpha
     */
    LOHA,

    /**
     * LoKr - Low-Rank Kronecker Product.
     * Uses Kronecker products for weight updates.
     * <p>
     * Key parameters: dim, factor, alpha
     */
    LOKR,

    /**
     * VeRA - Vectorized Rank Adaptation.
     * Uses shared frozen random matrices with learned scaling vectors.
     * Extremely parameter efficient.
     */
    VERA,

    /**
     * DyLoRA - Dynamic Low-Rank Adaptation.
     * Trains with nested ranks simultaneously.
     */
    DYLORA,

    /**
     * LoftQ - LoRA-Fine-Tuning-aware Quantization.
     * Initializes LoRA matrices via SVD of the pretrained weight residual
     * after quantization. Provides better initialization than random/Kaiming
     * for quantized models. At inference/training time it behaves as LORA.
     * <p>
     * Key parameters: same as LORA plus quantization settings (quantType,
     * quantBits, blockSize, numIterations)
     *
     * @see <a href="https://arxiv.org/abs/2310.08659">LoftQ Paper</a>
     */
    LOFTQ
}
