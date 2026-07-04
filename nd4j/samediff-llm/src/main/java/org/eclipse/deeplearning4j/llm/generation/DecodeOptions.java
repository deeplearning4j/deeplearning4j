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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.Builder;
import lombok.Getter;
import org.eclipse.deeplearning4j.llm.generation.speculative.Speculator;

/**
 * Per-call decode options for {@link GenerationPipeline#generate(org.nd4j.linalg.api.ndarray.INDArray, int[], int, DecodeOptions)}.
 *
 * <p>These options override pipeline-level configuration for a single generate() call.
 * Useful for benchmark/test harnesses that sweep multiple configurations against the
 * same pipeline.</p>
 */
@Getter
@Builder
public class DecodeOptions {

    /** Enable per-step argmax + top-K logit tracing. */
    @Builder.Default
    private final boolean argmaxTraceEnabled = false;

    /** Number of top-K logit entries to emit when argmaxTraceEnabled is on. */
    @Builder.Default
    private final int argmaxTraceTopK = 5;

    /** Reference token stream for golden-token comparison. Null to disable. */
    private final int[] referenceTokenStream;

    /** Override maximum speculative tokens for this call. 0 = use pipeline default. */
    @Builder.Default
    private final int maxSpeculativeTokens = 0;

    /** Override speculator for this call. Null = use pipeline's built-in speculator. */
    private final Speculator speculator;

    /**
     * Encoder hidden states for encoder-decoder models (Whisper, T5-style): fed to
     * the decoder input declared by {@code ModelIOConfig.encoderHiddenStatesName}
     * (cross-attention keys/values source). Constant across decode steps — supplied
     * once at prefill, persisted by the frozen plan's external-input table. Null for
     * decoder-only models.
     */
    private final org.nd4j.linalg.api.ndarray.INDArray encoderOutputs;

    /** Optional encoder attention mask companion to {@link #encoderOutputs}. */
    private final org.nd4j.linalg.api.ndarray.INDArray encoderAttentionMask;
}
