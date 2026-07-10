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

/**
 * How a {@link GenerationPipeline.GenerationSession}'s KV cache capacity is managed for
 * "continue generation" (resume / incremental decode).
 *
 * <p>A session prefills once and then resumes autoregressive decoding across multiple calls,
 * reusing the already-populated in-graph KV cache. The KV buffer is a fixed-size (STATIC) tensor,
 * so how much continuation is possible depends on how the buffer is sized up front.</p>
 */
public enum KvContinuationMode {

    /**
     * Default. The STATIC KV buffer is pre-sized <em>once</em>, at session start, to the model's
     * context ceiling (the resolved session capacity — see
     * {@link GenerationPipeline#startSession(String, int)}). Continuation reuses the unused portion
     * of that budget; when the buffer fills, the next continue returns
     * {@link GenerationResult.FinishReason#MAX_TOKENS} with a "context full" signal.
     *
     * <p>This keeps the frozen DSP plan / CUDA-graph pointer contract intact (no reallocation),
     * which is why it is the supported default.</p>
     */
    STATIC_CONTEXT_CEILING,

    /**
     * Reserved for a future growable / dynamic KV cache that reallocates and re-captures when
     * decoding beyond the original budget. <strong>Not implemented in this release</strong> — selecting
     * it throws {@link UnsupportedOperationException}. Tracked as a follow-up ADR; growing the buffer
     * requires re-establishing frozen-plan pointer stability and CUDA-graph capture/replay.
     */
    GROWABLE
}
