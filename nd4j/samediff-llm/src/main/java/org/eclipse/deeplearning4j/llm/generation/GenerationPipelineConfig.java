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
import org.nd4j.autodiff.samediff.SameDiff;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.speculative.Speculator;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;

import java.util.List;
import java.util.Set;


/**
 * Configuration for {@link GenerationPipeline}.
 *
 * <p>Encapsulates all parameters needed to set up an LLM text generation pipeline:
 * the decoder model, embedding model, tokenizer, sampling strategy, KV cache strategy,
 * and optional speculative decoding.</p>
 *
 * <p>Usage (VLM / two-model mode):</p>
 * <pre>{@code
 * GenerationPipelineConfig config = GenerationPipelineConfig.builder()
 *     .decoder(decoder)
 *     .embedTokens(embedTokens)
 *     .tokenizer(tokenizer)
 *     .samplingConfig(SamplingConfig.greedy())
 *     .maxNewTokens(256)
 *     .build();
 * }</pre>
 *
 * <p>Usage (GGUF / single-model mode — embeddings extracted from decoder):</p>
 * <pre>{@code
 * GenerationPipelineConfig config = GenerationPipelineConfig.builder()
 *     .decoder(model)
 *     .tokenizer(tokenizer)
 *     .maxNewTokens(256)
 *     .build();
 * }</pre>
 */
@Getter
@Builder
public class GenerationPipelineConfig {

    /** The decoder SameDiff model. Required. */
    private final SameDiff decoder;

    /** The token embedding SameDiff model. Optional — if null, embeddings are extracted from the decoder (single-model mode). */
    private final SameDiff embedTokens;

    /** The tokenizer for encoding/decoding text. Required. */
    private final Tokenizer tokenizer;

    /** Sampling configuration (temperature, top-k, top-p, etc.). */
    @Builder.Default
    private final SamplingConfig samplingConfig = SamplingConfig.greedy();

    /** Maximum number of new tokens to generate. */
    @Builder.Default
    private final int maxNewTokens = 256;

    /**
     * Fixed maximum prefill (prompt) length for pre-allocated buffers.
     * When &gt; 0, all KV cache buffers, causal masks, and attention masks are
     * allocated at this fixed size so the DSP plan is compiled once and
     * reused across generations with different prompt lengths — no plan
     * reset or recompilation needed. Prompts shorter than this are
     * right-padded; prompts longer are truncated with a warning.
     *
     * <p>When 0 (default), buffers are sized per-call to
     * {@code promptLen + maxNewTokens}, which requires DSP reset between
     * calls with different prompt lengths.</p>
     */
    @Builder.Default
    private final int maxPrefillLength = 0;

    /**
     * Hard upper bound on total KV cache sequence length.
     * When &gt; 0, the effective {@code maxKvLen} used for buffer allocation and
     * attention masks is clamped to this value regardless of
     * {@code prefillSeqLen + maxNewTokens}. This prevents shape hash changes
     * (and costly plan recompilation) when callers vary {@code maxNewTokens}
     * between calls on a cached pipeline.
     *
     * <p>When 0 (default), no capping is applied and buffers are sized to
     * {@code prefillSeqLen + maxNewTokens} each call.</p>
     */
    @Builder.Default
    private final int maxKvCacheLength = 0;

    /**
     * Default total continuation capacity, in new tokens, for
     * {@link GenerationPipeline#startSession(String)}.
     *
     * <p>A "continue generation" session pre-sizes its STATIC KV buffer <em>once</em> to
     * {@code prefillLen + capacity} so decoding can resume across calls without reallocation. When
     * {@code > 0} this is the capacity used by the no-argument {@code startSession(prompt)}. When
     * {@code 0} (default) the capacity is derived from the context ceiling
     * ({@code maxKvCacheLength - prefillLen}) or, failing that, {@link #maxNewTokens}.</p>
     *
     * <p>To get true context-ceiling continuation, set {@link #maxKvCacheLength} to the model's
     * context window (e.g. 512), or pass an explicit capacity to
     * {@link GenerationPipeline#startSession(String, int)}.</p>
     */
    @Builder.Default
    private final int sessionCapacity = 0;

    /**
     * How a continuation session's KV cache capacity is managed. Defaults to
     * {@link KvContinuationMode#STATIC_CONTEXT_CEILING} (static buffer sized to the context ceiling);
     * {@link KvContinuationMode#GROWABLE} is reserved and not yet implemented.
     */
    @Builder.Default
    private final KvContinuationMode kvContinuationMode = KvContinuationMode.STATIC_CONTEXT_CEILING;

    /**
     * Degenerate-loop ("thinking trap") guard applied only by
     * {@link GenerationPipeline.GenerationSession#continueToCompletion(int)}. Defaults to
     * {@link RepetitionGuard#defaultOn()}. Never affects the numerically-pure {@code generate} /
     * {@code continueGeneration} primitives.
     */
    @Builder.Default
    private final RepetitionGuard repetitionGuard = RepetitionGuard.defaultOn();

    /** Model hidden size. Auto-detected from embeddings if 0. */
    @Builder.Default
    private final long hiddenSize = 0;

    /** KV cache strategy. */
    @Builder.Default
    private final KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;

    /** TurboQuant bit budget per coordinate for TURBOQUANT KV cache strategy. Defaults to 3. */
    @Builder.Default
    private final int turboQuantBits = 3;

    /**
     * KV cache quantization format for QUANTIZED strategy. 0 = disabled (default), 1 = INT8,
     * 2 = FP8_E4M3, 3 = FP8_E5M2, 4 = INT4. Matches {@code ConversionOptions.kvQuantFormat}
     * and the C++ {@code KVQuantFormat} enum. Has no effect unless
     * {@link #kvCacheStrategy} == {@link org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy#QUANTIZED}
     * and the decoder graph was built with matching {@code ConversionOptions.kvQuantFormat}.
     */
    @Builder.Default
    private final int kvQuantFormat = 0;

    /** Optional model I/O configuration. Auto-discovered if null. */
    private final ModelIOConfig ioConfig;

    /** Optional speculator for speculative decoding. */
    private final Speculator speculator;

    /** Maximum speculative tokens per step (0 disables speculation). */
    @Builder.Default
    private final int maxSpeculativeTokens = 0;

    /** Additional stop token IDs beyond EOS. */
    private final Set<Integer> additionalStopTokenIds;

    /** Additional ordered token sequences. Unlike scalar stops these match only as full suffixes. */
    private final List<int[]> additionalStopTokenSequences;

    public List<int[]> getAdditionalStopTokenSequences() {
        if (additionalStopTokenSequences == null) return java.util.Collections.emptyList();
        List<int[]> copy = new java.util.ArrayList<>(additionalStopTokenSequences.size());
        for (int[] sequence : additionalStopTokenSequences) {
            if (sequence != null) copy.add(sequence.clone());
        }
        return java.util.Collections.unmodifiableList(copy);
    }

    /** Whether imported model stop IDs participate in this request. */
    @Builder.Default private final boolean inheritModelStopTokenIds = true;

    /** Whether imported/tokenizer EOS participates when sampling does not explicitly override it. */
    @Builder.Default private final boolean inheritDefaultEos = true;

    /** Whether atomic assistant-turn stops inferred from the chat template participate. */
    @Builder.Default private final boolean inheritChatTemplateStopTokenIds = true;

    /** Name of the embedding model input (auto-discovered if null). */
    private final String embedInputName;

    /** Names of the embedding model outputs (auto-discovered if null). */
    private final String[] embedOutputNames;

    /** Path to decoder ONNX/SDZ model file. Alternative to providing a pre-loaded decoder. */
    private final String decoderPath;

    /** Path to embed_tokens ONNX/SDZ model file. Alternative to providing a pre-loaded embedTokens. */
    private final String embedTokensPath;

    /** Path to draft decoder ONNX/SDZ model for speculative decoding. */
    private final String draftModelPath;

    /** Pre-loaded draft decoder model for speculative decoding. */
    private final SameDiff draftDecoder;

    /** Whether the GraphOptimizer runs during pipeline construction (fuses rms_norm, swish_mul, xw_plus_b, etc.).
     *  Defaults to true — models from any import path (GGUF, ONNX, etc.) benefit from
     *  graph-level fusion and simplification. The optimizer is idempotent. */
    @Builder.Default
    private final boolean graphOptimizerEnabled = true;

    /** Whether DSP (DynamicShapePlan) is enabled for this pipeline. */
    @Builder.Default
    private final boolean dspEnabled = true;

    /** Optional benchmark config applied to the optimized in-pipeline SameDiff models. */
    private final BenchmarkConfig benchmarkConfig;

    /** Optional model loader for loading models from paths. Uses default loader if null. */
    private final GenerationPipeline.ModelLoader modelLoader;

    /** Optional preprocessor config for VLM image preprocessing. Nullable for text-only pipelines. */
    private final PreprocessorConfig preprocessorConfig;

    /** Optional path to preprocessor_config.json for auto-loading. */
    private final String preprocessorConfigPath;

    /**
     * Chat template string (Jinja2 format from tokenizer_config.json or GGUF metadata).
     * When set, plain text prompts are wrapped with this template before tokenization.
     * Supports ChatML ({@code <|im_start|>}), Llama ([INST]), and other common formats.
     * If null, prompts are tokenized as-is (raw completion mode).
     */
    private final String chatTemplate;

    /**
     * Model-owned generation metadata for a caller-supplied decoder graph.
     *
     * <p>When a caller builds the pipeline around an ALREADY-imported SameDiff
     * ({@code .decoder(...)}), the import boundary would otherwise discard the source
     * container's protocol metadata (BOS/EOS/pad ids, chat template, special tokens) and
     * every caller has to re-derive it by hand. Loaders recover this automatically via
     * {@code ModelLoader#getModelMetadata}; this field is the equivalent seam for callers
     * that imported the graph themselves (e.g. via GGUF import with metadata) and hand the
     * graph to the pipeline. Explicit config wins over loader-provided metadata.</p>
     */
    @Builder.Default
    private final GenerationPipeline.ModelMetadata modelMetadata =
            GenerationPipeline.ModelMetadata.empty();

    /** Shape used when exposing tool definitions to the active template. */
    @Builder.Default
    private final org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolDefinitionFormat toolDefinitionFormat =
            org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolDefinitionFormat.STANDARD;

    /**
     * Optional tool-call envelope override. When null, the imported tokenizer/model
     * chat template selects its own protocol.
     */
    private final org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate.ToolCallFormat toolCallFormat;

    // ── Rotating KV cache (StreamingLLM-style attention sinks + sliding window) ─────────────────

    /**
     * Enable StreamingLLM-style rotating KV cache (attention sinks + sliding window).
     *
     * <p>When {@code true}, the fixed-size KV buffer is treated as a ring: the first
     * {@link #rotatingKvSinkCount} tokens are pinned as "attention sinks" and never evicted;
     * all subsequent tokens rotate through the remaining slots, oldest-first. This removes the
     * hard generation stop that occurs when the buffer fills, enabling unbounded generation
     * within a fixed memory footprint.</p>
     *
     * <p><strong>Quality caveat:</strong> for sessions whose total token count never exceeds
     * {@code maxKvLen}, behavior is bit-identical to rotating disabled. Beyond that boundary,
     * evicted non-sink keys retain their original RoPE encoding (post-RoPE cache); the model
     * sees a positional jump between sinks and the recent window — the StreamingLLM approximation.
     * Quality degrades gracefully for most LLMs (empirically acceptable for streaming) but is
     * NOT bit-identical to full-history attention.</p>
     *
     * <p><strong>GDN / recurrent-state models</strong> (Qwen GDN): recurrent state is
     * position-free and is unaffected by rotation. Only attention KV buffers rotate.</p>
     *
     * <p>CUDA-graph safe: the KV buffer shape is fixed; ring semantics are expressed as data
     * (physical write position and attention mask values) never as structural changes.</p>
     *
     * <p>Defaults to {@code false} (conservative — preserves exact current behavior).</p>
     *
     * @see #rotatingKvSinkCount
     * @see RotatingKvSlotMap
     */
    @Builder.Default
    private final boolean rotatingKvEnabled = false;

    /**
     * Number of "attention sink" tokens pinned at the start of the KV buffer when rotating KV is
     * enabled. Sinks are never evicted; their K/V occupy physical slots 0 .. sinkCount-1.
     *
     * <p>Defaults to {@link RotatingKvSlotMap#DEFAULT_SINK_COUNT} (4) when {@code 0}.
     * Can also be set via system property {@code nd4j.generation.rotatingKv.sinkCount}.</p>
     *
     * <p>Ignored when {@link #rotatingKvEnabled} is {@code false}.</p>
     */
    @Builder.Default
    private final int rotatingKvSinkCount = 0;   // 0 → resolved via RotatingKvSlotMap.resolveSinkCount()

    /**
     * When {@code true}, generation prefill will request the {@code "lm_logits_last"} output
     * (last-position logits only, shape {@code [B, 1, vocab]}) instead of full
     * {@code "lm_logits"} ({@code [B, S, vocab]}).  DSP then naturally prunes the graph to only
     * compute logits for the final hidden state, skipping the full-vocab projection over all S
     * prompt positions — a TTFT win proportional to prompt length and vocabulary size.
     *
     * <p>This optimisation applies ONLY when:</p>
     * <ul>
     *   <li>The decoder graph contains a variable named {@code "lm_logits_last"}, AND</li>
     *   <li>The pipeline is running prefill (sequence length &gt; 1). Decode steps (S=1) are
     *       unchanged regardless of this flag.</li>
     * </ul>
     *
     * <p>For decoder graphs that do NOT have {@code "lm_logits_last"}, the flag is silently
     * ignored and the full-logits path is used.</p>
     *
     * <p>Defaults to {@code false} (conservative). Enable via this flag or the system property
     * {@code nd4j.generation.prefill.lastPositionLogits=true}.  Set to {@code false} to
     * unconditionally disable regardless of the system property.</p>
     *
     * @see #PREFILL_LAST_POSITION_LOGITS_PROP
     */
    @Builder.Default
    private final boolean prefillLastPositionLogitsEnabled = false;

    /**
     * System property name for the prefill last-position logits optimisation.
     * Value {@code "true"} enables it when {@link #prefillLastPositionLogitsEnabled} is also
     * {@code true} (the config flag gates it; this property provides a runtime kill-switch).
     */
    public static final String PREFILL_LAST_POSITION_LOGITS_PROP = "nd4j.generation.prefill.lastPositionLogits";

    /**
     * Returns the effective value of the prefill last-position logits optimisation,
     * combining the config flag with the system property (system property can disable but not enable).
     */
    public boolean isEffectivePrefillLastPositionLogitsEnabled() {
        if (!prefillLastPositionLogitsEnabled) return false;
        // System property can disable (kill-switch), but cannot enable beyond the config flag
        String prop = System.getProperty(PREFILL_LAST_POSITION_LOGITS_PROP);
        if ("false".equalsIgnoreCase(prop)) return false;
        return true;
    }

    // ── Cross-request KV prefix cache ────────────────────────────────────────────────

    /**
     * Enable cross-request KV prefix caching.
     *
     * <p>When {@code true}, completed prefills are stored in a device-resident block pool keyed by
     * their token hashes (via {@link org.eclipse.deeplearning4j.llm.generation.kvcache.RadixPrefixCache}).
     * A subsequent generate call whose prompt shares a block-aligned prefix with a cached entry will
     * restore those KV blocks directly from the pool, then run prefill only for the <em>uncached
     * suffix</em> — reducing TTFT proportionally to the shared prefix length.</p>
     *
     * <p><strong>Restrictions (v1):</strong></p>
     * <ul>
     *   <li>Only supported with {@code STATIC} KV cache strategy (not QUANTIZED or TURBOQUANT).</li>
     *   <li>Not compatible with {@code rotatingKvEnabled} (rotating changes physical write positions).</li>
     *   <li>For GDN/recurrent-state models, recurrent state reuse is possible only on exact block-boundary
     *       hits; partial-boundary hits fall back to full re-prefill of the suffix from zero recurrent
     *       state (correct but slower).</li>
     * </ul>
     *
     * <p>Defaults to {@code false} — zero behavior change unless opted in.</p>
     */
    @Builder.Default
    private final boolean prefixCacheEnabled = false;

    /**
     * Maximum device bytes for the cross-request KV prefix block pool.
     *
     * <p>When {@code prefixCacheEnabled} is {@code true}, the pool evicts LRU blocks once this
     * budget is reached. A value of {@code 0} (default) applies a heuristic: 10% of the device's
     * total memory (or 512 MB on CPU), whichever is smaller.</p>
     *
     * <p>Ignored when {@link #prefixCacheEnabled} is {@code false}.</p>
     */
    @Builder.Default
    private final long prefixCacheMaxBytes = 0L;

    /**
     * Block size (in tokens) for the cross-request KV prefix pool.
     *
     * <p>Prefix matches are aligned to this granularity. Larger blocks reduce trie overhead but
     * require longer shared prefixes for a cache hit. Smaller blocks increase trie size but find
     * partial hits more often. Defaults to
     * {@link org.eclipse.deeplearning4j.llm.generation.kvcache.KvPrefixBlockPool#DEFAULT_BLOCK_SIZE}
     * (16 tokens).</p>
     *
     * <p>Ignored when {@link #prefixCacheEnabled} is {@code false}.</p>
     */
    @Builder.Default
    private final int prefixCacheBlockSize = 0;   // 0 → resolved to DEFAULT_BLOCK_SIZE
}
