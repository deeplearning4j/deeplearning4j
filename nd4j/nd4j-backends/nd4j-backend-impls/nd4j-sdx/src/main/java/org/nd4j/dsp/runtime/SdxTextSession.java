/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.dsp.runtime;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.dsp.runtime.bindings.SdxNative;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_cancel_callback_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_generation_callbacks_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_generation_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_generation_report_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_generation_session_options_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_generation_session_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_model_t;
import org.nd4j.dsp.runtime.bindings.SdxNative.sdx_token_callback_t;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BooleanSupplier;
import java.util.function.LongConsumer;

/**
 * Thin JavaCPP lifecycle facade over the reusable native SDX generation
 * session.
 *
 * <p>This class does not implement graph-name discovery, tensor setup, KV-cache
 * management, sampling, or an autoregressive loop. Those responsibilities stay
 * in the portable SDX C implementation so Android/JavaCPP and iOS/Swift execute
 * exactly the same engine. The Java layer only owns native resource lifetime,
 * callback reachability, and Java-friendly value objects.</p>
 *
 * <p>A session is not concurrently generatable. {@link #cancel()} is the one
 * operation intentionally safe from another thread while generation is in
 * flight.</p>
 */
public final class SdxTextSession implements AutoCloseable {

    public enum FinishReason {
        NONE,
        MAX_TOKENS,
        EOS,
        CANCELLED,
        CONTEXT_LIMIT,
        UNKNOWN;

        private static FinishReason fromNative(int value) {
            switch (value) {
                case SdxNative.SDX_GENERATION_FINISH_NONE:
                    return NONE;
                case SdxNative.SDX_GENERATION_FINISH_MAX_TOKENS:
                    return MAX_TOKENS;
                case SdxNative.SDX_GENERATION_FINISH_EOS:
                    return EOS;
                case SdxNative.SDX_GENERATION_FINISH_CANCELLED:
                    return CANCELLED;
                case SdxNative.SDX_GENERATION_FINISH_CONTEXT_LIMIT:
                    return CONTEXT_LIMIT;
                default:
                    return UNKNOWN;
            }
        }
    }

    /**
     * An explicit override of all bundle-authored sampling defaults.
     *
     * <p>Pass {@code null}, or use an overload without this object, to consume
     * the defaults stored in the SDX bundle metadata. Supplying this object
     * replaces the complete native {@code TokenSampleConfig} policy.</p>
     */
    public static final class GenerationOptions {
        private int maxNewTokens;
        private int minNewTokens;
        private double temperature;
        private int topK;
        private double topP = 1.0;
        private double minP;
        private double repetitionPenalty = 1.0;
        private double frequencyPenalty;
        private double presencePenalty;
        private double typicalP = 1.0;
        private double xtcProbability;
        private double xtcThreshold = 0.1;
        private long seed;

        public GenerationOptions(int maxNewTokens) {
            this.maxNewTokens = positive(maxNewTokens, "maxNewTokens");
        }

        public GenerationOptions maxNewTokens(int value) {
            maxNewTokens = positive(value, "maxNewTokens");
            return this;
        }

        public GenerationOptions minNewTokens(int value) {
            if (value < 0) {
                throw new IllegalArgumentException("minNewTokens must be non-negative");
            }
            minNewTokens = value;
            return this;
        }

        public GenerationOptions temperature(double value) {
            temperature = value;
            return this;
        }

        public GenerationOptions topK(int value) {
            topK = value;
            return this;
        }

        public GenerationOptions topP(double value) {
            topP = value;
            return this;
        }

        public GenerationOptions minP(double value) {
            minP = value;
            return this;
        }

        public GenerationOptions repetitionPenalty(double value) {
            repetitionPenalty = value;
            return this;
        }

        public GenerationOptions frequencyPenalty(double value) {
            frequencyPenalty = value;
            return this;
        }

        public GenerationOptions presencePenalty(double value) {
            presencePenalty = value;
            return this;
        }

        public GenerationOptions typicalP(double value) {
            typicalP = value;
            return this;
        }

        public GenerationOptions xtc(double probability, double threshold) {
            xtcProbability = probability;
            xtcThreshold = threshold;
            return this;
        }

        public GenerationOptions seed(long value) {
            seed = value;
            return this;
        }

        private sdx_generation_options_t toNative() {
            sdx_generation_options_t result = new sdx_generation_options_t();
            result.struct_size(result.sizeof())
                    .max_new_tokens(maxNewTokens)
                    .min_new_tokens(minNewTokens)
                    .temperature(temperature)
                    .top_k(topK)
                    .top_p(topP)
                    .min_p(minP)
                    .repetition_penalty(repetitionPenalty)
                    .frequency_penalty(frequencyPenalty)
                    .presence_penalty(presencePenalty)
                    .typical_p(typicalP)
                    .xtc_probability(xtcProbability)
                    .xtc_threshold(xtcThreshold)
                    .seed(seed);
            return result;
        }

        private static int positive(int value, String name) {
            if (value <= 0) {
                throw new IllegalArgumentException(name + " must be positive");
            }
            return value;
        }
    }

    public static final class GenerationReport {
        private final FinishReason finishReason;
        private final int nativeFinishReason;
        private final int promptTokenCount;
        private final int generatedTokenCount;
        private final int totalGeneratedTokenCount;
        private final int contextPosition;
        private final long elapsedTimeNanos;
        private final long prefillTimeNanos;
        private final long decodeTimeNanos;
        private final double decodeTokensPerSecond;

        private GenerationReport(sdx_generation_report_t source) {
            nativeFinishReason = source.finish_reason();
            finishReason = FinishReason.fromNative(nativeFinishReason);
            promptTokenCount = source.prompt_token_count();
            generatedTokenCount = source.generated_token_count();
            totalGeneratedTokenCount = source.total_generated_token_count();
            contextPosition = source.context_position();
            elapsedTimeNanos = source.elapsed_time_ns();
            prefillTimeNanos = source.prefill_time_ns();
            decodeTimeNanos = source.decode_time_ns();
            decodeTokensPerSecond = source.decode_tokens_per_second();
        }

        public FinishReason finishReason() {
            return finishReason;
        }

        public int nativeFinishReason() {
            return nativeFinishReason;
        }

        public int promptTokenCount() {
            return promptTokenCount;
        }

        public int generatedTokenCount() {
            return generatedTokenCount;
        }

        public int totalGeneratedTokenCount() {
            return totalGeneratedTokenCount;
        }

        public int contextPosition() {
            return contextPosition;
        }

        public long elapsedTimeNanos() {
            return elapsedTimeNanos;
        }

        public long prefillTimeNanos() {
            return prefillTimeNanos;
        }

        public long decodeTimeNanos() {
            return decodeTimeNanos;
        }

        public double decodeTokensPerSecond() {
            return decodeTokensPerSecond;
        }
    }

    public static final class GenerationResult {
        private final long[] tokenIds;
        private final GenerationReport report;

        private GenerationResult(long[] tokenIds, GenerationReport report) {
            this.tokenIds = tokenIds;
            this.report = report;
        }

        public long[] tokenIds() {
            return Arrays.copyOf(tokenIds, tokenIds.length);
        }

        public int tokenCount() {
            return tokenIds.length;
        }

        public GenerationReport report() {
            return report;
        }
    }

    private final SdxRuntime runtime;
    private final SdxRuntime.SdxModel modelOwner;
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private sdx_generation_session_t sessionHandle;

    private SdxTextSession(
            SdxRuntime runtime,
            SdxRuntime.SdxModel modelOwner,
            sdx_generation_session_t sessionHandle) {
        this.runtime = runtime;
        this.modelOwner = modelOwner;
        this.sessionHandle = sessionHandle;
    }

    static SdxTextSession create(
            SdxRuntime runtime,
            SdxRuntime.SdxModel modelOwner,
            sdx_model_t modelHandle) {
        Objects.requireNonNull(runtime, "runtime");
        Objects.requireNonNull(modelOwner, "modelOwner");

        try (sdx_generation_session_options_t options =
                     new sdx_generation_session_options_t()) {
            options.struct_size(options.sizeof());
            sdx_generation_session_t outSession =
                    new sdx_generation_session_t();
            int status = SdxNative.sdxCreateGenerationSession(
                    modelHandle, options, outSession);
            if (status != SdxRuntime.SDX_STATUS_OK || Pointer.isNull(outSession)) {
                throw new IllegalStateException(
                        "sdxCreateGenerationSession failed: "
                                + runtime.lastError() + " (status=" + status + ")");
            }
            return new SdxTextSession(runtime, modelOwner, outSession);
        }
    }

    public GenerationResult generate(long[] promptTokenIds) {
        return generate(promptTokenIds, null, null, null);
    }

    public GenerationResult generate(
            long[] promptTokenIds,
            LongConsumer onToken) {
        return generate(promptTokenIds, null, onToken, null);
    }

    public synchronized GenerationResult generate(
            long[] promptTokenIds,
            GenerationOptions options,
            LongConsumer onToken,
            BooleanSupplier shouldCancel) {
        Objects.requireNonNull(promptTokenIds, "promptTokenIds");
        if (promptTokenIds.length == 0) {
            throw new IllegalArgumentException("promptTokenIds must not be empty");
        }
        return execute(false, promptTokenIds, options, onToken, shouldCancel);
    }

    public GenerationResult continueGeneration() {
        return continueGeneration(null, null, null);
    }

    public GenerationResult continueGeneration(LongConsumer onToken) {
        return continueGeneration(null, onToken, null);
    }

    public synchronized GenerationResult continueGeneration(
            GenerationOptions options,
            LongConsumer onToken,
            BooleanSupplier shouldCancel) {
        return execute(true, null, options, onToken, shouldCancel);
    }

    /**
     * Cooperatively cancel an active native decode at its next committed token
     * boundary. This method deliberately does not synchronize on the generation
     * monitor.
     */
    public void cancel() {
        lifecycleLock.readLock().lock();
        try {
            if (!Pointer.isNull(sessionHandle)) {
                SdxNative.sdxCancelGeneration(sessionHandle);
            }
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    public synchronized void reset() {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            checkStatus(
                    SdxNative.sdxResetGenerationSession(sessionHandle),
                    "sdxResetGenerationSession");
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private GenerationResult execute(
            boolean continuation,
            long[] promptTokenIds,
            GenerationOptions options,
            LongConsumer onToken,
            BooleanSupplier shouldCancel) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
            TokenCollector collector = new TokenCollector();
            AtomicReference<Throwable> callbackFailure = new AtomicReference<>();

            try (TokenCallback tokenCallback =
                         new TokenCallback(collector, onToken, callbackFailure);
                 CancelCallback cancelCallback =
                         new CancelCallback(shouldCancel, callbackFailure);
                 sdx_generation_callbacks_t callbacks =
                         new sdx_generation_callbacks_t();
                 sdx_generation_options_t nativeOptions =
                         options == null ? null : options.toNative();
                 LongPointer prompt =
                         continuation ? null : new LongPointer(promptTokenIds);
                 IntPointer outCount = new IntPointer(1);
                 sdx_generation_report_t nativeReport =
                         new sdx_generation_report_t()) {
                callbacks.struct_size(callbacks.sizeof())
                        .on_token(tokenCallback)
                        .should_cancel(cancelCallback)
                        .user_data(null);
                nativeReport.struct_size(nativeReport.sizeof());
                outCount.put(0, 0);

                int status;
                if (continuation) {
                    status = SdxNative.sdxGenerationContinue(
                            sessionHandle,
                            nativeOptions,
                            callbacks,
                            (LongPointer) null,
                            0,
                            outCount,
                            nativeReport);
                } else {
                    status = SdxNative.sdxGenerationGenerate(
                            sessionHandle,
                            prompt,
                            promptTokenIds.length,
                            nativeOptions,
                            callbacks,
                            (LongPointer) null,
                            0,
                            outCount,
                            nativeReport);
                }
                checkStatus(
                        status,
                        continuation
                                ? "sdxGenerationContinue"
                                : "sdxGenerationGenerate");

                Throwable failure = callbackFailure.get();
                if (failure != null) {
                    throw new IllegalStateException(
                            "SDX generation callback failed", failure);
                }

                int nativeCount = outCount.get(0);
                if (nativeCount != collector.size()) {
                    throw new IllegalStateException(
                            "SDX callback/output count mismatch: native="
                                    + nativeCount + ", callbacks=" + collector.size());
                }
                return new GenerationResult(
                        collector.toArray(),
                        new GenerationReport(nativeReport));
            }
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    private void checkStatus(int status, String operation) {
        if (status != SdxRuntime.SDX_STATUS_OK) {
            throw new IllegalStateException(
                    operation + " failed: " + runtime.lastError()
                            + " (status=" + status + ")");
        }
    }

    private void ensureOpen() {
        if (Pointer.isNull(sessionHandle)) {
            throw new IllegalStateException("SDX text session is closed");
        }
    }

    @Override
    public synchronized void close() {
        lifecycleLock.writeLock().lock();
        try {
            if (Pointer.isNull(sessionHandle)) {
                return;
            }
            SdxNative.sdxDestroyGenerationSession(sessionHandle);
            sessionHandle.setNull();
            sessionHandle = null;
            modelOwner.textSessionClosed();
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    private static final class TokenCollector {
        private long[] values = new long[16];
        private int size;

        private void add(long value) {
            if (size == values.length) {
                values = Arrays.copyOf(values, values.length * 2);
            }
            values[size++] = value;
        }

        private int size() {
            return size;
        }

        private long[] toArray() {
            return Arrays.copyOf(values, size);
        }
    }

    private static final class TokenCallback extends sdx_token_callback_t {
        private final TokenCollector collector;
        private final LongConsumer consumer;
        private final AtomicReference<Throwable> failure;

        private TokenCallback(
                TokenCollector collector,
                LongConsumer consumer,
                AtomicReference<Throwable> failure) {
            this.collector = collector;
            this.consumer = consumer;
            this.failure = failure;
        }

        @Override
        public void call(long tokenId, Pointer userData) {
            collector.add(tokenId);
            if (consumer == null || failure.get() != null) {
                return;
            }
            try {
                consumer.accept(tokenId);
            } catch (Throwable t) {
                failure.compareAndSet(null, t);
            }
        }
    }

    private static final class CancelCallback extends sdx_cancel_callback_t {
        private final BooleanSupplier supplier;
        private final AtomicReference<Throwable> failure;

        private CancelCallback(
                BooleanSupplier supplier,
                AtomicReference<Throwable> failure) {
            this.supplier = supplier;
            this.failure = failure;
        }

        @Override
        public int call(Pointer userData) {
            if (failure.get() != null) {
                return 1;
            }
            if (supplier == null) {
                return 0;
            }
            try {
                return supplier.getAsBoolean() ? 1 : 0;
            } catch (Throwable t) {
                failure.compareAndSet(null, t);
                return 1;
            }
        }
    }
}
