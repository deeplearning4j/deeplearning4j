/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.audio.synthesis;

import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Loaded end-to-end SameDiff text-to-waveform model that materializes one
 * completed PCM WAV file per request.
 *
 * <p>This is the reusable model boundary. It owns tokenization, tensor binding,
 * inference, output validation, streaming WAV writing, and native resource
 * lifecycle. Serving applications retain responsibility for registration,
 * model integrity, filesystem policy, artifact storage, and transport.</p>
 *
 * <p>The graph output must already be normalized mono waveform samples. This
 * class deliberately does not invent a vocoder, normalize model output, or
 * accept arbitrary inference parameters.</p>
 */
public final class SameDiffWaveformGenerator implements AudioFileGenerator {

    public enum TokenizerType {
        HUGGING_FACE,
        UTF8_BYTES
    }

    private final SameDiff sameDiff;
    private final Tokenizer tokenizer;
    private final Builder config;
    private final PcmWavFileWriter writer;

    private SameDiffWaveformGenerator(Builder config) throws Exception {
        this.config = config.validate();
        SameDiff loaded = null;
        Tokenizer loadedTokenizer = null;
        try {
            loaded = SameDiff.load(config.modelFile.toFile(), true);
            requireVariable(loaded, config.tokenIdsInput, "tokenIdsInput");
            requireOptionalVariable(loaded, config.attentionMaskInput, "attentionMaskInput");
            requireOptionalVariable(loaded, config.tokenLengthsInput, "tokenLengthsInput");
            requireVariable(loaded, config.waveformOutput, "waveformOutput");
            requireOptionalVariable(loaded, config.confidenceOutput, "confidenceOutput");
            if (config.tokenizerType == TokenizerType.HUGGING_FACE) {
                loadedTokenizer = HuggingFaceTokenizer.fromFile(config.tokenizerFile.toFile());
            }
            this.sameDiff = loaded;
            this.tokenizer = loadedTokenizer;
            this.writer = new PcmWavFileWriter(config.sampleRateHz, config.maxOutputSamples);
        } catch (Exception | Error failure) {
            closeAfterFailedLoad(loadedTokenizer, loaded, failure);
            throw failure;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public GeneratedAudioFile generate(AudioSynthesisRequest request, Path outputDirectory)
            throws Exception {
        validateRequest(request);
        EncodedInput encoded = encode(request.getText());
        if (encoded.ids.length == 0) {
            throw new IllegalArgumentException("tokenizer returned no input tokens");
        }
        if (encoded.ids.length > config.maxInputTokens) {
            throw new IllegalArgumentException("input exceeds the configured token limit");
        }

        Map<String, INDArray> placeholders = new LinkedHashMap<>();
        placeholders.put(config.tokenIdsInput,
                Nd4j.createFromArray(encoded.ids).reshape(1, encoded.ids.length)
                        .castTo(config.tokenDataType));
        if (isPresent(config.attentionMaskInput)) {
            placeholders.put(config.attentionMaskInput,
                    Nd4j.createFromArray(encoded.attentionMask)
                            .reshape(1, encoded.attentionMask.length)
                            .castTo(config.tokenDataType));
        }
        if (isPresent(config.tokenLengthsInput)) {
            placeholders.put(config.tokenLengthsInput,
                    Nd4j.createFromArray(encoded.ids.length)
                            .reshape(1).castTo(config.tokenDataType));
        }

        List<String> outputNames = new ArrayList<>();
        outputNames.add(config.waveformOutput);
        if (isPresent(config.confidenceOutput)) {
            outputNames.add(config.confidenceOutput);
        }
        Map<String, INDArray> outputs = sameDiff.output(
                placeholders, outputNames.toArray(new String[0]));
        INDArray waveform = Objects.requireNonNull(
                outputs.get(config.waveformOutput),
                "SameDiff graph returned no configured waveform output");
        double confidence = resolveConfidence(outputs);
        Path completed = writer.write(waveform, outputDirectory,
                request.getRunId() + ".wav");

        Map<String, Object> evidence = new LinkedHashMap<>();
        evidence.put("generator", "samediff_waveform");
        evidence.put("tokenizerType", config.tokenizerType.name().toLowerCase(java.util.Locale.ROOT));
        evidence.put("tokenIdsInput", config.tokenIdsInput);
        evidence.put("waveformOutput", config.waveformOutput);
        evidence.put("sampleRateHz", config.sampleRateHz);
        evidence.put("channels", 1);
        evidence.put("sampleFormat", PcmWavFileWriter.SAMPLE_FORMAT);
        evidence.put("voice", config.voice);
        evidence.put("language", config.language);
        evidence.put("inputTokens", encoded.ids.length);
        evidence.put("outputSamples", waveform.length());
        evidence.put("confidenceSource",
                isPresent(config.confidenceOutput) ? "model_output" : "registered_default");
        evidence.putAll(config.configurationEvidence);

        return new GeneratedAudioFile(
                completed, PcmWavFileWriter.MEDIA_TYPE, config.modelId,
                config.modelVersion, config.configurationVersion, confidence, evidence);
    }

    private void validateRequest(AudioSynthesisRequest request) {
        Objects.requireNonNull(request, "request");
        if (!request.getConfiguration().isEmpty()) {
            throw new IllegalArgumentException(
                    "SameDiff waveform generation does not accept arbitrary request configuration");
        }
        if (!request.getVoice().isBlank() && !config.voice.equals(request.getVoice())) {
            throw new IllegalArgumentException(
                    "requested voice is not the voice configured for this model");
        }
        if (!request.getLanguage().isBlank() && !config.language.equals(request.getLanguage())) {
            throw new IllegalArgumentException(
                    "requested language is not the language configured for this model");
        }
    }

    private EncodedInput encode(String text) {
        if (config.tokenizerType == TokenizerType.UTF8_BYTES) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[bytes.length];
            int[] mask = new int[bytes.length];
            for (int index = 0; index < bytes.length; index++) {
                ids[index] = Byte.toUnsignedInt(bytes[index]);
                mask[index] = 1;
            }
            return new EncodedInput(ids, mask);
        }
        Encoding encoding = tokenizer.encode(text, config.addSpecialTokens);
        int[] ids = Objects.requireNonNull(encoding.getIds(), "tokenizer returned no ids");
        int[] mask = encoding.getAttentionMask();
        if (mask == null) {
            mask = new int[ids.length];
            Arrays.fill(mask, 1);
        }
        if (mask.length != ids.length) {
            throw new IllegalArgumentException(
                    "tokenizer attention mask length does not match token ids");
        }
        return new EncodedInput(ids, mask);
    }

    private double resolveConfidence(Map<String, INDArray> outputs) {
        if (!isPresent(config.confidenceOutput)) {
            return config.defaultConfidence;
        }
        INDArray value = Objects.requireNonNull(
                outputs.get(config.confidenceOutput),
                "SameDiff graph returned no configured confidence output");
        if (value.length() != 1) {
            throw new IllegalArgumentException("confidence output must be a scalar");
        }
        double confidence = value.getDouble(0);
        if (!Double.isFinite(confidence) || confidence < 0.0d || confidence > 1.0d) {
            throw new IllegalArgumentException(
                    "confidence output must be finite and between 0 and 1");
        }
        return confidence;
    }

    @Override
    public void close() throws Exception {
        Exception failure = null;
        try {
            if (tokenizer != null) {
                tokenizer.close();
            }
        } catch (Exception error) {
            failure = error;
        }
        try {
            sameDiff.close();
        } catch (Exception error) {
            if (failure == null) {
                failure = error;
            } else {
                failure.addSuppressed(error);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private static void requireVariable(SameDiff sameDiff, String variable, String field) {
        if (!sameDiff.getVariables().containsKey(variable)) {
            throw new IllegalArgumentException(
                    field + " names a variable that is not present in the SameDiff graph");
        }
    }

    private static void requireOptionalVariable(SameDiff sameDiff, String variable, String field) {
        if (isPresent(variable)) {
            requireVariable(sameDiff, variable, field);
        }
    }

    private static boolean isPresent(String value) {
        return value != null && !value.isBlank();
    }

    private static void closeAfterFailedLoad(Tokenizer tokenizer, SameDiff sameDiff,
                                             Throwable failure) {
        try {
            if (tokenizer != null) {
                tokenizer.close();
            }
        } catch (Exception closeError) {
            failure.addSuppressed(closeError);
        }
        try {
            if (sameDiff != null) {
                sameDiff.close();
            }
        } catch (Exception closeError) {
            failure.addSuppressed(closeError);
        }
    }

    private static final class EncodedInput {
        private final int[] ids;
        private final int[] attentionMask;

        private EncodedInput(int[] ids, int[] attentionMask) {
            this.ids = ids;
            this.attentionMask = attentionMask;
        }
    }

    /**
     * Explicit model ABI builder. All paths are supplied by the serving process,
     * never taken from an inference request.
     */
    public static final class Builder {
        private Path modelFile;
        private TokenizerType tokenizerType = TokenizerType.HUGGING_FACE;
        private Path tokenizerFile;
        private String tokenIdsInput = "input_ids";
        private String attentionMaskInput;
        private String tokenLengthsInput;
        private String waveformOutput = "waveform";
        private String confidenceOutput;
        private DataType tokenDataType = DataType.INT64;
        private boolean addSpecialTokens = true;
        private int sampleRateHz = 22_050;
        private int maxInputTokens = 2_048;
        private long maxOutputSamples = 6_615_000L;
        private String voice;
        private String language;
        private String modelId;
        private String modelVersion;
        private String configurationVersion;
        private double defaultConfidence = 1.0d;
        private Map<String, Object> configurationEvidence = Map.of();

        private Builder() {
        }

        public Builder modelFile(Path value) {
            this.modelFile = value;
            return this;
        }

        public Builder tokenizerType(TokenizerType value) {
            this.tokenizerType = value;
            return this;
        }

        public Builder tokenizerFile(Path value) {
            this.tokenizerFile = value;
            return this;
        }

        public Builder tokenIdsInput(String value) {
            this.tokenIdsInput = value;
            return this;
        }

        public Builder attentionMaskInput(String value) {
            this.attentionMaskInput = value;
            return this;
        }

        public Builder tokenLengthsInput(String value) {
            this.tokenLengthsInput = value;
            return this;
        }

        public Builder waveformOutput(String value) {
            this.waveformOutput = value;
            return this;
        }

        public Builder confidenceOutput(String value) {
            this.confidenceOutput = value;
            return this;
        }

        public Builder tokenDataType(DataType value) {
            this.tokenDataType = value;
            return this;
        }

        public Builder addSpecialTokens(boolean value) {
            this.addSpecialTokens = value;
            return this;
        }

        public Builder sampleRateHz(int value) {
            this.sampleRateHz = value;
            return this;
        }

        public Builder maxInputTokens(int value) {
            this.maxInputTokens = value;
            return this;
        }

        public Builder maxOutputSamples(long value) {
            this.maxOutputSamples = value;
            return this;
        }

        public Builder voice(String value) {
            this.voice = value;
            return this;
        }

        public Builder language(String value) {
            this.language = value;
            return this;
        }

        public Builder modelId(String value) {
            this.modelId = value;
            return this;
        }

        public Builder modelVersion(String value) {
            this.modelVersion = value;
            return this;
        }

        public Builder configurationVersion(String value) {
            this.configurationVersion = value;
            return this;
        }

        public Builder defaultConfidence(double value) {
            this.defaultConfidence = value;
            return this;
        }

        public Builder configurationEvidence(Map<String, Object> value) {
            this.configurationEvidence = value == null ? Map.of() : Map.copyOf(value);
            return this;
        }

        public SameDiffWaveformGenerator build() throws Exception {
            return new SameDiffWaveformGenerator(this);
        }

        private Builder validate() {
            modelFile = requireRegularFile(modelFile, "modelFile");
            tokenizerType = Objects.requireNonNull(tokenizerType, "tokenizerType");
            if (tokenizerType == TokenizerType.HUGGING_FACE) {
                tokenizerFile = requireRegularFile(tokenizerFile, "tokenizerFile");
            }
            tokenIdsInput = requireNonBlank(tokenIdsInput, "tokenIdsInput");
            waveformOutput = requireNonBlank(waveformOutput, "waveformOutput");
            tokenDataType = Objects.requireNonNull(tokenDataType, "tokenDataType");
            if (tokenDataType != DataType.INT32 && tokenDataType != DataType.INT64) {
                throw new IllegalArgumentException("tokenDataType must be INT32 or INT64");
            }
            if (sampleRateHz < 8_000 || sampleRateHz > 384_000) {
                throw new IllegalArgumentException("sampleRateHz is outside the supported range");
            }
            if (maxInputTokens <= 0) {
                throw new IllegalArgumentException("maxInputTokens must be positive");
            }
            if (maxOutputSamples <= 0) {
                throw new IllegalArgumentException("maxOutputSamples must be positive");
            }
            voice = requireNonBlank(voice, "voice");
            language = requireNonBlank(language, "language");
            modelId = requireNonBlank(modelId, "modelId");
            modelVersion = requireNonBlank(modelVersion, "modelVersion");
            configurationVersion = requireNonBlank(
                    configurationVersion, "configurationVersion");
            if (!Double.isFinite(defaultConfidence)
                    || defaultConfidence < 0.0d || defaultConfidence > 1.0d) {
                throw new IllegalArgumentException(
                        "defaultConfidence must be between 0 and 1");
            }
            return this;
        }

        private static Path requireRegularFile(Path path, String field) {
            Path normalized = Objects.requireNonNull(path, field)
                    .toAbsolutePath().normalize();
            if (!Files.isRegularFile(normalized, LinkOption.NOFOLLOW_LINKS)
                    || Files.isSymbolicLink(normalized)) {
                throw new IllegalArgumentException(field + " must be a regular file");
            }
            return normalized;
        }

        private static String requireNonBlank(String value, String field) {
            if (value == null || value.isBlank()) {
                throw new IllegalArgumentException(field + " must not be blank");
            }
            return value;
        }
    }
}
