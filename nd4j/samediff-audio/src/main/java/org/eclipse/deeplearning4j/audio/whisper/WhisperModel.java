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

package org.eclipse.deeplearning4j.audio.whisper;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.audio.feature.WhisperMelSpectrogram;
import org.eclipse.deeplearning4j.audio.io.AudioLoader;
import org.eclipse.deeplearning4j.audio.transform.AudioPreprocessor;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.StaticKvCacheDecodeLoop;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Main Whisper speech recognition model.
 * <p>
 * Uses the encoder-decoder path in {@link StaticKvCacheDecodeLoop} for efficient
 * autoregressive decoding with static KV cache and optional CUDA graph replay.
 * <p>
 * Usage:
 * <pre>
 * WhisperModel model = WhisperModel.fromOnnx(modelDir);
 * WhisperDecoderResult result = model.transcribe(new File("audio.wav"));
 * System.out.println(result.getText());
 * model.close();
 * </pre>
 */
@Slf4j
@Getter
public class WhisperModel implements Closeable {

    private final SameDiff encoder;
    private final SameDiff decoder;
    private final SameDiff decoderWithPast;
    private final WhisperTokenizer tokenizer;
    private final WhisperConfig config;
    private final WhisperMelSpectrogram melSpectrogram;

    private KvCacheStrategy kvCacheStrategy = KvCacheStrategy.STATIC;
    private SamplingConfig samplingConfig = SamplingConfig.greedy();
    private int maxTokens = 448;

    @Builder
    private WhisperModel(SameDiff encoder, SameDiff decoder, SameDiff decoderWithPast,
                         WhisperTokenizer tokenizer, WhisperConfig config,
                         WhisperMelSpectrogram melSpectrogram,
                         KvCacheStrategy kvCacheStrategy, SamplingConfig samplingConfig,
                         int maxTokens) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.decoderWithPast = decoderWithPast;
        this.tokenizer = tokenizer;
        this.config = config;
        this.melSpectrogram = melSpectrogram != null ? melSpectrogram : new WhisperMelSpectrogram(config);
        this.kvCacheStrategy = kvCacheStrategy != null ? kvCacheStrategy : KvCacheStrategy.STATIC;
        this.samplingConfig = samplingConfig != null ? samplingConfig : SamplingConfig.greedy();
        this.maxTokens = maxTokens > 0 ? maxTokens : 448;
    }

    /**
     * Load a Whisper model from an ONNX model directory.
     */
    public static WhisperModel fromOnnx(File modelDir) throws IOException {
        return fromOnnx(modelDir, null);
    }

    /**
     * Load a Whisper model from an ONNX model directory with a specific config.
     */
    public static WhisperModel fromOnnx(File modelDir, WhisperConfig config) throws IOException {
        File encoderFile = new File(modelDir, "encoder_model.onnx");
        File decoderFile = new File(modelDir, "decoder_model.onnx");
        File decoderWithPastFile = new File(modelDir, "decoder_with_past_model.onnx");
        File tokenizerFile = new File(modelDir, "tokenizer.json");

        if (!encoderFile.exists()) {
            throw new IOException("encoder_model.onnx not found in " + modelDir);
        }
        if (!decoderFile.exists()) {
            throw new IOException("decoder_model.onnx not found in " + modelDir);
        }

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();

        log.info("Loading Whisper encoder from {}...", encoderFile);
        SameDiff encoder = importer.runImport(encoderFile.getAbsolutePath(), Collections.emptyMap(), false, false);

        log.info("Loading Whisper decoder from {}...", decoderFile);
        SameDiff decoder = importer.runImport(decoderFile.getAbsolutePath(), Collections.emptyMap(), false, false);

        SameDiff decoderWithPast = null;
        if (decoderWithPastFile.exists()) {
            log.info("Loading Whisper decoder-with-past from {}...", decoderWithPastFile);
            decoderWithPast = importer.runImport(decoderWithPastFile.getAbsolutePath(), Collections.emptyMap(), false, false);
        }

        WhisperTokenizer tokenizer = null;
        if (tokenizerFile.exists()) {
            tokenizer = WhisperTokenizer.fromFile(tokenizerFile);
        } else {
            log.warn("tokenizer.json not found in {}. Tokenizer will not be available.", modelDir);
        }

        if (config == null) {
            config = detectConfig(modelDir);
        }

        return WhisperModel.builder()
                .encoder(encoder)
                .decoder(decoder)
                .decoderWithPast(decoderWithPast)
                .tokenizer(tokenizer)
                .config(config)
                .build();
    }

    /**
     * Load a Whisper model from a GGML file (whisper.cpp format).
     */
    public static WhisperModel fromGgml(File ggmlFile) throws IOException {
        return fromGgml(ggmlFile, null);
    }

    /**
     * Load a Whisper model from a GGML file with a specific config.
     */
    public static WhisperModel fromGgml(File ggmlFile, WhisperConfig config) throws IOException {
        log.info("Loading Whisper model from GGML file: {}", ggmlFile);

        SameDiff fullModel;
        try {
            GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(ConversionOptions.forInference());
            fullModel = converter.convert(ggmlFile);
        } catch (org.nd4j.ggml.GGMLImportException e) {
            throw new IOException("Failed to load GGML model: " + ggmlFile, e);
        }

        if (config == null) {
            config = detectConfigFromGgmlName(ggmlFile.getName());
        }

        return WhisperModel.builder()
                .encoder(fullModel)
                .decoder(fullModel)
                .config(config)
                .build();
    }

    /**
     * Download and load a Whisper model.
     */
    public static WhisperModel fromDownload(WhisperModelDownloader.WhisperModelSize size,
                                             WhisperModelDownloader.WhisperModelFormat format) throws IOException {
        WhisperModelDownloader downloader = new WhisperModelDownloader();
        WhisperModelDownloader.DownloadResult result = downloader.download(size, format);

        if (format == WhisperModelDownloader.WhisperModelFormat.ONNX) {
            return fromOnnx(result.getModelDir(), result.getConfig());
        } else {
            File ggmlFile = new File(result.getModelDir(),
                    "ggml-" + size.getHuggingFaceName() + ".bin");
            return fromGgml(ggmlFile, result.getConfig());
        }
    }

    /**
     * Transcribe an audio file.
     */
    public WhisperDecoderResult transcribe(File audioFile) throws IOException {
        return transcribe(audioFile, "en", "transcribe", false);
    }

    /**
     * Transcribe an audio file with language and task options.
     */
    public WhisperDecoderResult transcribe(File audioFile, String language,
                                            String task, boolean timestamps) throws IOException {
        INDArray melFeatures = melSpectrogram.extractFeaturesFromFile(audioFile);
        return transcribeMel(melFeatures, language, task, timestamps);
    }

    /**
     * Transcribe from raw audio samples.
     */
    public WhisperDecoderResult transcribe(INDArray audio, int sampleRate) {
        return transcribe(audio, sampleRate, "en", "transcribe", false);
    }

    /**
     * Transcribe from raw audio samples with options.
     */
    public WhisperDecoderResult transcribe(INDArray audio, int sampleRate,
                                            String language, String task, boolean timestamps) {
        if (sampleRate != config.getSampleRate()) {
            AudioPreprocessor preprocessor = AudioPreprocessor.builder()
                    .targetSampleRate(config.getSampleRate())
                    .normalize(false)
                    .applyPreEmphasis(false)
                    .build();
            audio = preprocessor.process(audio, sampleRate);
        }

        INDArray melFeatures = melSpectrogram.extractFeatures(audio);
        return transcribeMel(melFeatures, language, task, timestamps);
    }

    /**
     * Transcribe from pre-computed mel spectrogram features.
     */
    public WhisperDecoderResult transcribeMel(INDArray melFeatures) {
        return transcribeMel(melFeatures, "en", "transcribe", false);
    }

    /**
     * Transcribe from pre-computed mel spectrogram features using StaticKvCacheDecodeLoop.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>Run encoder on mel features → encoder hidden states</li>
     *   <li>Build prompt tokens (SOT + language + task)</li>
     *   <li>Build prefill embeddings from prompt tokens</li>
     *   <li>Create StaticKvCacheDecodeLoop with encoderDecoder=true, pass encoder outputs</li>
     *   <li>Decode → GenerationResult → WhisperDecoderResult</li>
     * </ol>
     */
    public WhisperDecoderResult transcribeMel(INDArray melFeatures, String language,
                                               String task, boolean timestamps) {
        // Step 1: Run encoder
        log.debug("Running Whisper encoder...");
        Map<String, INDArray> encoderPlaceholders = new HashMap<>();
        // Auto-detect encoder input name
        List<String> encoderInputNames = encoder.inputs();
        String encoderInputName = encoderInputNames.isEmpty() ? "input_features" : encoderInputNames.get(0);
        encoderPlaceholders.put(encoderInputName, melFeatures);

        // Find encoder output name (typically "last_hidden_state")
        String[] encoderOutputNames = encoder.outputs().toArray(new String[0]);
        Map<String, INDArray> encoderOutput = encoder.output(encoderPlaceholders, encoderOutputNames);
        INDArray encoderHidden = encoderOutput.values().iterator().next();
        log.debug("Encoder output shape: {}", Arrays.toString(encoderHidden.shape()));

        // Step 2: Create prompt tokens
        int[] promptTokens = tokenizer.createPromptTokens(language, task, timestamps);

        // Step 3: Build prefill embeddings from prompt tokens
        // For encoder-decoder models, the prompt tokens need to be embedded via the decoder's
        // token embedding table. We'll use input_ids mode for prefill.
        INDArray prefillInputIds = Nd4j.createFromArray(promptTokens)
                .reshape(1, promptTokens.length)
                .castTo(DataType.INT64);

        // Build prefill embeddings: embed prompt tokens via decoder's embedding
        // For the initial approach, we construct embeddings from the decoder's weight table
        INDArray prefillEmbeddings = buildPrefillEmbeddings(encoderHidden, prefillInputIds);

        // Step 4: Build decode loop with encoder-decoder support
        Set<Integer> stopTokens = new HashSet<>();
        stopTokens.add(WhisperTokenizer.EOT);

        // Discover decoder's embedding model (use decoder itself if no separate embed model)
        SameDiff embedModel = decoder; // Whisper decoder has internal embeddings

        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedModel)
                .tokenizer(tokenizer)
                .samplingConfig(samplingConfig)
                .maxNewTokens(maxTokens)
                .kvCacheStrategy(kvCacheStrategy)
                .additionalStopTokenIds(stopTokens)
                .hiddenSize(encoderHidden.size(2))
                .encoderDecoder(true)
                .encoderOutputs(encoderHidden)
                .ioConfig(ModelIOConfig.builder()
                        .encoderDecoder(true)
                        .encoderHiddenStatesName(findEncoderHiddenInputName())
                        .build())
                .build();

        // Step 5: Decode
        log.debug("Running Whisper decoder via StaticKvCacheDecodeLoop (encoder-decoder mode)...");
        GenerationResult genResult = loop.decode(prefillEmbeddings, promptTokens);

        // Step 6: Build result
        int[] decodedTokens = genResult.getTokenIds();
        String text = tokenizer.decodeSkippingSpecial(decodedTokens);

        WhisperDecoderResult.WhisperDecoderResultBuilder resultBuilder = WhisperDecoderResult.builder()
                .text(text.trim())
                .language(language != null ? language : detectLanguage(encoderHidden));

        if (timestamps && decodedTokens.length > 0) {
            resultBuilder.segments(parseTimestampSegments(decodedTokens));
        }

        return resultBuilder.build();
    }

    /**
     * Find the encoder hidden states input name in the decoder model.
     */
    private String findEncoderHiddenInputName() {
        for (String input : decoder.inputs()) {
            if (input.contains("encoder_hidden") || input.contains("encoder_output")) {
                return input;
            }
        }
        return "encoder_hidden_states";
    }

    /**
     * Build prefill embeddings from encoder output and prompt tokens.
     * Uses the decoder's own embedding weights for prompt tokens.
     */
    private INDArray buildPrefillEmbeddings(INDArray encoderHidden, INDArray promptInputIds) {
        // For Whisper ONNX, the decoder takes input_ids + encoder_hidden_states.
        // During prefill, we pass prompt token IDs to the decoder which embeds them internally.
        // The "prefill embeddings" here are a dummy that the decode loop uses for shape inference.
        // The actual embedding happens inside the decoder graph.
        //
        // Create a dummy embedding with the right shape: [1, promptLen, hiddenSize]
        long promptLen = promptInputIds.size(1);
        long hiddenSize = encoderHidden.size(2);
        return Nd4j.zeros(DataType.FLOAT, 1, promptLen, hiddenSize);
    }

    /**
     * Detect the language of the audio from encoder output.
     */
    private String detectLanguage(INDArray encoderOutput) {
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put(findEncoderHiddenInputName(), encoderOutput);
        placeholders.put("input_ids",
                Nd4j.createFromArray(new int[]{WhisperTokenizer.SOT})
                        .reshape(1, 1)
                        .castTo(DataType.INT64));

        Map<String, INDArray> output = decoder.output(placeholders, "logits");
        INDArray logits = output.get("logits");
        INDArray langLogits = logits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.interval(
                        WhisperTokenizer.LANGUAGE_TOKEN_START,
                        WhisperTokenizer.LANGUAGE_TOKEN_END + 1)
        );

        int langIdx = Nd4j.argMax(langLogits, -1).getInt(0);
        int langToken = WhisperTokenizer.LANGUAGE_TOKEN_START + langIdx;
        String detected = WhisperTokenizer.getLanguageCode(langToken);
        log.debug("Detected language: {}", detected);
        return detected != null ? detected : "en";
    }

    /**
     * Parse timestamp segments from decoded tokens.
     */
    private List<WhisperDecoderResult.Segment> parseTimestampSegments(int[] tokens) {
        List<WhisperDecoderResult.Segment> segments = new ArrayList<>();
        List<Integer> currentTokens = new ArrayList<>();
        double segmentStart = 0.0;

        for (int token : tokens) {
            if (WhisperTokenizer.isTimestampToken(token)) {
                double time = WhisperTokenizer.timestampToSeconds(token);

                if (currentTokens.isEmpty()) {
                    segmentStart = time;
                } else {
                    int[] segTokens = currentTokens.stream().mapToInt(Integer::intValue).toArray();
                    String segText = tokenizer.decodeSkippingSpecial(segTokens);

                    segments.add(WhisperDecoderResult.Segment.builder()
                            .start(segmentStart)
                            .end(time)
                            .text(segText.trim())
                            .tokens(segTokens)
                            .build());

                    currentTokens.clear();
                    segmentStart = time;
                }
            } else if (token != WhisperTokenizer.EOT) {
                currentTokens.add(token);
            }
        }

        if (!currentTokens.isEmpty()) {
            int[] segTokens = currentTokens.stream().mapToInt(Integer::intValue).toArray();
            String segText = tokenizer.decodeSkippingSpecial(segTokens);
            segments.add(WhisperDecoderResult.Segment.builder()
                    .start(segmentStart)
                    .end(30.0)
                    .text(segText.trim())
                    .tokens(segTokens)
                    .build());
        }

        return segments;
    }

    private static WhisperConfig detectConfig(File modelDir) {
        String name = modelDir.getName().toLowerCase();
        if (name.contains("turbo")) return WhisperConfig.turbo();
        if (name.contains("large-v3")) return WhisperConfig.largeV3();
        if (name.contains("large-v2")) return WhisperConfig.largeV2();
        if (name.contains("medium")) return WhisperConfig.medium();
        if (name.contains("small")) return WhisperConfig.small();
        if (name.contains("base")) return WhisperConfig.base();
        if (name.contains("tiny")) return WhisperConfig.tiny();
        log.warn("Could not detect Whisper model size from directory name: {}. Defaulting to base.", name);
        return WhisperConfig.base();
    }

    private static WhisperConfig detectConfigFromGgmlName(String fileName) {
        String name = fileName.toLowerCase();
        if (name.contains("turbo")) return WhisperConfig.turbo();
        if (name.contains("large-v3")) return WhisperConfig.largeV3();
        if (name.contains("large-v2")) return WhisperConfig.largeV2();
        if (name.contains("large")) return WhisperConfig.largeV2();
        if (name.contains("medium")) return WhisperConfig.medium();
        if (name.contains("small")) return WhisperConfig.small();
        if (name.contains("base")) return WhisperConfig.base();
        if (name.contains("tiny")) return WhisperConfig.tiny();
        log.warn("Could not detect Whisper model size from filename: {}. Defaulting to base.", fileName);
        return WhisperConfig.base();
    }

    @Override
    public void close() {
        if (encoder != null) {
            try { encoder.close(); } catch (Exception e) { log.debug("Error closing encoder", e); }
        }
        if (decoder != null && decoder != encoder) {
            try { decoder.close(); } catch (Exception e) { log.debug("Error closing decoder", e); }
        }
        if (decoderWithPast != null) {
            try { decoderWithPast.close(); } catch (Exception e) { log.debug("Error closing decoder-with-past", e); }
        }
    }
}
