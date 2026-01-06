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

package org.eclipse.deeplearning4j.vlm.input.ocr;

import org.eclipse.deeplearning4j.vlm.output.BoundingBox;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

/**
 * TrOCR (Transformer OCR) engine implementation.
 *
 * TrOCR is Microsoft's transformer-based OCR model that uses a Vision Transformer
 * encoder and a text transformer decoder. This implementation supports ONNX-exported
 * TrOCR models.
 *
 * <p>TrOCR is particularly good at:</p>
 * <ul>
 *   <li>Handwritten text recognition</li>
 *   <li>Printed text recognition</li>
 *   <li>Scene text recognition</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * TrOCREngine ocr = TrOCREngine.create(new File("trocr-base-handwritten"));
 * OCRResult result = ocr.recognize(new File("handwritten.png"));
 * System.out.println(result.getFullText());
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class TrOCREngine extends AbstractOCREngine {

    private static final String ENGINE_NAME = "TrOCR";
    private static final List<String> SUPPORTED_LANGUAGES = Arrays.asList("en");

    /**
     * Model variants available.
     */
    public enum ModelVariant {
        /**
         * TrOCR base model for printed text.
         */
        BASE_PRINTED,

        /**
         * TrOCR base model for handwritten text.
         */
        BASE_HANDWRITTEN,

        /**
         * TrOCR large model for printed text.
         */
        LARGE_PRINTED,

        /**
         * TrOCR large model for handwritten text.
         */
        LARGE_HANDWRITTEN,

        /**
         * TrOCR small model (faster, less accurate).
         */
        SMALL
    }

    private final File modelDirectory;
    private final ModelVariant variant;
    private SameDiff encoder;
    private SameDiff decoder;
    private Map<Integer, String> vocabulary;
    private Map<String, Integer> tokenToId;

    // Model parameters
    private int imageSize = 384;
    private int maxLength = 128;
    private int vocabSize = 50265; // GPT-2 vocab size
    private int padTokenId = 1;
    private int bosTokenId = 0;
    private int eosTokenId = 2;

    // Image normalization
    private float[] imageMean = {0.5f, 0.5f, 0.5f};
    private float[] imageStd = {0.5f, 0.5f, 0.5f};

    private TrOCREngine(File modelDirectory, ModelVariant variant) {
        super();
        this.modelDirectory = modelDirectory;
        this.variant = variant;
    }

    /**
     * Create a TrOCR engine from a model directory.
     *
     * @param modelDirectory directory containing ONNX model files
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static TrOCREngine create(File modelDirectory) throws OCRException {
        return create(modelDirectory, ModelVariant.BASE_PRINTED);
    }

    /**
     * Create a TrOCR engine with specific variant.
     *
     * @param modelDirectory directory containing ONNX model files
     * @param variant model variant
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static TrOCREngine create(File modelDirectory, ModelVariant variant) throws OCRException {
        TrOCREngine engine = new TrOCREngine(modelDirectory, variant);
        engine.initialize();
        return engine;
    }

    /**
     * Create a TrOCR engine with custom config.
     *
     * @param modelDirectory directory containing ONNX model files
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static TrOCREngine create(File modelDirectory, OCRConfig config) throws OCRException {
        TrOCREngine engine = new TrOCREngine(modelDirectory, ModelVariant.BASE_PRINTED);
        engine.setConfig(config);
        engine.initialize();
        return engine;
    }

    @Override
    protected void initialize() throws OCRException {
        try {
            OnnxFrameworkImporter importer = new OnnxFrameworkImporter();

            // Load encoder
            File encoderFile = new File(modelDirectory, "encoder_model.onnx");
            if (!encoderFile.exists()) {
                encoderFile = new File(modelDirectory, "vision_encoder.onnx");
            }
            if (encoderFile.exists()) {
                encoder = importer.runImport(encoderFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            } else {
                throw new OCRException(OCRException.ErrorCode.MODEL_NOT_FOUND, "Encoder model not found");
            }

            // Load decoder
            File decoderFile = new File(modelDirectory, "decoder_model.onnx");
            if (!decoderFile.exists()) {
                decoderFile = new File(modelDirectory, "decoder_model_merged.onnx");
            }
            if (decoderFile.exists()) {
                decoder = importer.runImport(decoderFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            } else {
                throw new OCRException(OCRException.ErrorCode.MODEL_NOT_FOUND, "Decoder model not found");
            }

            // Load vocabulary
            loadVocabulary();

            // Load config if present
            loadModelConfig();

            initialized = true;
        } catch (OCRException e) {
            throw e;
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to initialize TrOCR", e);
        }
    }

    private void loadVocabulary() throws IOException {
        vocabulary = new HashMap<>();
        tokenToId = new HashMap<>();

        // Try vocab.json first
        File vocabFile = new File(modelDirectory, "vocab.json");
        if (!vocabFile.exists()) {
            vocabFile = new File(modelDirectory, "tokenizer.json");
        }

        if (vocabFile.exists()) {
            String content = new String(Files.readAllBytes(vocabFile.toPath()));
            // Simple JSON parsing for vocab
            parseVocabulary(content);
        } else {
            // Use default GPT-2 style vocab
            initializeDefaultVocabulary();
        }
    }

    private void parseVocabulary(String json) {
        // Simplified vocabulary parsing
        // Full implementation would use a JSON library
        if (json.contains("\"vocab\"")) {
            // tokenizer.json format
            int vocabStart = json.indexOf("\"vocab\"");
            if (vocabStart >= 0) {
                int braceStart = json.indexOf('{', vocabStart);
                int braceEnd = findMatchingBrace(json, braceStart);
                if (braceStart >= 0 && braceEnd >= 0) {
                    parseVocabSection(json.substring(braceStart, braceEnd + 1));
                }
            }
        } else {
            // Direct vocab.json format
            parseVocabSection(json);
        }
    }

    private int findMatchingBrace(String s, int start) {
        int count = 0;
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) == '{') count++;
            else if (s.charAt(i) == '}') {
                count--;
                if (count == 0) return i;
            }
        }
        return -1;
    }

    private void parseVocabSection(String vocabJson) {
        // Parse "token": id pairs
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("\"([^\"]+)\":\\s*(\\d+)");
        java.util.regex.Matcher matcher = pattern.matcher(vocabJson);
        while (matcher.find()) {
            String token = matcher.group(1);
            int id = Integer.parseInt(matcher.group(2));
            vocabulary.put(id, token);
            tokenToId.put(token, id);
        }
    }

    private void initializeDefaultVocabulary() {
        // Basic ASCII vocabulary as fallback
        vocabulary.put(0, "<s>");
        vocabulary.put(1, "<pad>");
        vocabulary.put(2, "</s>");
        vocabulary.put(3, "<unk>");

        tokenToId.put("<s>", 0);
        tokenToId.put("<pad>", 1);
        tokenToId.put("</s>", 2);
        tokenToId.put("<unk>", 3);

        int id = 4;
        for (char c = ' '; c <= '~'; c++) {
            vocabulary.put(id, String.valueOf(c));
            tokenToId.put(String.valueOf(c), id);
            id++;
        }
    }

    private void loadModelConfig() {
        File configFile = new File(modelDirectory, "config.json");
        if (configFile.exists()) {
            try {
                String content = new String(Files.readAllBytes(configFile.toPath()));
                // Parse relevant config values
                if (content.contains("\"image_size\"")) {
                    imageSize = extractIntValue(content, "image_size", imageSize);
                }
                if (content.contains("\"max_length\"")) {
                    maxLength = extractIntValue(content, "max_length", maxLength);
                }
                if (content.contains("\"vocab_size\"")) {
                    vocabSize = extractIntValue(content, "vocab_size", vocabSize);
                }
            } catch (IOException e) {
                // Use defaults
            }
        }
    }

    private int extractIntValue(String json, String key, int defaultValue) {
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile("\"" + key + "\":\\s*(\\d+)");
        java.util.regex.Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        return defaultValue;
    }

    @Override
    public OCRResult recognize(File imageFile) throws IOException, OCRException {
        checkInitialized();
        long startTime = System.currentTimeMillis();

        BufferedImage image = loadImage(imageFile);
        image = preprocessImage(image);

        int origWidth = image.getWidth();
        int origHeight = image.getHeight();

        // TrOCR processes the entire image as one region
        INDArray imageArray = preprocessForModel(image);
        String recognizedText = runInference(imageArray);

        List<OCRTextRegion> regions = new ArrayList<>();
        if (recognizedText != null && !recognizedText.isEmpty()) {
            // Single region covering the whole image
            regions.add(OCRTextRegion.builder()
                    .text(recognizedText)
                    .confidence(0.9) // TrOCR doesn't provide per-token confidence by default
                    .boundingBox(new BoundingBox(0, 0, origWidth, origHeight))
                    .level(OCRTextRegion.TextLevel.BLOCK)
                    .build());
        }

        return OCRResult.builder()
                .regions(regions)
                .confidence(regions.isEmpty() ? 0.0 : 0.9)
                .processingTimeMs(System.currentTimeMillis() - startTime)
                .imageWidth(origWidth)
                .imageHeight(origHeight)
                .engineName(ENGINE_NAME)
                .build();
    }

    @Override
    public OCRResult recognize(INDArray image) throws OCRException {
        checkInitialized();
        long startTime = System.currentTimeMillis();

        long[] shape = image.shape();
        int height = (int) shape[0];
        int width = (int) shape[1];

        INDArray preprocessed = preprocessForModel(image);
        String recognizedText = runInference(preprocessed);

        List<OCRTextRegion> regions = new ArrayList<>();
        if (recognizedText != null && !recognizedText.isEmpty()) {
            regions.add(OCRTextRegion.builder()
                    .text(recognizedText)
                    .confidence(0.9)
                    .boundingBox(new BoundingBox(0, 0, width, height))
                    .level(OCRTextRegion.TextLevel.BLOCK)
                    .build());
        }

        return OCRResult.builder()
                .regions(regions)
                .confidence(regions.isEmpty() ? 0.0 : 0.9)
                .processingTimeMs(System.currentTimeMillis() - startTime)
                .imageWidth(width)
                .imageHeight(height)
                .engineName(ENGINE_NAME)
                .build();
    }

    private INDArray preprocessForModel(BufferedImage image) {
        // Resize to model input size
        BufferedImage resized = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(
                image.getScaledInstance(imageSize, imageSize, java.awt.Image.SCALE_SMOOTH),
                0, 0, null
        );

        // Convert to tensor [1, 3, H, W] and normalize
        float[] data = new float[3 * imageSize * imageSize];
        int idx = 0;

        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < imageSize; y++) {
                for (int x = 0; x < imageSize; x++) {
                    int rgb = resized.getRGB(x, y);
                    float value;
                    switch (c) {
                        case 0:
                            value = ((rgb >> 16) & 0xFF) / 255.0f;
                            break;
                        case 1:
                            value = ((rgb >> 8) & 0xFF) / 255.0f;
                            break;
                        default:
                            value = (rgb & 0xFF) / 255.0f;
                            break;
                    }
                    data[idx++] = (value - imageMean[c]) / imageStd[c];
                }
            }
        }

        return Nd4j.create(data, new int[]{1, 3, imageSize, imageSize});
    }

    private INDArray preprocessForModel(INDArray image) {
        // Assuming [H, W, C] format
        long[] shape = image.shape();
        if (shape.length == 3) {
            // Permute to [C, H, W] and add batch dimension
            image = image.permute(2, 0, 1).reshape(1, shape[2], shape[0], shape[1]);
        }

        // TODO: Add resize and normalization
        return image;
    }

    private String runInference(INDArray image) throws OCRException {
        try {
            // Run encoder
            encoder.associateArrayWithVariable(image, "pixel_values");
            INDArray encoderOutput = encoder.outputSingle(null, "last_hidden_state");

            // Autoregressive decoding
            List<Integer> generatedIds = new ArrayList<>();
            generatedIds.add(bosTokenId);

            for (int step = 0; step < maxLength; step++) {
                // Prepare decoder input
                int[] inputIds = generatedIds.stream().mapToInt(Integer::intValue).toArray();
                INDArray decoderInput = Nd4j.createFromArray(inputIds).reshape(1, inputIds.length);

                decoder.associateArrayWithVariable(decoderInput, "input_ids");
                decoder.associateArrayWithVariable(encoderOutput, "encoder_hidden_states");

                INDArray logits = decoder.outputSingle(null, "logits");

                // Get the last token's logits
                INDArray lastLogits = logits.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(logits.size(1) - 1),
                        NDArrayIndex.all()
                );

                // Argmax to get next token
                int nextToken = Nd4j.argMax(lastLogits).getInt(0);

                if (nextToken == eosTokenId) {
                    break;
                }

                generatedIds.add(nextToken);
            }

            // Decode tokens to text
            return decodeTokens(generatedIds);
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Inference failed", e);
        }
    }

    private String decodeTokens(List<Integer> tokenIds) {
        StringBuilder result = new StringBuilder();
        for (int id : tokenIds) {
            if (id == bosTokenId || id == eosTokenId || id == padTokenId) {
                continue;
            }
            String token = vocabulary.get(id);
            if (token != null) {
                // Handle GPT-2 style byte-level BPE
                if (token.startsWith("Ġ")) {
                    result.append(" ").append(token.substring(1));
                } else {
                    result.append(token);
                }
            }
        }
        return result.toString().trim();
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME + "-" + variant.name().toLowerCase().replace('_', '-');
    }

    @Override
    public List<String> getSupportedLanguages() {
        return SUPPORTED_LANGUAGES;
    }

    /**
     * Get the model variant.
     *
     * @return the model variant
     */
    public ModelVariant getVariant() {
        return variant;
    }

    @Override
    public void close() throws Exception {
        if (encoder != null) {
            encoder.close();
        }
        if (decoder != null) {
            decoder.close();
        }
        initialized = false;
    }
}
