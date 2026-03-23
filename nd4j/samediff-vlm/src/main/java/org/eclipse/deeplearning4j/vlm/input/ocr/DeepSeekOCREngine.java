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
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * DeepSeek OCR engine implementation.
 *
 * DeepSeek OCR is a vision-language model optimized for text recognition
 * in documents. This implementation supports loading ONNX-exported models.
 *
 * <p>Usage:</p>
 * <pre>{@code
 * DeepSeekOCREngine ocr = DeepSeekOCREngine.create(new File("deepseek-ocr"));
 * OCRResult result = ocr.recognize(new File("document.png"));
 * System.out.println(result.getFullText());
 * }</pre>
 *
 * Adam Gibson
 */
public class DeepSeekOCREngine extends AbstractOCREngine {

    private static final String ENGINE_NAME = "DeepSeek-OCR";
    private static final List<String> SUPPORTED_LANGUAGES = Arrays.asList(
            "en", "zh", "ja", "ko", "ar", "hi", "ru", "de", "fr", "es", "pt", "it"
    );

    private final File modelDirectory;
    private SameDiff visionEncoder;
    private SameDiff textDecoder;
    private int imageSize = 1024;
    private float[] imageMean = {0.48145466f, 0.4578275f, 0.40821073f};
    private float[] imageStd = {0.26862954f, 0.26130258f, 0.27577711f};

    private DeepSeekOCREngine(File modelDirectory) {
        super();
        this.modelDirectory = modelDirectory;
    }

    /**
     * Create a DeepSeek OCR engine from a model directory.
     *
     * @param modelDirectory directory containing ONNX model files
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static DeepSeekOCREngine create(File modelDirectory) throws OCRException {
        DeepSeekOCREngine engine = new DeepSeekOCREngine(modelDirectory);
        engine.initialize();
        return engine;
    }

    /**
     * Create a DeepSeek OCR engine with custom config.
     *
     * @param modelDirectory directory containing ONNX model files
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static DeepSeekOCREngine create(File modelDirectory, OCRConfig config) throws OCRException {
        DeepSeekOCREngine engine = new DeepSeekOCREngine(modelDirectory);
        engine.setConfig(config);
        engine.initialize();
        return engine;
    }

    @Override
    protected void initialize() throws OCRException {
        try {
            OnnxFrameworkImporter importer = new OnnxFrameworkImporter();

            // Load vision encoder
            File visionEncoderFile = new File(modelDirectory, "vision_encoder.onnx");
            if (visionEncoderFile.exists()) {
                visionEncoder = importer.runImport(visionEncoderFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            }

            // Load text decoder
            File textDecoderFile = new File(modelDirectory, "text_decoder.onnx");
            if (!textDecoderFile.exists()) {
                textDecoderFile = new File(modelDirectory, "decoder.onnx");
            }
            if (textDecoderFile.exists()) {
                textDecoder = importer.runImport(textDecoderFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            }

            // Load config if present
            File configFile = new File(modelDirectory, "config.json");
            if (configFile.exists()) {
                loadConfig(configFile);
            }

            initialized = true;
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to initialize DeepSeek OCR", e);
        }
    }

    private void loadConfig(File configFile) {
        // Parse config.json for image size and normalization params
        // This is a simplified version - full implementation would parse JSON
    }

    @Override
    public OCRResult recognize(File imageFile) throws IOException, OCRException {
        checkInitialized();
        long startTime = System.currentTimeMillis();

        BufferedImage image = loadImage(imageFile);
        image = preprocessImage(image);

        INDArray imageArray = preprocessForModel(image);
        List<OCRTextRegion> regions = runInference(imageArray, image.getWidth(), image.getHeight());

        double avgConfidence = regions.stream()
                .mapToDouble(OCRTextRegion::getConfidence)
                .average()
                .orElse(0.0);

        return OCRResult.builder()
                .regions(regions)
                .confidence(avgConfidence)
                .processingTimeMs(System.currentTimeMillis() - startTime)
                .imageWidth(image.getWidth())
                .imageHeight(image.getHeight())
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
        List<OCRTextRegion> regions = runInference(preprocessed, width, height);

        double avgConfidence = regions.stream()
                .mapToDouble(OCRTextRegion::getConfidence)
                .average()
                .orElse(0.0);

        return OCRResult.builder()
                .regions(regions)
                .confidence(avgConfidence)
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

        // Convert to tensor and normalize
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
        // Resize and normalize INDArray input
        // Assuming [H, W, C] format
        long[] shape = image.shape();
        if (shape.length == 3) {
            // Add batch dimension and transpose to [N, C, H, W]
            image = image.permute(2, 0, 1).reshape(1, shape[2], shape[0], shape[1]);
        }

        // TODO: Add resize and normalization
        return image;
    }

    private List<OCRTextRegion> runInference(INDArray image, int origWidth, int origHeight) throws OCRException {
        List<OCRTextRegion> regions = new ArrayList<>();

        try {
            if (visionEncoder != null && textDecoder != null) {
                // Run vision encoder
                visionEncoder.associateArrayWithVariable(image, "input");
                INDArray features = visionEncoder.outputSingle(null, "output");

                // Run text decoder (this is simplified - real impl would do autoregressive decoding)
                textDecoder.associateArrayWithVariable(features, "encoder_hidden_states");
                INDArray outputs = textDecoder.outputSingle(null, "logits");

                // Parse outputs to text regions
                regions = parseOutputs(outputs, origWidth, origHeight);
            }
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Inference failed", e);
        }

        return regions;
    }

    private List<OCRTextRegion> parseOutputs(INDArray outputs, int origWidth, int origHeight) {
        List<OCRTextRegion> regions = new ArrayList<>();

        // This is a simplified parser - real implementation would decode
        // the model's output format (token IDs, bounding boxes, etc.)

        // Placeholder: Return empty list if no real outputs
        return regions;
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public List<String> getSupportedLanguages() {
        return SUPPORTED_LANGUAGES;
    }

    @Override
    public void close() throws Exception {
        if (visionEncoder != null) {
            visionEncoder.close();
        }
        if (textDecoder != null) {
            textDecoder.close();
        }
        initialized = false;
    }
}
