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
 * PaddleOCR engine implementation.
 *
 * PaddleOCR is a popular open-source OCR system from Baidu that provides
 * accurate text detection and recognition. This implementation supports
 * ONNX-exported PaddleOCR models (PP-OCR series).
 *
 * <p>The pipeline consists of:</p>
 * <ul>
 *   <li>Detection model (DB/EAST) - finds text regions</li>
 *   <li>Direction classifier (optional) - determines text orientation</li>
 *   <li>Recognition model (CRNN) - recognizes text in regions</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * PaddleOCREngine ocr = PaddleOCREngine.create(new File("paddleocr-models"));
 * OCRResult result = ocr.recognize(new File("document.png"));
 * for (OCRTextRegion region : result.getRegions()) {
 *     System.out.println(region.getText() + " @ " + region.getBoundingBox());
 * }
 * }</pre>
 *
 * Adam Gibson
 */
public class PaddleOCREngine extends AbstractOCREngine {

    private static final String ENGINE_NAME = "PaddleOCR";
    private static final List<String> SUPPORTED_LANGUAGES = Arrays.asList(
            "en", "ch", "chinese_cht", "japan", "korean", "french", "german",
            "arabic", "cyrillic", "devanagari", "latin"
    );

    private final File modelDirectory;
    private SameDiff detectionModel;
    private SameDiff classifierModel;
    private SameDiff recognitionModel;
    private List<String> vocabulary;

    // Model parameters
    private int detInputHeight = 960;
    private int detInputWidth = 960;
    private int recInputHeight = 48;
    private int recInputWidth = 320;
    private float detThreshold = 0.3f;
    private float boxThreshold = 0.6f;
    private float unclipRatio = 1.5f;

    private PaddleOCREngine(File modelDirectory) {
        super();
        this.modelDirectory = modelDirectory;
    }

    /**
     * Create a PaddleOCR engine from a model directory.
     *
     * @param modelDirectory directory containing ONNX model files
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static PaddleOCREngine create(File modelDirectory) throws OCRException {
        PaddleOCREngine engine = new PaddleOCREngine(modelDirectory);
        engine.initialize();
        return engine;
    }

    /**
     * Create a PaddleOCR engine with custom config.
     *
     * @param modelDirectory directory containing ONNX model files
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if loading fails
     */
    public static PaddleOCREngine create(File modelDirectory, OCRConfig config) throws OCRException {
        PaddleOCREngine engine = new PaddleOCREngine(modelDirectory);
        engine.setConfig(config);
        engine.initialize();
        return engine;
    }

    @Override
    protected void initialize() throws OCRException {
        try {
            OnnxFrameworkImporter importer = new OnnxFrameworkImporter();

            // Load detection model
            File detFile = findModel("det", "detection");
            if (detFile != null) {
                detectionModel = importer.runImport(detFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            } else {
                throw new OCRException(OCRException.ErrorCode.MODEL_NOT_FOUND, "Detection model not found");
            }

            // Load classifier model (optional)
            File clsFile = findModel("cls", "classifier");
            if (clsFile != null) {
                classifierModel = importer.runImport(clsFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            }

            // Load recognition model
            File recFile = findModel("rec", "recognition");
            if (recFile != null) {
                recognitionModel = importer.runImport(recFile.getAbsolutePath(), java.util.Collections.emptyMap(), false, false);
            } else {
                throw new OCRException(OCRException.ErrorCode.MODEL_NOT_FOUND, "Recognition model not found");
            }

            // Load vocabulary
            loadVocabulary();

            initialized = true;
        } catch (OCRException e) {
            throw e;
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to initialize PaddleOCR", e);
        }
    }

    private File findModel(String... prefixes) {
        for (String prefix : prefixes) {
            File[] matches = modelDirectory.listFiles((dir, name) ->
                    name.toLowerCase().contains(prefix) && name.endsWith(".onnx")
            );
            if (matches != null && matches.length > 0) {
                return matches[0];
            }
        }
        return null;
    }

    private void loadVocabulary() {
        vocabulary = new ArrayList<>();
        // Try to load vocabulary file
        File vocabFile = new File(modelDirectory, "ppocr_keys_v1.txt");
        if (!vocabFile.exists()) {
            vocabFile = new File(modelDirectory, "vocab.txt");
        }
        if (!vocabFile.exists()) {
            vocabFile = new File(modelDirectory, "dict.txt");
        }

        if (vocabFile.exists()) {
            try {
                vocabulary.addAll(Files.readAllLines(vocabFile.toPath()));
            } catch (IOException e) {
                // Use default character set
                initDefaultVocabulary();
            }
        } else {
            initDefaultVocabulary();
        }
    }

    private void initDefaultVocabulary() {
        // Basic ASCII vocabulary
        vocabulary.add("<blank>");
        for (char c = ' '; c <= '~'; c++) {
            vocabulary.add(String.valueOf(c));
        }
    }

    @Override
    public OCRResult recognize(File imageFile) throws IOException, OCRException {
        checkInitialized();
        long startTime = System.currentTimeMillis();

        BufferedImage image = loadImage(imageFile);
        image = preprocessImage(image);

        int origWidth = image.getWidth();
        int origHeight = image.getHeight();

        // Step 1: Text Detection
        List<float[][]> boxes = detectText(image);

        // Step 2: Crop and recognize each region
        List<OCRTextRegion> regions = new ArrayList<>();
        for (float[][] box : boxes) {
            try {
                // Crop region
                BufferedImage cropped = cropRegion(image, box);
                if (cropped.getWidth() < 8 || cropped.getHeight() < 8) {
                    continue;
                }

                // Optional: Classify direction
                boolean rotated = false;
                if (classifierModel != null) {
                    rotated = classifyDirection(cropped);
                    if (rotated) {
                        cropped = rotateImage(cropped, 180);
                    }
                }

                // Recognize text
                RecognitionResult recResult = recognizeText(cropped);

                if (recResult.confidence >= config.getMinConfidence()) {
                    BoundingBox boundingBox = boxToBoundingBox(box, origWidth, origHeight);
                    regions.add(OCRTextRegion.builder()
                            .text(recResult.text)
                            .confidence(recResult.confidence)
                            .boundingBox(boundingBox)
                            .level(OCRTextRegion.TextLevel.LINE)
                            .build());
                }
            } catch (Exception e) {
                // Skip this region on error
            }
        }

        double avgConfidence = regions.stream()
                .mapToDouble(OCRTextRegion::getConfidence)
                .average()
                .orElse(0.0);

        return OCRResult.builder()
                .regions(regions)
                .confidence(avgConfidence)
                .processingTimeMs(System.currentTimeMillis() - startTime)
                .imageWidth(origWidth)
                .imageHeight(origHeight)
                .engineName(ENGINE_NAME)
                .build();
    }

    @Override
    public OCRResult recognize(INDArray image) throws OCRException {
        checkInitialized();
        // Convert INDArray to BufferedImage and process
        // This is a simplified implementation
        long[] shape = image.shape();
        int height = (int) shape[0];
        int width = (int) shape[1];

        BufferedImage bufferedImage = arrayToImage(image);

        try {
            File tempFile = File.createTempFile("ocr_", ".png");
            tempFile.deleteOnExit();
            javax.imageio.ImageIO.write(bufferedImage, "PNG", tempFile);
            return recognize(tempFile);
        } catch (IOException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to process image", e);
        }
    }

    private BufferedImage arrayToImage(INDArray array) {
        long[] shape = array.shape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        int channels = shape.length > 2 ? (int) shape[2] : 1;

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r, g, b;
                if (channels == 3) {
                    r = (int) (array.getDouble(y, x, 0) * 255);
                    g = (int) (array.getDouble(y, x, 1) * 255);
                    b = (int) (array.getDouble(y, x, 2) * 255);
                } else {
                    int gray = (int) (array.getDouble(y, x) * 255);
                    r = g = b = gray;
                }
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
    }

    private List<float[][]> detectText(BufferedImage image) throws OCRException {
        // Preprocess for detection
        float ratio = Math.min((float) detInputHeight / image.getHeight(),
                (float) detInputWidth / image.getWidth());
        int resizedHeight = Math.round(image.getHeight() * ratio);
        int resizedWidth = Math.round(image.getWidth() * ratio);

        // Make divisible by 32
        resizedHeight = ((resizedHeight + 31) / 32) * 32;
        resizedWidth = ((resizedWidth + 31) / 32) * 32;

        BufferedImage resized = new BufferedImage(resizedWidth, resizedHeight, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(
                image.getScaledInstance(resizedWidth, resizedHeight, java.awt.Image.SCALE_SMOOTH),
                0, 0, null
        );

        INDArray input = prepareDetInput(resized);

        try {
            detectionModel.associateArrayWithVariable(input, "x");
            INDArray output = detectionModel.outputSingle(null, "sigmoid_0.tmp_0");

            return postProcessDetection(output, image.getWidth(), image.getHeight(), ratio);
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Detection failed", e);
        }
    }

    private INDArray prepareDetInput(BufferedImage image) {
        float[] data = new float[3 * image.getHeight() * image.getWidth()];
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int rgb = image.getRGB(x, y);
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
                    data[idx++] = (value - mean[c]) / std[c];
                }
            }
        }

        return Nd4j.create(data, new int[]{1, 3, image.getHeight(), image.getWidth()});
    }

    private List<float[][]> postProcessDetection(INDArray output, int origWidth, int origHeight, float ratio) {
        // DB post-processing: threshold, find contours, create boxes
        List<float[][]> boxes = new ArrayList<>();

        // Simplified: This would normally use contour finding
        // For now, return placeholder

        return boxes;
    }

    private BufferedImage cropRegion(BufferedImage image, float[][] box) {
        // Find bounding rectangle of polygon
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;

        for (float[] point : box) {
            minX = Math.min(minX, point[0]);
            minY = Math.min(minY, point[1]);
            maxX = Math.max(maxX, point[0]);
            maxY = Math.max(maxY, point[1]);
        }

        int x = Math.max(0, (int) minX);
        int y = Math.max(0, (int) minY);
        int width = Math.min(image.getWidth() - x, (int) (maxX - minX));
        int height = Math.min(image.getHeight() - y, (int) (maxY - minY));

        return image.getSubimage(x, y, width, height);
    }

    private boolean classifyDirection(BufferedImage image) {
        // Resize for classifier
        int clsHeight = 48;
        int clsWidth = 192;
        BufferedImage resized = new BufferedImage(clsWidth, clsHeight, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(
                image.getScaledInstance(clsWidth, clsHeight, java.awt.Image.SCALE_SMOOTH),
                0, 0, null
        );

        INDArray input = prepareClsInput(resized);

        try {
            classifierModel.associateArrayWithVariable(input, "x");
            INDArray output = classifierModel.outputSingle(null, "softmax_0.tmp_0");
            return output.getDouble(0, 1) > 0.5;
        } catch (Exception e) {
            return false;
        }
    }

    private INDArray prepareClsInput(BufferedImage image) {
        float[] data = new float[3 * image.getHeight() * image.getWidth()];
        int idx = 0;

        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int rgb = image.getRGB(x, y);
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
                    data[idx++] = (value - 0.5f) / 0.5f;
                }
            }
        }

        return Nd4j.create(data, new int[]{1, 3, image.getHeight(), image.getWidth()});
    }

    private BufferedImage rotateImage(BufferedImage image, int degrees) {
        if (degrees == 180) {
            BufferedImage rotated = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    rotated.setRGB(image.getWidth() - 1 - x, image.getHeight() - 1 - y, image.getRGB(x, y));
                }
            }
            return rotated;
        }
        return image;
    }

    private RecognitionResult recognizeText(BufferedImage image) throws OCRException {
        // Resize for recognition
        float ratio = (float) recInputHeight / image.getHeight();
        int resizedWidth = Math.min(recInputWidth, Math.round(image.getWidth() * ratio));

        BufferedImage resized = new BufferedImage(resizedWidth, recInputHeight, BufferedImage.TYPE_INT_RGB);
        resized.getGraphics().drawImage(
                image.getScaledInstance(resizedWidth, recInputHeight, java.awt.Image.SCALE_SMOOTH),
                0, 0, null
        );

        // Pad to fixed width
        BufferedImage padded = new BufferedImage(recInputWidth, recInputHeight, BufferedImage.TYPE_INT_RGB);
        padded.getGraphics().drawImage(resized, 0, 0, null);

        INDArray input = prepareRecInput(padded);

        try {
            recognitionModel.associateArrayWithVariable(input, "x");
            INDArray output = recognitionModel.outputSingle(null, "softmax_0.tmp_0");

            return decodeRecOutput(output);
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Recognition failed", e);
        }
    }

    private INDArray prepareRecInput(BufferedImage image) {
        float[] data = new float[3 * image.getHeight() * image.getWidth()];
        int idx = 0;

        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    int rgb = image.getRGB(x, y);
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
                    data[idx++] = (value - 0.5f) / 0.5f;
                }
            }
        }

        return Nd4j.create(data, new int[]{1, 3, image.getHeight(), image.getWidth()});
    }

    private RecognitionResult decodeRecOutput(INDArray output) {
        // CTC decoding
        StringBuilder text = new StringBuilder();
        double totalConfidence = 0;
        int count = 0;
        int lastIdx = -1;

        long seqLen = output.shape()[1];
        for (int t = 0; t < seqLen; t++) {
            INDArray probs = output.get(NDArrayIndex.point(0), NDArrayIndex.point(t), NDArrayIndex.all());
            int idx = Nd4j.argMax(probs).getInt(0);
            double conf = probs.getDouble(idx);

            if (idx != 0 && idx != lastIdx) { // 0 is blank
                if (idx < vocabulary.size()) {
                    text.append(vocabulary.get(idx));
                    totalConfidence += conf;
                    count++;
                }
            }
            lastIdx = idx;
        }

        return new RecognitionResult(
                text.toString(),
                count > 0 ? totalConfidence / count : 0.0
        );
    }

    private BoundingBox boxToBoundingBox(float[][] box, int imgWidth, int imgHeight) {
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;

        for (float[] point : box) {
            minX = Math.min(minX, point[0]);
            minY = Math.min(minY, point[1]);
            maxX = Math.max(maxX, point[0]);
            maxY = Math.max(maxY, point[1]);
        }

        return new BoundingBox(
                (int) minX,
                (int) minY,
                (int) maxX,
                (int) maxY
        );
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
        if (detectionModel != null) {
            detectionModel.close();
        }
        if (classifierModel != null) {
            classifierModel.close();
        }
        if (recognitionModel != null) {
            recognitionModel.close();
        }
        initialized = false;
    }

    private static class RecognitionResult {
        final String text;
        final double confidence;

        RecognitionResult(String text, double confidence) {
            this.text = text;
            this.confidence = confidence;
        }
    }
}
