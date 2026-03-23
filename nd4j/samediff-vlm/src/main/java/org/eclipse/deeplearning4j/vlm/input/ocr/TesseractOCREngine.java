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
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tesseract OCR engine implementation.
 *
 * This implementation wraps the Tesseract command-line tool, providing
 * a Java interface to the powerful open-source OCR engine.
 *
 * <p>Prerequisites:</p>
 * <ul>
 *   <li>Tesseract must be installed and available in PATH</li>
 *   <li>Language data files must be installed</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * TesseractOCREngine ocr = TesseractOCREngine.create();
 * ocr.setLanguages("eng", "fra");
 * OCRResult result = ocr.recognize(new File("document.png"));
 * System.out.println(result.getFullText());
 * }</pre>
 *
 * Adam Gibson
 */
public class TesseractOCREngine extends AbstractOCREngine {

    private static final String ENGINE_NAME = "Tesseract";
    private static final List<String> SUPPORTED_LANGUAGES = Arrays.asList(
            "eng", "fra", "deu", "spa", "ita", "por", "nld", "pol", "rus",
            "chi_sim", "chi_tra", "jpn", "kor", "ara", "hin", "vie", "tha"
    );

    private String tesseractPath;
    private String dataPath;

    private TesseractOCREngine() {
        super();
        this.tesseractPath = "tesseract"; // Assume in PATH
    }

    /**
     * Create a Tesseract OCR engine with default settings.
     *
     * @return the OCR engine
     * @throws OCRException if Tesseract is not available
     */
    public static TesseractOCREngine create() throws OCRException {
        TesseractOCREngine engine = new TesseractOCREngine();
        engine.initialize();
        return engine;
    }

    /**
     * Create a Tesseract OCR engine with custom path.
     *
     * @param tesseractPath path to tesseract executable
     * @return the OCR engine
     * @throws OCRException if initialization fails
     */
    public static TesseractOCREngine create(String tesseractPath) throws OCRException {
        TesseractOCREngine engine = new TesseractOCREngine();
        engine.tesseractPath = tesseractPath;
        engine.initialize();
        return engine;
    }

    /**
     * Create a Tesseract OCR engine with custom config.
     *
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if initialization fails
     */
    public static TesseractOCREngine create(OCRConfig config) throws OCRException {
        TesseractOCREngine engine = new TesseractOCREngine();
        engine.setConfig(config);
        if (config.getModelPath() != null) {
            engine.dataPath = config.getModelPath();
        }
        engine.initialize();
        return engine;
    }

    @Override
    protected void initialize() throws OCRException {
        // Check if tesseract is available
        try {
            ProcessBuilder pb = new ProcessBuilder(tesseractPath, "--version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                        "Tesseract not found. Please install Tesseract OCR.");
            }
            initialized = true;
        } catch (IOException | InterruptedException e) {
            throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                    "Failed to initialize Tesseract: " + e.getMessage(), e);
        }
    }

    @Override
    public OCRResult recognize(File imageFile) throws IOException, OCRException {
        checkInitialized();
        long startTime = System.currentTimeMillis();

        BufferedImage image = loadImage(imageFile);
        int origWidth = image.getWidth();
        int origHeight = image.getHeight();

        // Preprocess if needed
        File processedFile = imageFile;
        if (config.isPreprocessImages()) {
            image = preprocessImage(image);
            processedFile = File.createTempFile("tesseract_", ".png");
            processedFile.deleteOnExit();
            ImageIO.write(image, "PNG", processedFile);
        }

        // Build command
        List<String> command = buildCommand(processedFile, OutputFormat.TSV);

        // Execute
        String output = executeCommand(command);

        // Parse results
        List<OCRTextRegion> regions = parseTsvOutput(output, origWidth, origHeight);

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

        // Convert INDArray to BufferedImage and save to temp file
        long[] shape = image.shape();
        int height = (int) shape[0];
        int width = (int) shape[1];
        int channels = shape.length > 2 ? (int) shape[2] : 1;

        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r, g, b;
                if (channels == 3) {
                    r = (int) (image.getDouble(y, x, 0) * 255);
                    g = (int) (image.getDouble(y, x, 1) * 255);
                    b = (int) (image.getDouble(y, x, 2) * 255);
                } else {
                    int gray = (int) (image.getDouble(y, x) * 255);
                    r = g = b = gray;
                }
                bufferedImage.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }

        try {
            File tempFile = File.createTempFile("tesseract_", ".png");
            tempFile.deleteOnExit();
            ImageIO.write(bufferedImage, "PNG", tempFile);
            return recognize(tempFile);
        } catch (IOException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to process image", e);
        }
    }

    private List<String> buildCommand(File imageFile, OutputFormat format) {
        List<String> command = new ArrayList<>();
        command.add(tesseractPath);
        command.add(imageFile.getAbsolutePath());
        command.add("stdout");

        // Language
        if (languages != null && languages.length > 0) {
            command.add("-l");
            command.add(String.join("+", languages));
        }

        // Data path
        if (dataPath != null) {
            command.add("--tessdata-dir");
            command.add(dataPath);
        }

        // Page segmentation mode
        command.add("--psm");
        command.add(String.valueOf(getPageSegMode()));

        // OEM mode
        command.add("--oem");
        command.add(String.valueOf(getOemMode()));

        // Output format
        switch (format) {
            case TSV:
                command.add("-c");
                command.add("tessedit_create_tsv=1");
                break;
            case HOCR:
                command.add("-c");
                command.add("tessedit_create_hocr=1");
                break;
            case ALTO:
                command.add("-c");
                command.add("tessedit_create_alto=1");
                break;
            default:
                // Plain text
                break;
        }

        return command;
    }

    private int getPageSegMode() {
        switch (config.getPageSegMode()) {
            case OSD_ONLY:
                return 0;
            case AUTO_OSD:
                return 1;
            case AUTO:
                return 3;
            case SINGLE_COLUMN:
                return 4;
            case SINGLE_BLOCK:
                return 6;
            case SINGLE_LINE:
                return 7;
            case SINGLE_WORD:
                return 8;
            case SINGLE_CHAR:
                return 10;
            case SPARSE_TEXT:
                return 11;
            case RAW_LINE:
                return 13;
            default:
                return 3;
        }
    }

    private int getOemMode() {
        switch (config.getEngineMode()) {
            case LEGACY:
                return 0;
            case NEURAL:
                return 1;
            case FAST:
                return 2;
            default:
                return 3; // Default (combine legacy and LSTM)
        }
    }

    private String executeCommand(List<String> command) throws OCRException {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                        "Tesseract failed with exit code " + exitCode + ": " + output);
            }

            return output.toString();
        } catch (IOException | InterruptedException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                    "Failed to execute Tesseract: " + e.getMessage(), e);
        }
    }

    private List<OCRTextRegion> parseTsvOutput(String output, int imgWidth, int imgHeight) {
        List<OCRTextRegion> regions = new ArrayList<>();

        String[] lines = output.split("\n");
        if (lines.length < 2) {
            return regions;
        }

        // Skip header line
        for (int i = 1; i < lines.length; i++) {
            String[] parts = lines[i].split("\t");
            if (parts.length < 12) {
                continue;
            }

            try {
                int level = Integer.parseInt(parts[0]);
                int left = Integer.parseInt(parts[6]);
                int top = Integer.parseInt(parts[7]);
                int width = Integer.parseInt(parts[8]);
                int height = Integer.parseInt(parts[9]);
                float conf = Float.parseFloat(parts[10]);
                String text = parts[11];

                // Only include words (level 5) with actual text
                if (level == 5 && text != null && !text.trim().isEmpty() && conf >= 0) {
                    BoundingBox box = new BoundingBox(left, top, left + width, top + height);
                    OCRTextRegion.TextLevel textLevel;
                    switch (level) {
                        case 2:
                            textLevel = OCRTextRegion.TextLevel.PARAGRAPH;
                            break;
                        case 3:
                            textLevel = OCRTextRegion.TextLevel.BLOCK;
                            break;
                        case 4:
                            textLevel = OCRTextRegion.TextLevel.LINE;
                            break;
                        case 5:
                            textLevel = OCRTextRegion.TextLevel.WORD;
                            break;
                        default:
                            textLevel = OCRTextRegion.TextLevel.WORD;
                    }

                    regions.add(OCRTextRegion.builder()
                            .text(text.trim())
                            .confidence(conf / 100.0) // Tesseract returns 0-100
                            .boundingBox(box)
                            .level(textLevel)
                            .build());
                }
            } catch (NumberFormatException e) {
                // Skip malformed lines
            }
        }

        return regions;
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public List<String> getSupportedLanguages() {
        // Could dynamically query tesseract --list-langs
        return SUPPORTED_LANGUAGES;
    }

    /**
     * Get list of installed languages.
     *
     * @return list of installed language codes
     * @throws OCRException if query fails
     */
    public List<String> getInstalledLanguages() throws OCRException {
        try {
            List<String> command = new ArrayList<>();
            command.add(tesseractPath);
            command.add("--list-langs");
            if (dataPath != null) {
                command.add("--tessdata-dir");
                command.add(dataPath);
            }

            String output = executeCommand(command);
            List<String> langs = new ArrayList<>();
            for (String line : output.split("\n")) {
                line = line.trim();
                if (!line.isEmpty() && !line.contains("List of available languages")) {
                    langs.add(line);
                }
            }
            return langs;
        } catch (Exception e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                    "Failed to get installed languages", e);
        }
    }

    @Override
    public void close() throws Exception {
        initialized = false;
    }

    private enum OutputFormat {
        TEXT,
        TSV,
        HOCR,
        ALTO
    }
}
