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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * EasyOCR engine implementation.
 *
 * EasyOCR is a Python-based OCR library that supports 80+ languages.
 * This implementation wraps EasyOCR via command-line interface.
 *
 * <p>Prerequisites:</p>
 * <ul>
 *   <li>Python 3.x must be installed</li>
 *   <li>EasyOCR must be installed: pip install easyocr</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * EasyOCREngine ocr = EasyOCREngine.create();
 * ocr.setLanguages("en", "ch_sim");
 * OCRResult result = ocr.recognize(new File("document.png"));
 * System.out.println(result.getFullText());
 * }</pre>
 *
 * Adam Gibson
 */
public class EasyOCREngine extends AbstractOCREngine {

    private static final String ENGINE_NAME = "EasyOCR";
    private static final List<String> SUPPORTED_LANGUAGES = Arrays.asList(
            "en", "ch_sim", "ch_tra", "ja", "ko", "th", "vi", "ar", "fa",
            "fr", "de", "it", "pt", "es", "nl", "pl", "ru", "uk", "hi",
            "ta", "te", "kn", "bn", "mr", "ne", "id", "ms", "tl", "tr",
            "az", "be", "bg", "bs", "cs", "cy", "da", "et", "fi", "ga",
            "hr", "hu", "is", "lt", "lv", "mi", "mn", "mt", "no", "oc",
            "ro", "sk", "sl", "sq", "sr_cyrl", "sv", "sw", "tg", "uz"
    );

    private String pythonPath;
    private File easyocrScript;

    private EasyOCREngine() {
        super();
        this.pythonPath = "python3";
    }

    /**
     * Create an EasyOCR engine with default settings.
     *
     * @return the OCR engine
     * @throws OCRException if EasyOCR is not available
     */
    public static EasyOCREngine create() throws OCRException {
        EasyOCREngine engine = new EasyOCREngine();
        engine.initialize();
        return engine;
    }

    /**
     * Create an EasyOCR engine with custom Python path.
     *
     * @param pythonPath path to Python executable
     * @return the OCR engine
     * @throws OCRException if initialization fails
     */
    public static EasyOCREngine create(String pythonPath) throws OCRException {
        EasyOCREngine engine = new EasyOCREngine();
        engine.pythonPath = pythonPath;
        engine.initialize();
        return engine;
    }

    /**
     * Create an EasyOCR engine with custom config.
     *
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if initialization fails
     */
    public static EasyOCREngine create(OCRConfig config) throws OCRException {
        EasyOCREngine engine = new EasyOCREngine();
        engine.setConfig(config);
        engine.initialize();
        return engine;
    }

    @Override
    protected void initialize() throws OCRException {
        // Check if Python is available
        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, "--version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                        "Python not found. Please install Python 3.x.");
            }
        } catch (IOException | InterruptedException e) {
            throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                    "Failed to check Python: " + e.getMessage(), e);
        }

        // Check if EasyOCR is installed
        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, "-c", "import easyocr; print('ok')");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                        "EasyOCR not installed. Run: pip install easyocr");
            }
        } catch (IOException | InterruptedException e) {
            throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                    "Failed to check EasyOCR: " + e.getMessage(), e);
        }

        // Create wrapper script
        createWrapperScript();

        initialized = true;
    }

    private void createWrapperScript() throws OCRException {
        try {
            easyocrScript = File.createTempFile("easyocr_wrapper_", ".py");
            easyocrScript.deleteOnExit();

            String script = "import sys\n" +
                    "import json\n" +
                    "import easyocr\n" +
                    "\n" +
                    "def main():\n" +
                    "    if len(sys.argv) < 3:\n" +
                    "        print(json.dumps({'error': 'Usage: script.py <image_path> <languages>'}))\n" +
                    "        return\n" +
                    "    \n" +
                    "    image_path = sys.argv[1]\n" +
                    "    languages = sys.argv[2].split(',')\n" +
                    "    gpu = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False\n" +
                    "    \n" +
                    "    reader = easyocr.Reader(languages, gpu=gpu)\n" +
                    "    results = reader.readtext(image_path)\n" +
                    "    \n" +
                    "    output = []\n" +
                    "    for (bbox, text, confidence) in results:\n" +
                    "        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]\n" +
                    "        x1 = int(min(p[0] for p in bbox))\n" +
                    "        y1 = int(min(p[1] for p in bbox))\n" +
                    "        x2 = int(max(p[0] for p in bbox))\n" +
                    "        y2 = int(max(p[1] for p in bbox))\n" +
                    "        output.append({\n" +
                    "            'text': text,\n" +
                    "            'confidence': float(confidence),\n" +
                    "            'bbox': [x1, y1, x2, y2]\n" +
                    "        })\n" +
                    "    \n" +
                    "    print(json.dumps({'results': output}))\n" +
                    "\n" +
                    "if __name__ == '__main__':\n" +
                    "    main()\n";

            try (FileWriter writer = new FileWriter(easyocrScript)) {
                writer.write(script);
            }
        } catch (IOException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                    "Failed to create wrapper script", e);
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
            processedFile = File.createTempFile("easyocr_", ".png");
            processedFile.deleteOnExit();
            ImageIO.write(image, "PNG", processedFile);
        }

        // Build command
        List<String> command = new ArrayList<>();
        command.add(pythonPath);
        command.add(easyocrScript.getAbsolutePath());
        command.add(processedFile.getAbsolutePath());
        command.add(String.join(",", languages != null && languages.length > 0 ? languages : new String[]{"en"}));
        command.add(String.valueOf(config.isUseGpu()));

        // Execute
        String output = executeCommand(command);

        // Parse results
        List<OCRTextRegion> regions = parseJsonOutput(output);

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
            File tempFile = File.createTempFile("easyocr_", ".png");
            tempFile.deleteOnExit();
            ImageIO.write(bufferedImage, "PNG", tempFile);
            return recognize(tempFile);
        } catch (IOException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR, "Failed to process image", e);
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
                        "EasyOCR failed with exit code " + exitCode + ": " + output);
            }

            return output.toString();
        } catch (IOException | InterruptedException e) {
            throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                    "Failed to execute EasyOCR: " + e.getMessage(), e);
        }
    }

    private List<OCRTextRegion> parseJsonOutput(String output) {
        List<OCRTextRegion> regions = new ArrayList<>();

        // Find JSON in output
        int jsonStart = output.indexOf('{');
        int jsonEnd = output.lastIndexOf('}');
        if (jsonStart < 0 || jsonEnd < 0) {
            return regions;
        }

        String json = output.substring(jsonStart, jsonEnd + 1);

        // Simple JSON parsing (could use Jackson/Gson for full implementation)
        if (json.contains("\"error\"")) {
            return regions;
        }

        // Parse results array
        Pattern resultPattern = Pattern.compile(
                "\\{\"text\":\\s*\"([^\"]*)\",\\s*\"confidence\":\\s*([\\d.]+),\\s*\"bbox\":\\s*\\[(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\]\\}"
        );
        Matcher matcher = resultPattern.matcher(json);

        while (matcher.find()) {
            String text = matcher.group(1);
            double confidence = Double.parseDouble(matcher.group(2));
            int x1 = Integer.parseInt(matcher.group(3));
            int y1 = Integer.parseInt(matcher.group(4));
            int x2 = Integer.parseInt(matcher.group(5));
            int y2 = Integer.parseInt(matcher.group(6));

            if (confidence >= config.getMinConfidence()) {
                regions.add(OCRTextRegion.builder()
                        .text(text)
                        .confidence(confidence)
                        .boundingBox(new BoundingBox(x1, y1, x2, y2))
                        .level(OCRTextRegion.TextLevel.LINE)
                        .build());
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
        return SUPPORTED_LANGUAGES;
    }

    @Override
    public void close() throws Exception {
        if (easyocrScript != null && easyocrScript.exists()) {
            easyocrScript.delete();
        }
        initialized = false;
    }
}
