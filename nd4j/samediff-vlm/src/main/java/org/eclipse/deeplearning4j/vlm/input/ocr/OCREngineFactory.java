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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Factory for creating OCR engine instances.
 *
 * <p>Usage:</p>
 * <pre>{@code
 * // Create default engine (Tesseract if available, otherwise throws)
 * OCREngine ocr = OCREngineFactory.createDefault();
 *
 * // Create specific engine
 * OCREngine ocr = OCREngineFactory.create(EngineType.PADDLE_OCR, new File("models"));
 *
 * // Create best available engine
 * OCREngine ocr = OCREngineFactory.createBestAvailable();
 * }</pre>
 *
 * Adam Gibson
 */
public class OCREngineFactory {

    /**
     * Available OCR engine types.
     */
    public enum EngineType {
        /**
         * Tesseract OCR (command-line wrapper).
         */
        TESSERACT,

        /**
         * PaddleOCR (ONNX models).
         */
        PADDLE_OCR,

        /**
         * EasyOCR (Python wrapper).
         */
        EASY_OCR,

        /**
         * DeepSeek OCR (ONNX models).
         */
        DEEPSEEK_OCR,

        /**
         * TrOCR (Microsoft transformer-based OCR).
         */
        TR_OCR
    }

    private OCREngineFactory() {
        // Utility class
    }

    /**
     * Create an OCR engine of the specified type.
     *
     * @param type engine type
     * @return the OCR engine
     * @throws OCRException if creation fails
     */
    public static OCREngine create(EngineType type) throws OCRException {
        switch (type) {
            case TESSERACT:
                return TesseractOCREngine.create();
            case EASY_OCR:
                return EasyOCREngine.create();
            default:
                throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                        "Engine type " + type + " requires a model directory");
        }
    }

    /**
     * Create an OCR engine with a model directory.
     *
     * @param type engine type
     * @param modelDirectory model directory
     * @return the OCR engine
     * @throws OCRException if creation fails
     */
    public static OCREngine create(EngineType type, File modelDirectory) throws OCRException {
        switch (type) {
            case TESSERACT:
                OCRConfig config = OCRConfig.builder()
                        .modelPath(modelDirectory.getAbsolutePath())
                        .build();
                return TesseractOCREngine.create(config);
            case PADDLE_OCR:
                return PaddleOCREngine.create(modelDirectory);
            case EASY_OCR:
                return EasyOCREngine.create();
            case DEEPSEEK_OCR:
                return DeepSeekOCREngine.create(modelDirectory);
            case TR_OCR:
                return TrOCREngine.create(modelDirectory);
            default:
                throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                        "Unknown engine type: " + type);
        }
    }

    /**
     * Create an OCR engine with configuration.
     *
     * @param type engine type
     * @param modelDirectory model directory (can be null for some engines)
     * @param config OCR configuration
     * @return the OCR engine
     * @throws OCRException if creation fails
     */
    public static OCREngine create(EngineType type, File modelDirectory, OCRConfig config) throws OCRException {
        switch (type) {
            case TESSERACT:
                return TesseractOCREngine.create(config);
            case PADDLE_OCR:
                return PaddleOCREngine.create(modelDirectory, config);
            case EASY_OCR:
                return EasyOCREngine.create(config);
            case DEEPSEEK_OCR:
                return DeepSeekOCREngine.create(modelDirectory, config);
            case TR_OCR:
                return TrOCREngine.create(modelDirectory, config);
            default:
                throw new OCRException(OCRException.ErrorCode.ENGINE_ERROR,
                        "Unknown engine type: " + type);
        }
    }

    /**
     * Create the default OCR engine (Tesseract).
     *
     * @return Tesseract OCR engine
     * @throws OCRException if Tesseract is not available
     */
    public static OCREngine createDefault() throws OCRException {
        return TesseractOCREngine.create();
    }

    /**
     * Create the best available OCR engine.
     *
     * This method tries engines in order of preference and returns
     * the first one that initializes successfully.
     *
     * @return the best available OCR engine
     * @throws OCRException if no engine is available
     */
    public static OCREngine createBestAvailable() throws OCRException {
        List<String> errors = new ArrayList<>();

        // Try Tesseract first (most widely available)
        try {
            return TesseractOCREngine.create();
        } catch (OCRException e) {
            errors.add("Tesseract: " + e.getMessage());
        }

        // Try EasyOCR (Python-based, good accuracy)
        try {
            return EasyOCREngine.create();
        } catch (OCRException e) {
            errors.add("EasyOCR: " + e.getMessage());
        }

        throw new OCRException(OCRException.ErrorCode.NOT_INITIALIZED,
                "No OCR engine available. Tried:\n" + String.join("\n", errors));
    }

    /**
     * Check if an engine type is available.
     *
     * @param type the engine type
     * @return true if the engine can be created
     */
    public static boolean isAvailable(EngineType type) {
        try {
            switch (type) {
                case TESSERACT:
                    TesseractOCREngine tesseract = TesseractOCREngine.create();
                    tesseract.close();
                    return true;
                case EASY_OCR:
                    EasyOCREngine easyocr = EasyOCREngine.create();
                    easyocr.close();
                    return true;
                default:
                    // Model-based engines need a model directory
                    return false;
            }
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Check if an engine type with a model directory is available.
     *
     * @param type the engine type
     * @param modelDirectory the model directory
     * @return true if the engine can be created
     */
    public static boolean isAvailable(EngineType type, File modelDirectory) {
        try {
            OCREngine engine = create(type, modelDirectory);
            engine.close();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get a list of all available engine types.
     *
     * @return list of available engine types
     */
    public static List<EngineType> getAvailableEngines() {
        List<EngineType> available = new ArrayList<>();
        for (EngineType type : new EngineType[]{EngineType.TESSERACT, EngineType.EASY_OCR}) {
            if (isAvailable(type)) {
                available.add(type);
            }
        }
        return available;
    }
}
