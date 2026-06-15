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

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Unified interface for OCR (Optical Character Recognition) engines.
 *
 * This interface provides a common API for various OCR backends including:
 * - DeepSeek OCR
 * - PaddleOCR
 * - Tesseract
 * - EasyOCR
 * - TrOCR (Transformer-based)
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * OCREngine ocr = PaddleOCREngine.create();
 * OCRResult result = ocr.recognize(new File("document.png"));
 *
 * // Get full text
 * String text = result.getFullText();
 *
 * // Get structured results with bounding boxes
 * for (OCRTextRegion region : result.getRegions()) {
 *     System.out.println(region.getText() + " @ " + region.getBoundingBox());
 * }
 * }</pre>
 *
 * Adam Gibson
 */
public interface OCREngine extends AutoCloseable {

    /**
     * Recognize text in an image file.
     *
     * @param imageFile the image file
     * @return OCR result with detected text and regions
     * @throws IOException if reading fails
     * @throws OCRException if recognition fails
     */
    OCRResult recognize(File imageFile) throws IOException, OCRException;

    /**
     * Recognize text in an image tensor.
     *
     * @param image the image tensor [H, W, C] or [C, H, W]
     * @return OCR result with detected text and regions
     * @throws OCRException if recognition fails
     */
    OCRResult recognize(INDArray image) throws OCRException;

    /**
     * Recognize text in multiple images.
     *
     * @param imageFiles the image files
     * @return list of OCR results
     * @throws IOException if reading fails
     * @throws OCRException if recognition fails
     */
    List<OCRResult> recognizeBatch(List<File> imageFiles) throws IOException, OCRException;

    /**
     * Get the name of this OCR engine.
     *
     * @return engine name
     */
    String getEngineName();

    /**
     * Get supported languages.
     *
     * @return list of language codes (e.g., "en", "zh", "ja")
     */
    List<String> getSupportedLanguages();

    /**
     * Set the recognition language(s).
     *
     * @param languages language codes to use
     */
    void setLanguages(String... languages);

    /**
     * Check if the engine is ready for use.
     *
     * @return true if initialized and ready
     */
    boolean isReady();

    /**
     * Get configuration options for this engine.
     *
     * @return configuration object
     */
    OCRConfig getConfig();

    /**
     * Set configuration options.
     *
     * @param config the configuration
     */
    void setConfig(OCRConfig config);
}
