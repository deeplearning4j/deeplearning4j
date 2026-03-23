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
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class for OCR engine implementations.
 *
 * Adam Gibson
 */
public abstract class AbstractOCREngine implements OCREngine {

    protected OCRConfig config;
    protected String[] languages;
    protected boolean initialized;

    protected AbstractOCREngine() {
        this.config = OCRConfig.defaults();
        this.languages = new String[]{"en"};
        this.initialized = false;
    }

    @Override
    public List<OCRResult> recognizeBatch(List<File> imageFiles) throws IOException, OCRException {
        List<OCRResult> results = new ArrayList<>();
        for (File file : imageFiles) {
            results.add(recognize(file));
        }
        return results;
    }

    @Override
    public void setLanguages(String... languages) {
        this.languages = languages;
    }

    @Override
    public boolean isReady() {
        return initialized;
    }

    @Override
    public OCRConfig getConfig() {
        return config;
    }

    @Override
    public void setConfig(OCRConfig config) {
        this.config = config;
    }

    /**
     * Convert a BufferedImage to INDArray.
     *
     * @param image the image
     * @return INDArray [H, W, C] format
     */
    protected INDArray imageToArray(BufferedImage image) {
        int height = image.getHeight();
        int width = image.getWidth();
        int channels = 3;

        float[] data = new float[height * width * channels];
        int idx = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                data[idx++] = ((rgb >> 16) & 0xFF) / 255.0f; // R
                data[idx++] = ((rgb >> 8) & 0xFF) / 255.0f;  // G
                data[idx++] = (rgb & 0xFF) / 255.0f;         // B
            }
        }

        return Nd4j.create(data, new int[]{height, width, channels});
    }

    /**
     * Load image from file.
     *
     * @param file the image file
     * @return loaded image
     * @throws IOException if reading fails
     */
    protected BufferedImage loadImage(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        if (image == null) {
            throw new IOException("Failed to read image: " + file.getAbsolutePath());
        }
        return image;
    }

    /**
     * Preprocess image according to config.
     *
     * @param image the input image
     * @return preprocessed image
     */
    protected BufferedImage preprocessImage(BufferedImage image) {
        if (!config.isPreprocessImages()) {
            return image;
        }

        // Resize if too large
        int maxDim = config.getMaxImageDimension();
        if (image.getWidth() > maxDim || image.getHeight() > maxDim) {
            double scale = Math.min(
                    (double) maxDim / image.getWidth(),
                    (double) maxDim / image.getHeight()
            );
            int newWidth = (int) (image.getWidth() * scale);
            int newHeight = (int) (image.getHeight() * scale);

            BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
            resized.getGraphics().drawImage(
                    image.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH),
                    0, 0, null
            );
            return resized;
        }

        return image;
    }

    /**
     * Validate that the engine is initialized.
     *
     * @throws OCRException if not initialized
     */
    protected void checkInitialized() throws OCRException {
        if (!initialized) {
            throw new OCRException(
                    OCRException.ErrorCode.NOT_INITIALIZED,
                    "OCR engine not initialized. Call initialize() first."
            );
        }
    }

    /**
     * Initialize the engine. Subclasses should override.
     *
     * @throws OCRException if initialization fails
     */
    protected abstract void initialize() throws OCRException;
}
