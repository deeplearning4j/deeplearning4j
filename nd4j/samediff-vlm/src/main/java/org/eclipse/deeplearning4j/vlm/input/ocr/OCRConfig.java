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

import lombok.Builder;
import lombok.Data;

import java.util.Arrays;
import java.util.List;

/**
 * Configuration options for OCR engines.
 *
 * Adam Gibson
 */
@Data
@Builder
public class OCRConfig {

    /**
     * Languages to recognize.
     */
    @Builder.Default
    private List<String> languages = Arrays.asList("en");

    /**
     * Minimum confidence threshold (0-1).
     */
    @Builder.Default
    private double minConfidence = 0.5;

    /**
     * Whether to detect text orientation.
     */
    @Builder.Default
    private boolean detectOrientation = true;

    /**
     * Whether to detect text direction (LTR/RTL).
     */
    @Builder.Default
    private boolean detectDirection = true;

    /**
     * Whether to enable paragraph detection.
     */
    @Builder.Default
    private boolean detectParagraphs = false;

    /**
     * Whether to enable table detection.
     */
    @Builder.Default
    private boolean detectTables = false;

    /**
     * Page segmentation mode.
     */
    @Builder.Default
    private PageSegMode pageSegMode = PageSegMode.AUTO;

    /**
     * OCR engine mode.
     */
    @Builder.Default
    private EngineMode engineMode = EngineMode.DEFAULT;

    /**
     * Whether to use GPU acceleration.
     */
    @Builder.Default
    private boolean useGpu = false;

    /**
     * GPU device ID to use.
     */
    @Builder.Default
    private int gpuDeviceId = 0;

    /**
     * Number of threads for CPU processing.
     */
    @Builder.Default
    private int numThreads = 4;

    /**
     * Maximum image dimension (resize if larger).
     */
    @Builder.Default
    private int maxImageDimension = 4096;

    /**
     * Whether to preprocess images (deskew, denoise).
     */
    @Builder.Default
    private boolean preprocessImages = true;

    /**
     * DPI for image processing.
     */
    @Builder.Default
    private int dpi = 300;

    /**
     * Custom model path (engine-specific).
     */
    private String modelPath;

    /**
     * Additional engine-specific options.
     */
    private java.util.Map<String, Object> additionalOptions;

    /**
     * Page segmentation modes.
     */
    public enum PageSegMode {
        /**
         * Orientation and script detection only.
         */
        OSD_ONLY,

        /**
         * Automatic page segmentation with OSD.
         */
        AUTO_OSD,

        /**
         * Automatic page segmentation, no OSD.
         */
        AUTO,

        /**
         * Single column of text.
         */
        SINGLE_COLUMN,

        /**
         * Single block of text.
         */
        SINGLE_BLOCK,

        /**
         * Single line of text.
         */
        SINGLE_LINE,

        /**
         * Single word.
         */
        SINGLE_WORD,

        /**
         * Single character.
         */
        SINGLE_CHAR,

        /**
         * Sparse text.
         */
        SPARSE_TEXT,

        /**
         * Raw line (no processing).
         */
        RAW_LINE
    }

    /**
     * OCR engine modes.
     */
    public enum EngineMode {
        /**
         * Default mode for the engine.
         */
        DEFAULT,

        /**
         * Fast mode (may be less accurate).
         */
        FAST,

        /**
         * Accurate mode (may be slower).
         */
        ACCURATE,

        /**
         * Legacy mode (for backwards compatibility).
         */
        LEGACY,

        /**
         * Neural network mode.
         */
        NEURAL
    }

    /**
     * Create default configuration.
     *
     * @return default config
     */
    public static OCRConfig defaults() {
        return OCRConfig.builder().build();
    }

    /**
     * Create configuration for fast processing.
     *
     * @return fast config
     */
    public static OCRConfig fast() {
        return OCRConfig.builder()
                .engineMode(EngineMode.FAST)
                .preprocessImages(false)
                .detectParagraphs(false)
                .detectTables(false)
                .build();
    }

    /**
     * Create configuration for accurate processing.
     *
     * @return accurate config
     */
    public static OCRConfig accurate() {
        return OCRConfig.builder()
                .engineMode(EngineMode.ACCURATE)
                .preprocessImages(true)
                .detectParagraphs(true)
                .detectTables(true)
                .minConfidence(0.7)
                .build();
    }

    /**
     * Create configuration for GPU processing.
     *
     * @param deviceId GPU device ID
     * @return GPU config
     */
    public static OCRConfig gpu(int deviceId) {
        return OCRConfig.builder()
                .useGpu(true)
                .gpuDeviceId(deviceId)
                .build();
    }
}
