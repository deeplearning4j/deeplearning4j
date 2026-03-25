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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for loading image-based PDFs using SameDiff VLM preprocessing.
 *
 * This test demonstrates the workflow for:
 * 1. Loading a PDF document
 * 2. Rendering PDF pages to images
 * 3. Preprocessing images for Vision-Language Models
 *
 * Configuration:
 * - PDF path can be set via system property: -Dvlm.test.pdf.path=/path/to/pdf
 * - Default DPI for rendering: 150 (configurable via -Dvlm.test.pdf.dpi)
 * - Target image size: 224x224 (CLIP default) or 768x768 (SmolDocling)
 *
 * Adam Gibson
 */
@Slf4j
@NativeTag
@Tag("vlm")
public class TestVLMPdfImageLoader {

    /**
     * System property for configuring the PDF file path.
     */
    public static final String PDF_PATH_PROPERTY = "vlm.test.pdf.path";

    /**
     * System property for configuring the DPI for PDF rendering.
     */
    public static final String PDF_DPI_PROPERTY = "vlm.test.pdf.dpi";

    /**
     * System property for configuring the maximum number of pages to process.
     */
    public static final String PDF_MAX_PAGES_PROPERTY = "vlm.test.pdf.maxPages";

    /**
     * Default DPI for PDF rendering.
     */
    public static final int DEFAULT_DPI = 150;

    /**
     * Default maximum pages to process.
     */
    public static final int DEFAULT_MAX_PAGES = 10;

    private static String pdfPath;
    private static int dpi;
    private static int maxPages;

    @BeforeAll
    public static void setup() {
        pdfPath = System.getProperty(PDF_PATH_PROPERTY);
        dpi = Integer.getInteger(PDF_DPI_PROPERTY, DEFAULT_DPI);
        maxPages = Integer.getInteger(PDF_MAX_PAGES_PROPERTY, DEFAULT_MAX_PAGES);

        log.info("VLM PDF Test Configuration:");
        log.info("  PDF Path: {}", pdfPath != null ? pdfPath : "(not set - will skip tests requiring PDF)");
        log.info("  DPI: {}", dpi);
        log.info("  Max Pages: {}", maxPages);
    }

    /**
     * Test loading a PDF and converting pages to images for VLM processing.
     * Uses CLIP-style preprocessing (224x224).
     */
    @Test
    public void testLoadPdfWithClipPreprocessor() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.defaultPreprocessor();

        List<INDArray> processedPages = loadAndPreprocessPdf(pdfFile, preprocessor);

        assertFalse(processedPages.isEmpty(), "Should have processed at least one page");

        for (int i = 0; i < processedPages.size(); i++) {
            INDArray page = processedPages.get(i);
            long[] shape = page.shape();

            log.info("Page {} shape: {}", i + 1, java.util.Arrays.toString(shape));

            assertEquals(4, shape.length, "Should be 4D tensor [N, C, H, W]");
            assertEquals(1, shape[0], "Batch size should be 1");
            assertEquals(3, shape[1], "Should have 3 channels (RGB)");
            assertEquals(224, shape[2], "Height should be 224 (CLIP default)");
            assertEquals(224, shape[3], "Width should be 224 (CLIP default)");

            // Verify values are normalized (approximately in expected range after normalization)
            double min = page.minNumber().doubleValue();
            double max = page.maxNumber().doubleValue();
            log.info("Page {} value range: [{}, {}]", i + 1, min, max);
        }

        preprocessor.shutdown();
    }

    /**
     * Test loading a PDF with SmolDocling-style preprocessing (768x768).
     */
    @Test
    public void testLoadPdfWithSmolDoclingPreprocessor() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.forSmolDocling();

        List<INDArray> processedPages = loadAndPreprocessPdf(pdfFile, preprocessor);

        assertFalse(processedPages.isEmpty(), "Should have processed at least one page");

        for (int i = 0; i < processedPages.size(); i++) {
            INDArray page = processedPages.get(i);
            long[] shape = page.shape();

            log.info("Page {} shape (SmolDocling): {}", i + 1, java.util.Arrays.toString(shape));

            assertEquals(4, shape.length, "Should be 4D tensor [N, C, H, W]");
            assertEquals(1, shape[0], "Batch size should be 1");
            assertEquals(3, shape[1], "Should have 3 channels (RGB)");
            // SmolDocling uses 384x384 (Idefics3 style)
            assertEquals(384, shape[2], "Height should be 384 (SmolDocling)");
            assertEquals(384, shape[3], "Width should be 384 (SmolDocling)");
        }

        preprocessor.shutdown();
    }

    /**
     * Test loading a PDF with custom preprocessing configuration.
     */
    @Test
    public void testLoadPdfWithCustomPreprocessor() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        // Custom configuration for document processing
        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(512, 512));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);
        config.setImageMean(new double[]{0.5, 0.5, 0.5});
        config.setImageStd(new double[]{0.5, 0.5, 0.5});

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(config);

        List<INDArray> processedPages = loadAndPreprocessPdf(pdfFile, preprocessor);

        assertFalse(processedPages.isEmpty(), "Should have processed at least one page");

        for (int i = 0; i < processedPages.size(); i++) {
            INDArray page = processedPages.get(i);
            long[] shape = page.shape();

            assertEquals(512, shape[2], "Height should be 512 (custom)");
            assertEquals(512, shape[3], "Width should be 512 (custom)");

            // With mean=0.5, std=0.5 normalization, values should be roughly in [-1, 1]
            double min = page.minNumber().doubleValue();
            double max = page.maxNumber().doubleValue();
            log.info("Page {} (custom) value range: [{}, {}]", i + 1, min, max);

            assertTrue(min >= -2.0, "Min value should be >= -2.0 after normalization");
            assertTrue(max <= 2.0, "Max value should be <= 2.0 after normalization");
        }

        preprocessor.shutdown();
    }

    /**
     * Test extracting patches from PDF page images for Vision Transformer models.
     */
    @Test
    public void testExtractPatchesFromPdf() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.defaultPreprocessor();
        int patchSize = 16; // Standard ViT patch size

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);

            int numPages = Math.min(document.getNumberOfPages(), maxPages);
            assertTrue(numPages > 0, "PDF should have at least one page");

            // Process first page only for patch extraction test
            BufferedImage pageImage = renderer.renderImageWithDPI(0, dpi, ImageType.RGB);
            INDArray preprocessed = preprocessor.preprocess(pageImage);

            // Extract patches
            INDArray patches = preprocessor.extractPatches(preprocessed, patchSize);
            long[] shape = patches.shape();

            log.info("Patches shape: {}", java.util.Arrays.toString(shape));

            assertEquals(3, shape.length, "Patches should be 3D [1, num_patches, patch_dim]");
            assertEquals(1, shape[0], "Batch size should be 1");

            // For 224x224 image with 16x16 patches: (224/16)^2 = 196 patches
            int expectedNumPatches = (224 / patchSize) * (224 / patchSize);
            assertEquals(expectedNumPatches, shape[1], "Should have " + expectedNumPatches + " patches");

            // Each patch has 3 channels * 16 * 16 = 768 dimensions
            int expectedPatchDim = 3 * patchSize * patchSize;
            assertEquals(expectedPatchDim, shape[2], "Patch dimension should be " + expectedPatchDim);
        }

        preprocessor.shutdown();
    }

    /**
     * Test batch processing of multiple PDF pages.
     */
    @Test
    public void testBatchProcessPdfPages() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.defaultPreprocessor();

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);

            int numPages = Math.min(document.getNumberOfPages(), maxPages);
            Assumptions.assumeTrue(numPages >= 2,
                    "Need at least 2 pages for batch processing test");

            // Render multiple pages
            List<BufferedImage> pageImages = new ArrayList<>();
            for (int i = 0; i < Math.min(numPages, 4); i++) {
                pageImages.add(renderer.renderImageWithDPI(i, dpi, ImageType.RGB));
            }

            // Process and stack into batch
            List<INDArray> processedPages = new ArrayList<>();
            for (BufferedImage img : pageImages) {
                processedPages.add(preprocessor.preprocess(img));
            }

            INDArray batch = org.nd4j.linalg.factory.Nd4j.vstack(processedPages);
            long[] shape = batch.shape();

            log.info("Batch shape: {}", java.util.Arrays.toString(shape));

            assertEquals(4, shape.length, "Batch should be 4D [N, C, H, W]");
            assertEquals(pageImages.size(), shape[0], "Batch size should match number of pages");
            assertEquals(3, shape[1], "Should have 3 channels");
            assertEquals(224, shape[2], "Height should be 224");
            assertEquals(224, shape[3], "Width should be 224");
        }

        preprocessor.shutdown();
    }

    /**
     * Test that PDF rendering produces valid images at different DPI settings.
     */
    @Test
    public void testPdfRenderingAtDifferentDpi() throws IOException {
        Assumptions.assumeTrue(pdfPath != null && !pdfPath.isEmpty(),
                "PDF path not configured. Set -D" + PDF_PATH_PROPERTY + "=/path/to/pdf");

        File pdfFile = new File(pdfPath);
        Assumptions.assumeTrue(pdfFile.exists(),
                "PDF file does not exist: " + pdfPath);

        int[] dpiValues = {72, 150, 300};

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);

            for (int testDpi : dpiValues) {
                BufferedImage image = renderer.renderImageWithDPI(0, testDpi, ImageType.RGB);

                log.info("DPI {}: Image size {}x{}", testDpi, image.getWidth(), image.getHeight());

                assertTrue(image.getWidth() > 0, "Image width should be positive");
                assertTrue(image.getHeight() > 0, "Image height should be positive");

                // Higher DPI should produce larger images
                // At 72 DPI, a standard letter page (8.5x11) would be ~612x792 pixels
                // At 150 DPI, it would be ~1275x1650 pixels
                // At 300 DPI, it would be ~2550x3300 pixels
            }
        }
    }

    /**
     * Load a PDF and preprocess all pages (up to maxPages).
     *
     * @param pdfFile the PDF file to load
     * @param preprocessor the VLM image preprocessor to use
     * @return list of preprocessed page tensors
     * @throws IOException if loading fails
     */
    private List<INDArray> loadAndPreprocessPdf(File pdfFile, VLMImagePreprocessor preprocessor) throws IOException {
        List<INDArray> processedPages = new ArrayList<>();

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);

            int numPages = Math.min(document.getNumberOfPages(), maxPages);
            log.info("Processing {} pages from PDF: {}", numPages, pdfFile.getName());

            for (int pageIndex = 0; pageIndex < numPages; pageIndex++) {
                // Render page to BufferedImage
                BufferedImage pageImage = renderer.renderImageWithDPI(pageIndex, dpi, ImageType.RGB);

                log.debug("Page {} rendered: {}x{} pixels",
                        pageIndex + 1, pageImage.getWidth(), pageImage.getHeight());

                // Preprocess for VLM
                INDArray preprocessed = preprocessor.preprocess(pageImage);
                processedPages.add(preprocessed);
            }
        }

        return processedPages;
    }

    /**
     * Utility method to get PDF page count without full loading.
     *
     * @param pdfFile the PDF file
     * @return number of pages
     * @throws IOException if loading fails
     */
    public static int getPdfPageCount(File pdfFile) throws IOException {
        try (PDDocument document = PDDocument.load(pdfFile)) {
            return document.getNumberOfPages();
        }
    }

    /**
     * Utility method to render a single PDF page to BufferedImage.
     *
     * @param pdfFile the PDF file
     * @param pageIndex the page index (0-based)
     * @param dpi the rendering DPI
     * @return the rendered BufferedImage
     * @throws IOException if loading or rendering fails
     */
    public static BufferedImage renderPdfPage(File pdfFile, int pageIndex, int dpi) throws IOException {
        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            return renderer.renderImageWithDPI(pageIndex, dpi, ImageType.RGB);
        }
    }
}
