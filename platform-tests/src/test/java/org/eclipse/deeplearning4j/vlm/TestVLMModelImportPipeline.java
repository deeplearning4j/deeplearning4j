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
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.output.DocTagsParser;
import org.eclipse.deeplearning4j.vlm.output.DocumentStructure;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.PipelinedVisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;
import org.bytedeco.javacpp.Pointer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for VLM model download, import, and inference pipeline.
 *
 * This test suite demonstrates:
 * 1. Downloading models from HuggingFace (ONNX and GGUF formats)
 * 2. Importing models into SameDiff
 * 3. Preprocessing images (including PDF pages)
 * 4. Running inference through vision models
 *
 * Configuration via system properties:
 * - vlm.test.pdf.path: Path to a PDF file for testing
 * - vlm.test.pdf.page: (Optional) Specific page index to process (0-based). If set, only this page is processed.
 * - vlm.test.pdf.maxPages: (Optional) Maximum number of pages to process. If not set, processes all pages.
 * - vlm.test.pdf.dpi: (Optional) DPI for rendering PDF pages (default: 150)
 * - vlm.test.maxTiles: (Optional) Maximum number of tiles per image (default: unlimited). Useful for faster testing.
 * - vlm.model.cache.dir: Directory to cache downloaded models
 *
 * Examples:
 *   # Process first 5 pages of a book at 200 DPI
 *   -Dvlm.test.pdf.path=/path/to/book.pdf -Dvlm.test.pdf.maxPages=5 -Dvlm.test.pdf.dpi=200
 *
 *   # Process only page 10 (0-based index)
 *   -Dvlm.test.pdf.path=/path/to/book.pdf -Dvlm.test.pdf.page=10
 *
 *   # Process all pages at default DPI
 *   -Dvlm.test.pdf.path=/path/to/book.pdf
 *
 *   # Fast testing: limit to 4 tiles (3 content tiles + 1 global) at lower DPI
 *   -Dvlm.test.pdf.path=/path/to/book.pdf -Dvlm.test.pdf.page=10 -Dvlm.test.pdf.dpi=100 -Dvlm.test.maxTiles=4
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestVLMModelImportPipeline {

    private static final String PDF_PATH_PROPERTY = "vlm.test.pdf.path";
    private static final String PDF_PAGE_PROPERTY = "vlm.test.pdf.page";       // Single page (0-based)
    private static final String PDF_START_PAGE_PROPERTY = "vlm.test.pdf.startPage"; // Starting page for batch (0-based)
    private static final String PDF_MAX_PAGES_PROPERTY = "vlm.test.pdf.maxPages"; // Max pages to process
    private static final String PDF_DPI_PROPERTY = "vlm.test.pdf.dpi";         // Render DPI (default 150)
    private static final String MAX_TILES_PROPERTY = "vlm.test.maxTiles";      // Max tiles per image (default -1 = no limit)
    private static final String MAX_TOKENS_PROPERTY = "vlm.test.maxTokens";    // Max tokens to generate (default 50)

    private static String pdfPath;
    private static int specificPage = -1;   // -1 means process all/range
    private static int startPage = 0;       // Starting page for batch processing (0-based)
    private static int maxPages = -1;       // -1 means no limit
    private static int renderDpi = 150;
    private static int maxTiles = -1;       // -1 means no limit
    private static int maxTokensConfig = 50; // default token generation limit

    @BeforeAll
    public static void setup() {
        String rawPdfPath = System.getProperty(PDF_PATH_PROPERTY);
        pdfPath = (rawPdfPath != null && !rawPdfPath.isEmpty()) ? rawPdfPath : null;

        // Parse page selection properties (handle empty strings from Maven property forwarding)
        String pageStr = System.getProperty(PDF_PAGE_PROPERTY);
        if (pageStr != null && !pageStr.isEmpty()) {
            specificPage = Integer.parseInt(pageStr);
        }

        String maxPagesStr = System.getProperty(PDF_MAX_PAGES_PROPERTY);
        if (maxPagesStr != null && !maxPagesStr.isEmpty()) {
            maxPages = Integer.parseInt(maxPagesStr);
        }

        String startPageStr = System.getProperty(PDF_START_PAGE_PROPERTY);
        if (startPageStr != null && !startPageStr.isEmpty()) {
            startPage = Integer.parseInt(startPageStr);
        }

        String dpiStr = System.getProperty(PDF_DPI_PROPERTY);
        if (dpiStr != null && !dpiStr.isEmpty()) {
            renderDpi = Integer.parseInt(dpiStr);
        }

        String maxTilesStr = System.getProperty(MAX_TILES_PROPERTY);
        if (maxTilesStr != null && !maxTilesStr.isEmpty()) {
            maxTiles = Integer.parseInt(maxTilesStr);
        }

        String maxTokensStr = System.getProperty(MAX_TOKENS_PROPERTY);
        if (maxTokensStr != null && !maxTokensStr.isEmpty()) {
            maxTokensConfig = Integer.parseInt(maxTokensStr);
        }

        log.info("VLM Model Import Pipeline Test Configuration:");
        log.info("  PDF Path: {}", pdfPath != null ? pdfPath : "(not set)");
        log.info("  Specific Page: {}", specificPage >= 0 ? specificPage : "(all pages)");
        log.info("  Start Page: {}", startPage);
        log.info("  Max Pages: {}", maxPages > 0 ? maxPages : "(no limit)");
        log.info("  Render DPI: {}", renderDpi);
        log.info("  Max Tiles: {}", maxTiles > 0 ? maxTiles : "(no limit)");
        log.info("  Max Tokens: {}", maxTokensConfig);
        log.info("  Model Cache Dir: {}", VLMModelDownloader.getCacheDir());
        log.info("  Debug mode: {}", Nd4j.getEnvironment().isDebug());
        log.info("  Verbose mode: {}", Nd4j.getEnvironment().isVerbose());
    }

    // ==================== ONNX Model Tests ====================

    @Test
    @DisplayName("Download and import MobileViT ONNX model (lightweight)")
    public void testMobileViTOnnxImport() throws Exception {
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.MOBILEVIT_SMALL;

        // Download model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        assertTrue(result.getModelFile().exists(), "Model file should exist");
        log.info("Model file size: {} MB", result.getFileSizeBytes() / (1024 * 1024));

        // Import to SameDiff
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);

        assertNotNull(sd, "SameDiff model should not be null");
        log.info("Imported model with {} variables", sd.variables().size());

        // Create test input
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray testInput = createTestInput(model);

        log.info("Test input shape: {}", java.util.Arrays.toString(testInput.shape()));

        // Run inference (just verify no errors)
        // Note: actual output validation depends on model architecture
        preprocessor.shutdown();
    }

    @Test
    @DisplayName("Download and import CLIP ViT ONNX model")
    public void testClipViTOnnxImport() throws Exception {
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.CLIP_VIT_BASE_PATCH32;

        // Download model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        assertTrue(result.getModelFile().exists(), "Model file should exist");
        log.info("Model file size: {} MB", result.getFileSizeBytes() / (1024 * 1024));

        // Import to SameDiff
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);

        assertNotNull(sd, "SameDiff model should not be null");
        log.info("Imported CLIP model with {} variables", sd.variables().size());

        // Verify CLIP-style preprocessing works
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.defaultPreprocessor();
        INDArray testInput = createTestInput(model);

        assertEquals(224, testInput.shape()[2], "CLIP expects 224x224 input");
        assertEquals(224, testInput.shape()[3], "CLIP expects 224x224 input");

        preprocessor.shutdown();
    }

    @Test
    @DisplayName("Download and import ViT-384 ONNX model")
    public void testViT384OnnxImport() throws Exception {
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.VIT_BASE_PATCH16_384;

        // Download model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        assertTrue(result.getModelFile().exists(), "Model file should exist");

        // Import to SameDiff
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);

        assertNotNull(sd, "SameDiff model should not be null");
        log.info("Imported ViT-384 model with {} variables", sd.variables().size());

        // Verify 384x384 preprocessing
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray testInput = createTestInput(model);

        assertEquals(384, testInput.shape()[2], "ViT-384 expects 384x384 input");
        assertEquals(384, testInput.shape()[3], "ViT-384 expects 384x384 input");

        preprocessor.shutdown();
    }

    // ==================== GGUF Model Tests ====================

    @Test
    @DisplayName("Download and import CLIP GGUF model")
    public void testClipGgufImport() throws Exception {
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.CLIP_VIT_B32_VISION_GGUF;

        // Download model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        assertTrue(result.getModelFile().exists(), "Model file should exist");
        log.info("GGUF model file size: {} MB", result.getFileSizeBytes() / (1024 * 1024));

        // Inspect GGUF metadata
        var metadata = GGMLModelImport.inspectModel(result.getModelFile());
        log.info("GGUF Architecture: {}", metadata.getArchitecture());
        log.info("GGUF Total Parameters: {}", metadata.getTotalParameters());

        // Import to SameDiff
        SameDiff sd = GGMLModelImport.importModel(result.getModelFile());

        assertNotNull(sd, "SameDiff model should not be null");
        log.info("Imported GGUF model with {} variables", sd.variables().size());
    }

    @Test
    @DisplayName("Download and import LLaVA projector GGUF model")
    public void testLlavaProjectorGgufImport() throws Exception {
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.LLAVA_MMPROJ_F16_GGUF;

        // Download model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        assertTrue(result.getModelFile().exists(), "Model file should exist");

        // Inspect GGUF metadata
        var metadata = GGMLModelImport.inspectModel(result.getModelFile());
        log.info("LLaVA Projector Architecture: {}", metadata.getArchitecture());

        // Import to SameDiff
        SameDiff sd = GGMLModelImport.importModel(result.getModelFile());

        assertNotNull(sd, "SameDiff model should not be null");
        log.info("Imported LLaVA projector with {} variables", sd.variables().size());
    }

    // ==================== PDF + Model Pipeline Tests ====================
    // Run with: -Dvlm.test.pdf.path=/path/to/your.pdf

    @Test
    @DisplayName("PDF inference with SigLIP")
    public void testPdfWithSiglip() throws Exception {
        runPdfInference(VLMModelDownloader.VLMModel.SIGLIP_VISION);
    }

    @Test
    @Order(21)
    @DisplayName("PDF inference with SigLIP (alternative)")
    public void testPdfWithSiglipAlt() throws Exception {
        // Use SigLIP vision-only encoder instead of full CLIP (which requires text+vision inputs)
        runPdfInference(VLMModelDownloader.VLMModel.SIGLIP_VISION);
    }

    @Test
    @Order(22)
    @DisplayName("PDF inference with MobileViT")
    public void testPdfWithMobileVit() throws Exception {
        runPdfInference(VLMModelDownloader.VLMModel.MOBILEVIT_SMALL);
    }

    @Test
    @Order(23)
    @DisplayName("PDF inference with DeiT-384")
    public void testPdfWithDeit384() throws Exception {
        runPdfInference(VLMModelDownloader.VLMModel.VIT_BASE_PATCH16_384);
    }

    // ==================== PDF Page Rendering and Inspection Tests ====================

    @Test
    @Order(24)
    @DisplayName("Render and save PDF pages for inspection")
    public void testRenderAndSavePdfPages() throws Exception {
        Assumptions.assumeTrue(hasPdf(), "No PDF configured. Set -Dvlm.test.pdf.path=/path/to/book.pdf");

        log.info("=== Rendering PDF Pages for Inspection ===");

        int totalPages = getPdfPageCount();
        log.info("PDF has {} total pages", totalPages);

        List<Integer> pageIndices = getPageIndicesToProcess();
        log.info("Will process {} pages: {}", pageIndices.size(), pageIndices);

        List<String> savedFiles = new java.util.ArrayList<>();

        for (int pageIndex : pageIndices) {
            log.info("\n--- Rendering page {} ---", pageIndex);

            BufferedImage pageImage = loadPageFromPdf(pageIndex);

            // Log image details
            log.info("Page {} rendered:", pageIndex);
            log.info("  Dimensions: {}x{}", pageImage.getWidth(), pageImage.getHeight());
            log.info("  Type: {} ({})", pageImage.getType(), getImageTypeName(pageImage.getType()));
            log.info("  Color model: {}", pageImage.getColorModel());

            // Sample some pixels to verify content
            if (pageImage.getWidth() > 0 && pageImage.getHeight() > 0) {
                int centerX = pageImage.getWidth() / 2;
                int centerY = pageImage.getHeight() / 2;
                int centerPixel = pageImage.getRGB(centerX, centerY);
                int r = (centerPixel >> 16) & 0xFF;
                int g = (centerPixel >> 8) & 0xFF;
                int b = centerPixel & 0xFF;
                log.info("  Center pixel RGB: ({}, {}, {})", r, g, b);

                // Check corners
                int topLeft = pageImage.getRGB(0, 0);
                int topRight = pageImage.getRGB(pageImage.getWidth() - 1, 0);
                int bottomLeft = pageImage.getRGB(0, pageImage.getHeight() - 1);
                log.info("  Top-left RGB: ({}, {}, {})",
                        (topLeft >> 16) & 0xFF, (topLeft >> 8) & 0xFF, topLeft & 0xFF);
                log.info("  Top-right RGB: ({}, {}, {})",
                        (topRight >> 16) & 0xFF, (topRight >> 8) & 0xFF, topRight & 0xFF);
            }

            // Save the image
            String savedPath = savePageImage(pageImage, "pdf_render", pageIndex);
            savedFiles.add(savedPath);
        }

        log.info("\n=== Saved {} page images ===", savedFiles.size());
        for (String path : savedFiles) {
            log.info("  {}", path);
        }

        String outputDir = System.getProperty("vlm.test.output.dir", "target/vlm-test-output");
        log.info("\nInspect saved images in: {}", new File(outputDir).getAbsolutePath());
    }

    @Test
    @Order(25)
    @DisplayName("Compare different DPI renderings")
    public void testCompareDpiRenderings() throws Exception {
        Assumptions.assumeTrue(hasPdf(), "No PDF configured. Set -Dvlm.test.pdf.path=/path/to/book.pdf");

        log.info("=== Comparing DPI Renderings ===");

        int pageIndex = specificPage >= 0 ? specificPage : 0;
        int[] dpiValues = {72, 150, 300};

        for (int dpi : dpiValues) {
            log.info("\n--- Rendering page {} at {} DPI ---", pageIndex, dpi);

            BufferedImage pageImage = loadPageFromPdf(pageIndex, dpi);
            log.info("  Dimensions: {}x{}", pageImage.getWidth(), pageImage.getHeight());

            String outputDir = System.getProperty("vlm.test.output.dir", "target/vlm-test-output");
            String filename = String.format("pdf_page_%03d_dpi_%d.png", pageIndex, dpi);
            String outputPath = outputDir + File.separator + filename;
            saveImage(pageImage, outputPath);
        }

        log.info("\n=== DPI Comparison Complete ===");
    }

    /**
     * Get human-readable name for BufferedImage type constant.
     */
    private String getImageTypeName(int type) {
        switch (type) {
            case BufferedImage.TYPE_INT_RGB: return "TYPE_INT_RGB";
            case BufferedImage.TYPE_INT_ARGB: return "TYPE_INT_ARGB";
            case BufferedImage.TYPE_INT_ARGB_PRE: return "TYPE_INT_ARGB_PRE";
            case BufferedImage.TYPE_INT_BGR: return "TYPE_INT_BGR";
            case BufferedImage.TYPE_3BYTE_BGR: return "TYPE_3BYTE_BGR";
            case BufferedImage.TYPE_4BYTE_ABGR: return "TYPE_4BYTE_ABGR";
            case BufferedImage.TYPE_4BYTE_ABGR_PRE: return "TYPE_4BYTE_ABGR_PRE";
            case BufferedImage.TYPE_BYTE_GRAY: return "TYPE_BYTE_GRAY";
            case BufferedImage.TYPE_BYTE_BINARY: return "TYPE_BYTE_BINARY";
            case BufferedImage.TYPE_BYTE_INDEXED: return "TYPE_BYTE_INDEXED";
            case BufferedImage.TYPE_USHORT_GRAY: return "TYPE_USHORT_GRAY";
            case BufferedImage.TYPE_USHORT_565_RGB: return "TYPE_USHORT_565_RGB";
            case BufferedImage.TYPE_USHORT_555_RGB: return "TYPE_USHORT_555_RGB";
            case BufferedImage.TYPE_CUSTOM: return "TYPE_CUSTOM";
            default: return "UNKNOWN(" + type + ")";
        }
    }

    // ==================== Docling Document Understanding Tests ====================

    @Test
    @DisplayName("Full SmolDocling pipeline: PDF -> Vision -> Decoder -> Text")
    public void testSmolDoclingFullPipeline() throws Exception {
        // ==================== STEP 1: Download Models ====================
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE: All models downloaded.");

        // ==================== STEP 2: Load Tokenizer ====================
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("STEP 2 DONE: vocab_size={}, eos={}, bos={}",
                tokenizer.getVocabSize(), tokenizer.getEosTokenId(), tokenizer.getBosTokenId());

        // ==================== STEP 3: Import ONNX Models (with SDZ caching) ====================
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];
        log.info("  Vision encoder: {} variables", visionEncoder.variables().size());
        log.info("  Decoder: {} variables", decoder.variables().size());
        log.info("  Embed tokens: {} variables", embedTokens.variables().size());
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Load and Tile Image ====================
        long step4Start = System.currentTimeMillis();
        log.info("STEP 4: Loading image from PDF...");
        BufferedImage pdfImage = loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        log.info("  Raw image: {}x{}", pdfImage.getWidth(), pdfImage.getHeight());

        int targetSize = 512;
        BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(pdfImage, 2048);
        int effectiveMaxTiles = maxTiles > 0 ? maxTiles : 9;
        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(resizedForTiling, targetSize, effectiveMaxTiles);
        int numFrames = splitResult.getTotalFrames();
        log.info("STEP 4 DONE: {} frames ({} tiles + 1 global) [{}ms]", numFrames, splitResult.getTileCount(),
                System.currentTimeMillis() - step4Start);

        // ==================== STEP 5: Preprocess Frames ====================
        long step5Start = System.currentTimeMillis();
        log.info("STEP 5: Preprocessing frames...");
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, true);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, targetSize);
        preprocessor.shutdown();
        log.info("STEP 5 DONE: tensor shape={} [{}ms]", java.util.Arrays.toString(imageInput.shape()),
                System.currentTimeMillis() - step5Start);

        // ==================== STEP 6: Run Vision Encoder Per Frame ====================
        long step6Start = System.currentTimeMillis();
        log.info("STEP 6: Running vision encoder on {} frames...", numFrames);
        List<String> visionInputNames = visionEncoder.inputs();
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new java.util.ArrayList<>();

        // Enable debug+verbose BEFORE frame 1 so workspace canary checks
        // and "Executing op" messages run during the frame where corruption accumulates
        log.info("Enabling debug+verbose mode before vision encoder frames");
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            long frameStart = System.currentTimeMillis();
            INDArray frameSlice = imageInput.get(
                    NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray singleFrame = frameSlice.reshape(1, 1, 3, targetSize, targetSize).dup();

            Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                    visionInputMap.put(inputName,
                            ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                }
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            if (selected == null) {
                throw new RuntimeException("Vision encoder produced no usable outputs for frame " + frameIdx);
            }
            INDArray out = selected.tensor.dup();
            log.info("  Frame {}/{}: output shape={} [{}ms]", frameIdx + 1, numFrames,
                    java.util.Arrays.toString(out.shape()), System.currentTimeMillis() - frameStart);
            frameEmbeddings.add(out);

            // Release frame resources
            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
            }
            singleFrame.close();
            for (var entry : visionInputMap.entrySet()) {
                if (!entry.getKey().equals("pixel_values")) entry.getValue().close();
            }
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
        }

        // Concatenate frame embeddings
        INDArray visionEmbeddings;
        if (frameEmbeddings.size() == 1) {
            visionEmbeddings = frameEmbeddings.get(0).dup();
        } else {
            visionEmbeddings = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0])).dup();
        }
        for (INDArray fe : frameEmbeddings) {
            if (fe != null && fe.closeable() && !fe.wasClosed()) fe.close();
        }
        frameEmbeddings.clear();
        visionEncoder.resetSession();
        log.info("STEP 6 DONE: vision embeddings shape={} [{}ms total, {}ms/frame avg]",
                java.util.Arrays.toString(visionEmbeddings.shape()),
                System.currentTimeMillis() - step6Start,
                (System.currentTimeMillis() - step6Start) / numFrames);

        // ==================== STEP 7: Build Prompt and Get Text Embeddings ====================
        long step7Start = System.currentTimeMillis();
        log.info("STEP 7: Building prompt and computing text embeddings...");
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerImage = (int) visionEmbeddings.size(1) / numFrames;
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(splitResult.numRows, splitResult.numCols, imageSeqLenPerImage);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";

        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("  Prompt: {} tokens, {} <image> tokens", promptTokenIds.length,
                ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId));

        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);

        Map<String, INDArray> embedOutputs = embedTokens.output(Map.of(embedInputName, promptTokenIdsTensor), embedOutputNames);
        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }
        if (textEmbeddings == null) {
            throw new RuntimeException("embed_tokens produced no output");
        }

        long hiddenSize = visionEmbeddings.shape()[2];
        if (hiddenSize != textEmbeddings.shape()[2]) {
            throw new IllegalStateException("Hidden size mismatch: vision=" + hiddenSize + " text=" + textEmbeddings.shape()[2]);
        }

        // Merge vision embeddings into text embeddings at <image> token positions
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings,
                promptTokenIds, imageTokenId);
        log.info("STEP 7 DONE: merged embeddings shape={} [{}ms]", java.util.Arrays.toString(inputsEmbeds.shape()),
                System.currentTimeMillis() - step7Start);

        // Free vision encoder model - no longer needed after embedding merge.
        // Releases ~1.5-2GB of GPU constants from the CUDA pool, giving the decoder
        // more headroom for KV cache growth and activation memory.
        // NOTE: ONNX-imported constants have buffer.setConstant(true), making closeable()
        // return false. We must unset the constant flag before closing.
        log.info("  Freeing vision encoder model constants to reclaim GPU memory...");
        int closedVisionArrays = 0;
        long closedBytes = 0;
        // Close constant arrays (model weights) - these have isConstant=true on their buffers
        ArrayHolder constantHolder = visionEncoder.getConstantArrays();
        for (String name : new ArrayList<>(constantHolder.arrayNames())) {
            INDArray arr = constantHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedVisionArrays++;
            }
        }
        // Close variable arrays (trainable params, if any)
        ArrayHolder varHolder = visionEncoder.getVariablesArrays();
        for (String name : new ArrayList<>(varHolder.arrayNames())) {
            INDArray arr = varHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedVisionArrays++;
            }
        }
        Nd4j.getExecutioner().commit();
        log.info("  Closed {} vision encoder arrays ({}MB)", closedVisionArrays, closedBytes / (1024 * 1024));
        visionEncoder = null;

        // Also close text embeddings - merged into inputsEmbeds, no longer needed
        if (textEmbeddings != null && textEmbeddings.closeable() && !textEmbeddings.wasClosed()) {
            textEmbeddings.close();
        }

        // ==================== STEP 8: Autoregressive Decoding ====================
        long step8Start = System.currentTimeMillis();
        log.info("STEP 8: Generating text (max {} tokens, greedy)...", maxTokensConfig);

        // Enable native op timing tracking
        org.nd4j.nativeblas.NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.resetOpTiming();
        nativeOps.setOpTimingEnabled(1, 1);  // enabled=1, detailed=1 (phase-level)
        log.info("Native op timing ENABLED (detailed mode)");
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> presentKeyNames = kvNames.keyNames;
        List<String> presentValueNames = kvNames.valueNames;
        List<String> decoderInputNames = decoder.inputs();

        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxTokensConfig).doSample(false).build());

        List<Integer> generatedTokens = new java.util.ArrayList<>();
        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        INDArray currentEmbeddings = inputsEmbeds;
        INDArray currentInputIds = promptTokenIdsTensor;
        long batchSize = 1;
        long pastSeqLen = 0;

        // Memory-aware chunked decoding setup - query native CUDA for accurate memory info
        int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        long totalGpuMemory = nativeOps.getDeviceTotalMemory(deviceId);
        long initialFreeMemory = nativeOps.getDeviceFreeMemory(deviceId);
        log.info("  GPU device {}: free={}MB / total={}MB",
                deviceId, initialFreeMemory / (1024 * 1024), totalGpuMemory / (1024 * 1024));

        // Also log all GPU devices for multi-GPU awareness
        // Use native getAvailableDevices() which calls cudaGetDeviceCount() directly -
        // Java's getNumberOfDevices() returns only "configured" devices (may be 1 even with 2 GPUs)
        int gpuCount = nativeOps.getAvailableDevices();
        log.info("  CUDA device count (native): {}, Java configured: {}",
                gpuCount, Nd4j.getAffinityManager().getNumberOfDevices());
        for (int g = 0; g < gpuCount; g++) {
            long gFree = nativeOps.getDeviceFreeMemory(g);
            long gTotal = nativeOps.getDeviceTotalMemory(g);
            log.info("    GPU {}: free={}MB / total={}MB{}", g, gFree / (1024*1024), gTotal / (1024*1024),
                    g == deviceId ? " (active)" : "");
        }

        // KV cache memory tracking
        long kvIncrementPerStep = -1;
        long memoryPerStep = -1; // actual measured memory consumption per step
        // Memory pressure uses a HARD FLOOR (OOM safety net) + a dynamic budget approach:
        // - Hard floor: 512MB. Below this, we MUST flush or we'll OOM.
        // - Budget-based: After flush+re-prompt, we measure free memory and compute how many
        //   steps we can afford at the current per-step rate. We flush when we'd have fewer
        //   than MIN_STEPS_REMAINING steps of headroom above the hard floor.
        final long HARD_FLOOR_BYTES = 512L * 1024 * 1024; // 512MB absolute minimum
        final int MIN_STEPS_REMAINING = 20; // flush when only 20 steps of memory left
        final int MEMORY_CHECK_START_STEP = 10; // skip first 10 steps after chunk start (prefill stabilization)
        log.info("  Memory: hard floor={}MB, min steps remaining={}, check starts at step {}",
                HARD_FLOOR_BYTES / (1024*1024), MIN_STEPS_REMAINING, MEMORY_CHECK_START_STEP);

        // Log baseline poolUsed to see model constants vs working memory during decode
        {
            org.bytedeco.javacpp.LongPointer baseUsed = new org.bytedeco.javacpp.LongPointer(1);
            org.bytedeco.javacpp.LongPointer baseReserved = new org.bytedeco.javacpp.LongPointer(1);
            nativeOps.getMemoryPoolStats(deviceId, baseUsed, baseReserved);
            log.info("  Baseline pool: poolUsed={}MB, poolReserved={}MB (model constants + overhead)",
                    baseUsed.get(0) / (1024*1024), baseReserved.get(0) / (1024*1024));
        }

        // Reset native dbClose diagnostics to track per-chunk deallocation
        nativeOps.dbCloseResetDiagnostics();
        long prevPoolUsed = -1; // for per-step delta tracking

        int totalStepsAcrossChunks = 0;
        boolean reachedEndToken = false;
        long prevStepFreeMemory = -1;

        while (!reachedEndToken && totalStepsAcrossChunks < maxTokensConfig) {
            // At the start of each chunk, reset KV cache state
            pastSeqLen = 0;

            for (int step = 0; totalStepsAcrossChunks < maxTokensConfig; step++, totalStepsAcrossChunks++) {

                // Memory check: skip first N steps of each chunk.
                // After flush+re-prompt, free memory is low (~1-2GB) but the pool efficiently
                // reuses working memory within each step. We need several steps to establish
                // the actual per-step consumption rate before making flush decisions.
                if (step >= MEMORY_CHECK_START_STEP) {
                    Nd4j.getExecutioner().commit();
                    long freeMemory = nativeOps.getDeviceFreeMemory(deviceId);

                    // Track actual memory consumption per step
                    if (prevStepFreeMemory > 0 && prevStepFreeMemory > freeMemory) {
                        long consumed = prevStepFreeMemory - freeMemory;
                        if (memoryPerStep < 0) {
                            memoryPerStep = consumed;
                        } else {
                            // Exponential moving average
                            memoryPerStep = (memoryPerStep * 3 + consumed) / 4;
                        }
                    }
                    prevStepFreeMemory = freeMemory;

                    // Check 1: Hard floor - must flush immediately
                    if (freeMemory < HARD_FLOOR_BYTES) {
                        log.warn("MEMORY PRESSURE (hard floor) at step {} (chunk step {}): free={}MB < {}MB - flushing KV cache",
                                totalStepsAcrossChunks, step, freeMemory / (1024 * 1024), HARD_FLOOR_BYTES / (1024 * 1024));
                        break;
                    }

                    // Check 2: Budget-based - flush when running low on steps
                    if (memoryPerStep > 0) {
                        long availableAboveFloor = freeMemory - HARD_FLOOR_BYTES;
                        long stepsRemaining = availableAboveFloor / memoryPerStep;
                        if (stepsRemaining < MIN_STEPS_REMAINING) {
                            log.warn("MEMORY PRESSURE (budget) at step {} (chunk step {}): free={}MB, ~{}MB/step, ~{} steps remaining - flushing KV cache",
                                    totalStepsAcrossChunks, step, freeMemory / (1024 * 1024),
                                    memoryPerStep / (1024 * 1024), stepsRemaining);
                            break;
                        }
                    }

                    if (step % 50 == 0 || step <= 5) {
                        long stepsRemaining = memoryPerStep > 0 ? (freeMemory - HARD_FLOOR_BYTES) / memoryPerStep : -1;
                        org.bytedeco.javacpp.LongPointer pu = new org.bytedeco.javacpp.LongPointer(1);
                        org.bytedeco.javacpp.LongPointer pr = new org.bytedeco.javacpp.LongPointer(1);
                        nativeOps.getMemoryPoolStats(deviceId, pu, pr);
                        long currentPoolUsed = pu.get(0);
                        long poolDelta = prevPoolUsed > 0 ? currentPoolUsed - prevPoolUsed : 0;
                        prevPoolUsed = currentPoolUsed;
                        log.info("  Step {} memory: free={}MB, ~{}MB/step, ~{} steps remaining, poolUsed={}MB (delta={}MB)", totalStepsAcrossChunks,
                                freeMemory / (1024 * 1024), memoryPerStep > 0 ? memoryPerStep / (1024 * 1024) : -1, stepsRemaining,
                                currentPoolUsed / (1024*1024), poolDelta / (1024*1024));
                        // Log all GPUs
                        for (int g = 0; g < gpuCount; g++) {
                            log.info("    GPU {} free: {}MB", g, nativeOps.getDeviceFreeMemory(g) / (1024*1024));
                        }
                        // dbClose diagnostics every 50 steps
                        if (step > 0 && step % 50 == 0) {
                            org.bytedeco.javacpp.LongPointer dbStats = new org.bytedeco.javacpp.LongPointer(9);
                            nativeOps.dbCloseGetDiagnostics(dbStats);
                            log.info("  dbClose stats: total={}, deleted={}, freedMB={}, constant={}, alreadyClosed={}, notOwner={}, noDataBuf={}, deviceErr={}",
                                    dbStats.get(0), dbStats.get(7), dbStats.get(8) / (1024*1024),
                                    dbStats.get(2), dbStats.get(3), dbStats.get(5), dbStats.get(4), dbStats.get(6));
                            nativeOps.dbCloseResetDiagnostics();
                        }
                    }
                } else if (step <= 5) {
                    // Initialize memory tracking and log per-step poolUsed for first steps
                    Nd4j.getExecutioner().commit();
                    prevStepFreeMemory = nativeOps.getDeviceFreeMemory(deviceId);
                    if (step >= 1) {
                        org.bytedeco.javacpp.LongPointer pu2 = new org.bytedeco.javacpp.LongPointer(1);
                        org.bytedeco.javacpp.LongPointer pr2 = new org.bytedeco.javacpp.LongPointer(1);
                        nativeOps.getMemoryPoolStats(deviceId, pu2, pr2);
                        long currentPoolUsed = pu2.get(0);
                        long poolDelta = prevPoolUsed > 0 ? currentPoolUsed - prevPoolUsed : 0;
                        prevPoolUsed = currentPoolUsed;
                        log.info("  Step {} (early): free={}MB, poolUsed={}MB (delta={}MB)",
                                totalStepsAcrossChunks, prevStepFreeMemory / (1024*1024),
                                currentPoolUsed / (1024*1024), poolDelta / (1024*1024));
                    }
                }

                Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
                long currentSeqLen = currentEmbeddings.shape()[1];
                long totalSeqLen = currentSeqLen + pastSeqLen;

                for (String inputName : decoderInputNames) {
                    if (inputName.equals("inputs_embeds")) {
                        decoderInputMap.put(inputName, currentEmbeddings);
                    } else if (inputName.equals("attention_mask")) {
                        decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, batchSize, totalSeqLen));
                    } else if (inputName.equals("_causal_mask")) {
                        decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(currentSeqLen, totalSeqLen));
                    } else if (inputName.equals("input_ids")) {
                        decoderInputMap.put(inputName, currentInputIds);
                    } else if (inputName.equals("position_ids")) {
                        decoderInputMap.put(inputName, Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                                .reshape(1, currentSeqLen).castTo(DataType.LONG));
                    } else if (inputName.startsWith("past_key_values.")) {
                        String presentName = inputName.replace("past_key_values", "present");
                        if (kvCache.containsKey(presentName)) {
                            decoderInputMap.put(inputName, kvCache.get(presentName));
                        } else {
                            decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize));
                        }
                    }
                }

                // CRITICAL: Always ensure inputs_embeds is passed to the decoder
                if (!decoderInputMap.containsKey("inputs_embeds")) {
                    log.warn("inputs_embeds not in decoder.inputs() - adding explicitly (this indicates a graph wiring issue)");
                    decoderInputMap.put("inputs_embeds", currentEmbeddings);
                }

                // Request logits + KV cache outputs
                List<String> allOutputs = new java.util.ArrayList<>();
                allOutputs.add(logitsOutputName);
                allOutputs.addAll(presentKeyNames);
                allOutputs.addAll(presentValueNames);
                Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));

                INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
                if (logitsRaw == null) { log.error("No logits output"); break; }
                INDArray logits = logitsRaw.dup();
                // Close logitsRaw immediately - we have the dup'd copy in logits.
                // logitsRaw was allocated by DSP executor's dup() and is never freed otherwise.
                logitsRaw.setCloseable(true);
                logitsRaw.close();

                // Update KV cache
                // CRITICAL: SameDiff.directExecHelper() sets all placeholder arrays to closeable=false
                // (to prevent accidental closing during execution). But the old KV cache entries were
                // passed as placeholders in this step, so they're now non-closeable. We must restore
                // closeable=true before closing, otherwise old.close() is a silent no-op and the
                // old KV cache arrays accumulate on the GPU (~38MB/step = ~13GB over 350 steps).
                for (String presentName : presentKeyNames) {
                    INDArray pv = decoderOutputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        if (old != null) { old.setCloseable(true); old.close(); }
                    }
                }
                for (String presentName : presentValueNames) {
                    INDArray pv = decoderOutputs.get(presentName);
                    if (pv != null) {
                        INDArray old = kvCache.put(presentName, pv);
                        if (old != null) { old.setCloseable(true); old.close(); }
                    }
                }

                // Compute KV increment after first step
                if (totalStepsAcrossChunks == 0 && !kvCache.isEmpty()) {
                    INDArray sampleKv = kvCache.values().iterator().next();
                    long numHeadsKv = sampleKv.size(1);
                    long headDimKv = sampleKv.size(3);
                    int numLayers = presentKeyNames.size();
                    long bytesPerElement = sampleKv.dataType().width();
                    kvIncrementPerStep = numLayers * 2L * batchSize * numHeadsKv * headDimKv * bytesPerElement;
                    log.info("  KV cache: {} layers, {} heads, {} headDim => {}KB/step",
                            numLayers, numHeadsKv, headDimKv, kvIncrementPerStep / 1024);
                }

                // Sample from last position
                INDArray lastLogits;
                if (logits.rank() == 3) {
                    lastLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all());
                } else {
                    lastLogits = logits.getRow(0);
                }
                INDArray logitsForSampling = lastLogits.dup();
                int nextTokenId = sampler.sample(logitsForSampling);
                generatedTokens.add(nextTokenId);

                String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);
                log.info("  Step {}: '{}' (id={})", totalStepsAcrossChunks, tokenText, nextTokenId);

                if (nextTokenId == eosTokenId || (endOfUtteranceTokenId != null && nextTokenId == endOfUtteranceTokenId)) {
                    log.info("  Stop token at step {}", totalStepsAcrossChunks);
                    reachedEndToken = true;
                    break;
                }

                logits.close();
                logitsForSampling.close();

                // Close per-step input arrays that were created fresh this step.
                // SameDiff.directExecHelper() set them to closeable=false, so restore first.
                for (var entry : decoderInputMap.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();
                    // Skip arrays we still need or that were already closed by KV cache update above
                    if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                    if (name.startsWith("past_key_values.")) continue; // already handled by KV close above
                    if (arr != null && !arr.wasClosed()) {
                        arr.setCloseable(true);
                        arr.close();
                    }
                }
                decoder.clearPlaceholders(false);

                // Get embedding for next token
                INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
                Map<String, INDArray> newEmbedOutputs = embedTokens.output(Map.of(embedInputName, newTokenTensor), embedOutputNames);
                INDArray prevEmbeddings = currentEmbeddings;
                for (var entry : newEmbedOutputs.entrySet()) {
                    currentEmbeddings = entry.getValue();
                }
                // prevEmbeddings was set to closeable=false when passed as placeholder; restore before closing
                if (prevEmbeddings != null && !prevEmbeddings.wasClosed()) {
                    prevEmbeddings.setCloseable(true);
                    prevEmbeddings.close();
                }
                // Close old currentInputIds if it's different from newTokenTensor
                if (currentInputIds != null && currentInputIds != newTokenTensor && !currentInputIds.wasClosed()) {
                    currentInputIds.setCloseable(true);
                    currentInputIds.close();
                }
                currentInputIds = newTokenTensor;
                embedTokens.clearPlaceholders(false);
                pastSeqLen += currentSeqLen;
            }

            // If we broke due to memory pressure (not EOS), flush and re-prompt
            if (!reachedEndToken && totalStepsAcrossChunks < maxTokensConfig) {
                log.info("  Flushing KV cache after {} tokens, re-prompting...", generatedTokens.size());

                // Diagnostic: memory state BEFORE any cleanup
                Nd4j.getExecutioner().commit();
                long preFlushFree = nativeOps.getDeviceFreeMemory(deviceId);
                org.bytedeco.javacpp.LongPointer usedPtr = new org.bytedeco.javacpp.LongPointer(1);
                org.bytedeco.javacpp.LongPointer reservedPtr = new org.bytedeco.javacpp.LongPointer(1);
                nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
                log.info("  PRE-FLUSH: cudaFree={}MB, poolUsed={}MB, poolReserved={}MB",
                        preFlushFree / (1024*1024), usedPtr.get(0) / (1024*1024), reservedPtr.get(0) / (1024*1024));

                // Close all KV cache entries (restore closeable first - SameDiff sets it to false)
                int kvClosed = 0;
                for (var entry : kvCache.entrySet()) {
                    INDArray arr = entry.getValue();
                    if (arr != null && !arr.wasClosed()) {
                        arr.setCloseable(true);
                        arr.close();
                        kvClosed++;
                    }
                }
                kvCache.clear();
                log.info("  Closed {} KV cache arrays", kvClosed);

                Nd4j.getExecutioner().commit();
                long afterKvFree = nativeOps.getDeviceFreeMemory(deviceId);
                nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
                log.info("  AFTER KV CLOSE: cudaFree={}MB, poolUsed={}MB, poolReserved={}MB",
                        afterKvFree / (1024*1024), usedPtr.get(0) / (1024*1024), reservedPtr.get(0) / (1024*1024));

                // Reset session first to close DSP executor and flush its pool
                decoder.clearPlaceholders(false);
                decoder.clearOpInputs();
                decoder.resetSession();
                embedTokens.resetSession();
                Nd4j.getExecutioner().commit();

                long afterReset = nativeOps.getDeviceFreeMemory(deviceId);
                nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
                log.info("  AFTER RESET: cudaFree={}MB, poolUsed={}MB, poolReserved={}MB",
                        afterReset / (1024*1024), usedPtr.get(0) / (1024*1024), reservedPtr.get(0) / (1024*1024));

                // Aggressively reclaim dead arrays via GC + DeallocatorService.
                // Over 300+ decode steps, intermediate arrays accumulate as unreachable
                // Java objects. GC triggers DeallocatorService which calls cudaFreeAsync.
                // Multiple GC passes help since PhantomReferences need 2+ cycles.
                for (int gcPass = 0; gcPass < 3; gcPass++) {
                    System.gc();
                    try { Thread.sleep(200); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                    Nd4j.getExecutioner().commit();
                }
                // Commit syncs CUDA streams, then trim the pool to release
                // freed memory back to the driver.
                Nd4j.getExecutioner().commit();
                nativeOps.trimMemoryPool(deviceId);

                long afterGcTrim = nativeOps.getDeviceFreeMemory(deviceId);
                nativeOps.getMemoryPoolStats(deviceId, usedPtr, reservedPtr);
                log.info("  AFTER GC+TRIM: cudaFree={}MB, poolUsed={}MB, poolReserved={}MB",
                        afterGcTrim / (1024*1024), usedPtr.get(0) / (1024*1024), reservedPtr.get(0) / (1024*1024));

                // Re-build prompt with generated tokens so far
                int[] allTokensSoFar = new int[promptTokenIds.length + generatedTokens.size()];
                System.arraycopy(promptTokenIds, 0, allTokensSoFar, 0, promptTokenIds.length);
                for (int i = 0; i < generatedTokens.size(); i++) {
                    allTokensSoFar[promptTokenIds.length + i] = generatedTokens.get(i);
                }

                // Re-embed the full sequence
                INDArray fullTokensTensor = Nd4j.createFromArray(allTokensSoFar)
                        .reshape(1, allTokensSoFar.length).castTo(DataType.LONG);
                Map<String, INDArray> reEmbedOutputs = embedTokens.output(
                        Map.of(embedInputName, fullTokensTensor), embedOutputNames);
                INDArray textOnlyEmbeddings = null;
                for (var entry : reEmbedOutputs.entrySet()) {
                    textOnlyEmbeddings = entry.getValue().dup();
                }
                embedTokens.clearPlaceholders(false);

                // Re-merge vision embeddings at <image> token positions in the original prompt portion
                INDArray remergedEmbeddings = EmbeddingMerger.mergeEmbeddings(
                        textOnlyEmbeddings, visionEmbeddings, allTokensSoFar, imageTokenId);

                // Close intermediate re-embed arrays - merged result has the data we need
                if (textOnlyEmbeddings != null && textOnlyEmbeddings.closeable() && !textOnlyEmbeddings.wasClosed()) {
                    textOnlyEmbeddings.close();
                }

                // For the new chunk: use the FULL re-embedded sequence as the first step's embeddings
                if (currentEmbeddings != null) currentEmbeddings.close();
                currentEmbeddings = remergedEmbeddings;
                currentInputIds = fullTokensTensor;
                pastSeqLen = 0; // reset - KV cache is empty

                long postFlushFree = nativeOps.getDeviceFreeMemory(deviceId);
                // Log all GPUs to see if managed memory spilled to alternate GPU
                for (int g = 0; g < gpuCount; g++) {
                    long gFree = nativeOps.getDeviceFreeMemory(g);
                    long gTotal = nativeOps.getDeviceTotalMemory(g);
                    log.info("    GPU {}: free={}MB / total={}MB", g, gFree / (1024*1024), gTotal / (1024*1024));
                }
                log.info("  KV cache flushed. Free memory: {}MB. Re-prompting with {} tokens...",
                        postFlushFree / (1024 * 1024), allTokensSoFar.length);
            }
        }

        long step8End = System.currentTimeMillis();
        long step8Total = step8End - step8Start;

        // Flush and print native op timing stats
        nativeOps.setOpTimingEnabled(0, 0);  // disable before flush
        nativeOps.flushOpTiming();
        log.info("========== OP TIMING STATS (Top 30) ==========");
        nativeOps.printOpTimingStats(30);
        log.info("========== OP TIMING THREAD STATS ==========");
        nativeOps.printOpTimingThreadStats();
        // Export CSV for detailed analysis
        String csvPath = "/tmp/vlm-op-timing.csv";
        int csvResult = nativeOps.exportOpTimingCSV(csvPath);
        log.info("Op timing CSV export to {}: {}", csvPath, csvResult == 1 ? "SUCCESS" : "FAILED");
        nativeOps.resetOpTiming();

        // ==================== STEP 9: Output Results ====================
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");
        log.info("TIMING SUMMARY:");
        log.info("  Step 4 (tile):     {}ms", step5Start - step4Start);
        log.info("  Step 5 (preproc):  {}ms", step6Start - step5Start);
        log.info("  Step 6 (vision):   {}ms ({} frames, {}ms/frame)", step7Start - step6Start,
                numFrames, (step7Start - step6Start) / numFrames);
        log.info("  Step 7 (embed):    {}ms", step8Start - step7Start);
        log.info("  Step 8 (decode):   {}ms ({} tokens, {}ms/token)", step8Total,
                generatedTokens.size(), generatedTokens.size() > 0 ? step8Total / generatedTokens.size() : 0);
        log.info("  Total pipeline:    {}ms", step8End - step4Start);

        assertNotNull(generatedText, "Generated text should not be null");
        assertTrue(generatedTokens.size() > 0, "Should have generated at least one token");

        tokenizer.close();
        log.info("Pipeline complete.");

        // Suppress deallocation during JVM exit to avoid SIGABRT from corrupted heap metadata
        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
    }

    /**
     * Simplified SmolDocling test: no tiling, just resize to 512x512 and run one frame.
     * Useful for isolating decoder quality issues from tiling/multi-frame complexity.
     *
     * Run with:
     *   -Dtest=TestVLMModelImportPipeline#testSmolDoclingSimpleNoTiling
     *   -Dvlm.test.pdf.path=/path/to/file.pdf -Dvlm.test.pdf.page=10
     *   -Dvlm.test.maxTokens=200
     */
    @Test
    @DisplayName("SmolDocling simple: single 512x512 frame, no tiling")
    public void testSmolDoclingSimpleNoTiling() throws Exception {
        // STEP 1: Download models
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE.");

        // STEP 2: Load tokenizer
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("STEP 2 DONE: vocab_size={}", tokenizer.getVocabSize());

        // STEP 3: Import models (with SDZ caching)
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // STEP 4: Load image - resize directly to 512x512 (no tiling)
        log.info("STEP 4: Loading and resizing image (no tiling)...");
        int targetSize = 512;
        BufferedImage pdfImage = loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        log.info("  Raw image: {}x{}", pdfImage.getWidth(), pdfImage.getHeight());

        // Resize directly to 512x512
        BufferedImage resized = ImageTiler.resizeImage(pdfImage, targetSize, targetSize);
        log.info("  Resized to: {}x{}", resized.getWidth(), resized.getHeight());

        // STEP 5: Preprocess into tensor
        log.info("STEP 5: Preprocessing single frame...");
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, true);
        INDArray pixelValues = preprocessor.preprocess(resized);
        preprocessor.shutdown();
        // preprocessor gives [1, 3, 512, 512], vision encoder wants [1, 1, 3, 512, 512]
        pixelValues = pixelValues.reshape(1, 1, 3, targetSize, targetSize);
        log.info("STEP 5 DONE: tensor shape={}", java.util.Arrays.toString(pixelValues.shape()));

        // STEP 6: Run vision encoder (single frame)
        log.info("STEP 6: Running vision encoder (1 frame)...");
        Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
        for (String inputName : visionEncoder.inputs()) {
            if (inputName.equals("pixel_values")) {
                visionInputMap.put(inputName, pixelValues);
            } else if (inputName.equals("pixel_attention_mask")) {
                visionInputMap.put(inputName, Nd4j.ones(DataType.BOOL, 1, 1, targetSize, targetSize));
            }
        }
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, visionOutputNames);

        VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
        if (selected == null) throw new RuntimeException("Vision encoder produced no output");
        INDArray visionEmbeddings = selected.tensor.dup();
        log.info("STEP 6 DONE: vision embeddings shape={}, min={}, max={}, mean={}",
                java.util.Arrays.toString(visionEmbeddings.shape()),
                visionEmbeddings.minNumber(), visionEmbeddings.maxNumber(), visionEmbeddings.meanNumber());

        // Release vision encoder memory
        for (var entry : visionOutputs.entrySet()) {
            INDArray arr = entry.getValue();
            if (arr != null && arr.closeable() && !arr.wasClosed()) arr.close();
        }
        pixelValues.close();
        visionEncoder.clearPlaceholders(false);
        visionEncoder.clearOpInputs();
        visionEncoder.resetSession();
        Nd4j.getExecutioner().commit();

        // STEP 7: Build prompt with single image (no grid)
        log.info("STEP 7: Building prompt...");
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        long imageSeqLen = visionEmbeddings.size(1);

        // For a single frame (no tiling), use a simple prompt with imageSeqLen <image> tokens
        StringBuilder imageTokens = new StringBuilder();
        for (int i = 0; i < imageSeqLen; i++) {
            imageTokens.append("<image>");
        }
        String chatPrompt = "<|im_start|>User:" + imageTokens + "Convert this page to docling.<end_of_utterance>\nAssistant:";

        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        int imageCount = ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId);
        log.info("  Prompt: {} tokens, {} <image> tokens, vision seq_len={}", promptTokenIds.length, imageCount, imageSeqLen);

        // Get text embeddings
        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds).reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        Map<String, INDArray> embedOutputs = embedTokens.output(Map.of(embedInputName, promptIdsTensor), embedOutputNames);
        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }
        if (textEmbeddings == null) throw new RuntimeException("embed_tokens produced no output");

        long hiddenSize = visionEmbeddings.shape()[2];
        if (hiddenSize != textEmbeddings.shape()[2]) {
            throw new IllegalStateException("Hidden size mismatch: vision=" + hiddenSize + " text=" + textEmbeddings.shape()[2]);
        }

        // Merge vision embeddings into text at <image> positions
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings,
                promptTokenIds, imageTokenId);
        log.info("STEP 7 DONE: merged embeddings shape={}", java.util.Arrays.toString(inputsEmbeds.shape()));

        // STEP 8: Autoregressive decoding
        int maxTokens = maxTokensConfig;
        log.info("STEP 8: Generating text (max {} tokens, greedy)...", maxTokens);

        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> decoderInputNames = decoder.inputs();
        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxTokens).doSample(false).build());

        List<Integer> generatedTokens = new java.util.ArrayList<>();
        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        INDArray currentEmbeddings = inputsEmbeds;
        INDArray currentInputIds = promptIdsTensor;
        long pastSeqLen = 0;

        for (int step = 0; step < maxTokens; step++) {
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(currentSeqLen, totalSeqLen));
                } else if (inputName.equals("input_ids")) {
                    decoderInputMap.put(inputName, currentInputIds);
                } else if (inputName.equals("position_ids")) {
                    decoderInputMap.put(inputName, Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG));
                } else if (inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                }
            }

            // CRITICAL: Always ensure inputs_embeds is passed to the decoder
            if (!decoderInputMap.containsKey("inputs_embeds")) {
                log.warn("inputs_embeds not in decoder.inputs() - adding explicitly");
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            List<String> allOutputs = new java.util.ArrayList<>();
            allOutputs.add(logitsOutputName);
            allOutputs.addAll(kvNames.keyNames);
            allOutputs.addAll(kvNames.valueNames);
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));

            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) { log.error("No logits output"); break; }
            INDArray logits = logitsRaw.dup();

            for (String pn : kvNames.keyNames) {
                INDArray pv = decoderOutputs.get(pn);
                if (pv != null) { INDArray old = kvCache.put(pn, pv); if (old != null) old.close(); }
            }
            for (String pn : kvNames.valueNames) {
                INDArray pv = decoderOutputs.get(pn);
                if (pv != null) { INDArray old = kvCache.put(pn, pv); if (old != null) old.close(); }
            }

            INDArray lastLogits = logits.rank() == 3
                    ? logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all())
                    : logits.getRow(0);
            INDArray logitsForSampling = lastLogits.dup();
            int nextTokenId = sampler.sample(logitsForSampling);
            generatedTokens.add(nextTokenId);

            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);
            log.info("  Step {}: '{}' (id={})", step, tokenText, nextTokenId);

            if (nextTokenId == eosTokenId || (endOfUtteranceTokenId != null && nextTokenId == endOfUtteranceTokenId)) {
                log.info("  Stop token at step {}", step);
                break;
            }

            logits.close();
            logitsForSampling.close();
            decoder.clearPlaceholders(false);

            INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> newEmbedOutputs = embedTokens.output(Map.of(embedInputName, newTokenTensor), embedOutputNames);
            INDArray prevEmbeddings = currentEmbeddings;
            for (var entry : newEmbedOutputs.entrySet()) {
                currentEmbeddings = entry.getValue();
            }
            if (prevEmbeddings != null) prevEmbeddings.close();
            currentInputIds = newTokenTensor;
            embedTokens.clearPlaceholders(false);
            pastSeqLen += currentSeqLen;
        }

        // STEP 9: Output
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);

        log.info("========================================");
        log.info("GENERATED TEXT ({} tokens, no tiling):", generatedTokens.size());
        log.info("{}", generatedText);
        log.info("========================================");

        assertNotNull(generatedText);
        assertTrue(generatedTokens.size() > 0);
        tokenizer.close();
        log.info("Simple pipeline complete.");

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
    }

    @Test
    @DisplayName("Docling TableFormer: PDF -> Table Detection")
    public void testDoclingTableFormerPipeline() throws Exception {
        log.info("=== Docling TableFormer Pipeline ===");

        // Download model
        var result = VLMModelDownloader.download(VLMModelDownloader.VLMModel.DOCLING_TABLEFORMER_ACCURATE);

        // Import model
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff model = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("TableFormer loaded: {} variables", model.variables().size());
        log.info("Inputs: {}", model.inputs());
        log.info("Outputs: {}", model.outputs());

        // Load and preprocess image
        BufferedImage image = loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel.DOCLING_TABLEFORMER_ACCURATE);
        VLMImagePreprocessor preprocessor = createPreprocessor(VLMModelDownloader.VLMModel.DOCLING_TABLEFORMER_ACCURATE);
        INDArray input = preprocessor.preprocess(image);
        log.info("Input shape: {}", java.util.Arrays.toString(input.shape()));

        // Run inference
        String inputName = findInputVariable(model);
        if (inputName != null) {
            log.info("Running TableFormer inference...");
            Map<String, INDArray> outputs = model.output(
                    Map.of(inputName, input),
                    model.outputs().toArray(new String[0]));

            log.info("=== Table Detection Results ===");
            for (var entry : outputs.entrySet()) {
                INDArray out = entry.getValue();
                log.info("Output '{}': shape={}, min={}, max={}",
                        entry.getKey(),
                        java.util.Arrays.toString(out.shape()),
                        out.minNumber(), out.maxNumber());

                // Show detection results if it looks like bounding boxes or class predictions
                if (out.rank() == 2 || out.rank() == 3) {
                    long numDetections = out.shape()[out.rank() == 3 ? 1 : 0];
                    log.info("  Detected {} potential table regions", numDetections);
                }
            }
        }

        preprocessor.shutdown();
        log.info("=== TableFormer Pipeline Complete ===");
    }

    // ==================== Text Generation with Sampling Tests ====================

    @Test
    @Order(40)
    @DisplayName("Vision encoder output with greedy sampling")
    public void testVisionEncoderWithGreedySampling() throws Exception {
        log.info("=== Vision Encoder with Greedy Sampling ===");

        // Use SigLIP vision encoder (vision-only, doesn't require text input like CLIP)
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.SIGLIP_VISION;
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff visionEncoder = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Vision encoder loaded: {} variables", visionEncoder.variables().size());

        // Create test image and preprocess
        BufferedImage image = loadImageFromPdfOrGenerate(model);
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray imageInput = preprocessor.preprocess(image);
        log.info("Image tensor shape: {}", java.util.Arrays.toString(imageInput.shape()));

        // Run vision encoder
        String inputName = findInputVariable(visionEncoder);
        if (inputName != null) {
            Map<String, INDArray> outputs = visionEncoder.output(
                    Map.of(inputName, imageInput),
                    visionEncoder.outputs().toArray(new String[0]));

            for (var entry : outputs.entrySet()) {
                INDArray logits = entry.getValue();
                log.info("Output '{}': shape={}", entry.getKey(), java.util.Arrays.toString(logits.shape()));

                // If output is classification logits, use greedy sampling
                if (logits.rank() == 2 && logits.size(1) > 1) {
                    INDArray rowLogits = logits.getRow(0);

                    // Greedy sampling (argmax)
                    GreedySampler greedy = new GreedySampler();
                    int greedyToken = greedy.sample(rowLogits);
                    log.info("Greedy sampled token: {}", greedyToken);

                    // Also show top-5 using SamplerUtils
                    INDArray[] topK = SamplerUtils.topK(rowLogits, 5);
                    log.info("Top-5 indices: {}", topK[0]);
                    log.info("Top-5 values: {}", topK[1]);

                    // Verify greedy matches argmax
                    int argmaxToken = SamplerUtils.argmax(rowLogits);
                    assertEquals(argmaxToken, greedyToken, "Greedy should match argmax");
                }
            }
        }

        preprocessor.shutdown();
        log.info("=== Greedy Sampling Complete ===");
    }

    @Test
    @Order(41)
    @DisplayName("Vision output with temperature and top-k sampling")
    public void testVisionOutputWithTemperatureSampling() throws Exception {
        log.info("=== Vision Output with Temperature Sampling ===");

        // Use SigLIP (vision-only model, doesn't require text input like CLIP)
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.SIGLIP_VISION;
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff visionEncoder = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);

        BufferedImage image = loadImageFromPdfOrGenerate(model);
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray imageInput = preprocessor.preprocess(image);

        String inputName = findInputVariable(visionEncoder);
        if (inputName != null) {
            Map<String, INDArray> outputs = visionEncoder.output(
                    Map.of(inputName, imageInput),
                    visionEncoder.outputs().toArray(new String[0]));

            for (var entry : outputs.entrySet()) {
                INDArray logits = entry.getValue();

                if (logits.rank() == 2 && logits.size(1) > 1) {
                    INDArray rowLogits = logits.getRow(0);

                    // Create samplers with different configurations
                    SamplingConfig creativeConfig = SamplingConfig.creative();
                    SamplingConfig preciseConfig = SamplingConfig.precise();

                    Sampler creativeSampler = Sampler.fromConfig(creativeConfig);
                    Sampler preciseSampler = Sampler.fromConfig(preciseConfig);

                    log.info("Creative config: temp={}, topK={}, topP={}",
                            creativeConfig.getTemperature(),
                            creativeConfig.getTopK(),
                            creativeConfig.getTopP());

                    log.info("Precise config: temp={}, topK={}, topP={}",
                            preciseConfig.getTemperature(),
                            preciseConfig.getTopK(),
                            preciseConfig.getTopP());

                    // Sample multiple times to show variation
                    log.info("Creative samples (should vary):");
                    java.util.Set<Integer> creativeTokens = new java.util.HashSet<>();
                    for (int i = 0; i < 10; i++) {
                        int token = creativeSampler.sample(rowLogits);
                        creativeTokens.add(token);
                    }
                    log.info("  Unique tokens sampled: {} out of 10", creativeTokens.size());

                    log.info("Precise samples (should be more consistent):");
                    java.util.Set<Integer> preciseTokens = new java.util.HashSet<>();
                    for (int i = 0; i < 10; i++) {
                        int token = preciseSampler.sample(rowLogits);
                        preciseTokens.add(token);
                    }
                    log.info("  Unique tokens sampled: {} out of 10", preciseTokens.size());

                    // Precise should generally have fewer unique tokens
                    log.info("Precise sampling produced {} unique vs creative's {} unique",
                            preciseTokens.size(), creativeTokens.size());
                }
            }
        }

        preprocessor.shutdown();
        log.info("=== Temperature Sampling Complete ===");
    }

    @Test @DisplayName("Autoregressive generation simulation with sampling")
    public void testAutoregressiveGenerationSimulation() throws Exception {
        log.info("=== Autoregressive Generation Simulation ===");

        // Simulate autoregressive generation with a mock vocabulary
        // This demonstrates the generation loop pattern without needing a full decoder

        int vocabSize = 1000;
        int maxTokens = 20;
        int eosToken = 2;  // End of sequence token

        // Create a sampler for generation
        SamplingConfig config = SamplingConfig.builder()
                .temperature(0.8)
                .topK(50)
                .topP(0.95)
                .maxNewTokens(maxTokens)
                .doSample(true)
                .build();

        Sampler sampler = Sampler.fromConfig(config);
        log.info("Using sampler: {}", sampler.getName());

        // Simulate generation loop
        java.util.List<Integer> generatedTokens = new java.util.ArrayList<>();
        java.util.Random rng = new java.util.Random(42);

        for (int step = 0; step < maxTokens; step++) {
            // Create mock logits (in real use, this comes from decoder)
            INDArray logits = Nd4j.randn(vocabSize).mul(2);

            // Apply repetition penalty if we have previous tokens
            if (!generatedTokens.isEmpty()) {
                int[] prevTokens = generatedTokens.stream().mapToInt(i -> i).toArray();
                logits = SamplerUtils.applyRepetitionPenalty(logits, prevTokens, 1.2);
            }

            // Sample next token
            int nextToken = sampler.sample(logits);
            generatedTokens.add(nextToken);

            // Check for EOS or end_of_utterance (49279)
            if (nextToken == eosToken || nextToken == 49279) {
                log.info("EOS/stop token generated at step {} (token_id={})", step, nextToken);
                break;
            }
        }

        log.info("Generated {} tokens: {}", generatedTokens.size(), generatedTokens);

        // Verify token distribution makes sense
        INDArray probs = SamplerUtils.softmax(Nd4j.randn(vocabSize));
        assertTrue(SamplerUtils.isValidDistribution(probs, 1e-5), "Softmax should produce valid distribution");

        double entropy = SamplerUtils.entropy(probs);
        log.info("Sample distribution entropy: {}", entropy);

        log.info("=== Autoregressive Generation Complete ===");
    }

    @Test
    @DisplayName("Full pipeline: Image -> Vision Encoder -> Sampling -> Token Output")
    public void testFullPipelineWithSampling() throws Exception {
        log.info("=== Full Pipeline with Sampling ===");

        // Load vision model (use SigLIP - vision-only, no text input required)
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.SIGLIP_VISION;
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff visionModel = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Model loaded with {} outputs", visionModel.outputs().size());

        // Process image
        BufferedImage image = loadImageFromPdfOrGenerate(model);
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray imageInput = preprocessor.preprocess(image);

        String inputName = findInputVariable(visionModel);
        if (inputName == null) {
            log.warn("No input variable found, skipping");
            preprocessor.shutdown();
            return;
        }

        // Run vision model
        Map<String, INDArray> outputs = visionModel.output(
                Map.of(inputName, imageInput),
                visionModel.outputs().toArray(new String[0]));

        log.info("=== Sampling Results ===");

        for (var entry : outputs.entrySet()) {
            INDArray output = entry.getValue();
            log.info("Processing output '{}': shape={}", entry.getKey(), java.util.Arrays.toString(output.shape()));

            // Flatten to get logits if needed
            INDArray logits = output.rank() > 1 ? output.getRow(0) : output;

            if (logits.length() > 1) {
                // Compare different sampling strategies
                log.info("--- Sampling Comparison ---");

                // 1. Greedy (argmax)
                int greedyResult = SamplerUtils.argmax(logits);
                log.info("Greedy (argmax): token={}, logit={}",
                        greedyResult, logits.getDouble(greedyResult));

                // 2. Temperature sampling
                CompositeSampler tempSampler = CompositeSampler.withTemperature(0.7);
                int tempResult = tempSampler.sample(logits);
                log.info("Temperature (0.7): token={}, logit={}",
                        tempResult, logits.getDouble(tempResult));

                // 3. Top-K sampling
                CompositeSampler topKSampler = CompositeSampler.withTopK(10);
                int topKResult = topKSampler.sample(logits);
                log.info("Top-K (10): token={}, logit={}",
                        topKResult, logits.getDouble(topKResult));

                // 4. Top-P (nucleus) sampling
                CompositeSampler topPSampler = CompositeSampler.withTopP(0.9);
                int topPResult = topPSampler.sample(logits);
                log.info("Top-P (0.9): token={}, logit={}",
                        topPResult, logits.getDouble(topPResult));

                // 5. Combined sampling
                CompositeSampler combinedSampler = CompositeSampler.builder()
                        .temperature(0.8)
                        .topK(50)
                        .topP(0.95)
                        .seed(42L)
                        .build();
                int combinedResult = combinedSampler.sample(logits);
                log.info("Combined (temp=0.8, topK=50, topP=0.95): token={}, logit={}",
                        combinedResult, logits.getDouble(combinedResult));

                // Get probability distribution
                INDArray probs = SamplerUtils.softmax(logits);
                log.info("Probability at greedy token: {}", probs.getDouble(greedyResult));
                log.info("Entropy of distribution: {}", SamplerUtils.entropy(probs));

                // Show top-5 tokens with probabilities
                INDArray[] topK = SamplerUtils.topK(logits, 5);
                log.info("Top-5 predictions:");
                for (int i = 0; i < 5; i++) {
                    int idx = topK[0].getInt(i);
                    double logit = topK[1].getDouble(i);
                    double prob = probs.getDouble(idx);
                    log.info("  #{}: token={}, logit={}, prob={}", i + 1, idx, logit, prob);
                }
            }
        }

        preprocessor.shutdown();
        log.info("=== Full Pipeline Complete ===");
    }

    @Test
    @Order(44)
    @DisplayName("Batch sampling from vision outputs")
    public void testBatchSamplingFromVisionOutputs() throws Exception {
        log.info("=== Batch Sampling from Vision Outputs ===");

        // Use SigLIP (vision-only model, doesn't require text input like CLIP)
        VLMModelDownloader.VLMModel model = VLMModelDownloader.VLMModel.SIGLIP_VISION;
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff visionEncoder = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);

        // Create batch of images (use same image repeated for simplicity)
        BufferedImage image = loadImageFromPdfOrGenerate(model);
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray singleImage = preprocessor.preprocess(image);

        // Create batch by stacking
        int batchSize = 4;
        INDArray batchImages = Nd4j.concat(0, singleImage, singleImage, singleImage, singleImage);
        log.info("Batch input shape: {}", java.util.Arrays.toString(batchImages.shape()));

        String inputName = findInputVariable(visionEncoder);
        if (inputName != null) {
            Map<String, INDArray> outputs = visionEncoder.output(
                    Map.of(inputName, batchImages),
                    visionEncoder.outputs().toArray(new String[0]));

            for (var entry : outputs.entrySet()) {
                INDArray output = entry.getValue();
                log.info("Batch output '{}': shape={}", entry.getKey(), java.util.Arrays.toString(output.shape()));

                if (output.rank() == 2 && output.size(0) == batchSize) {
                    // Batch greedy sampling
                    int[] greedyResults = SamplerUtils.argmaxBatch(output);
                    log.info("Batch greedy results: {}", java.util.Arrays.toString(greedyResults));

                    // Batch sampling with sampler
                    GreedySampler greedy = new GreedySampler();
                    int[] sampledResults = greedy.sampleBatch(output);
                    log.info("Batch sampled results: {}", java.util.Arrays.toString(sampledResults));

                    // Results should match for greedy
                    assertArrayEquals(greedyResults, sampledResults, "Batch greedy should match argmax");
                }
            }
        }

        preprocessor.shutdown();
        log.info("=== Batch Sampling Complete ===");
    }

    /**
     * Run PDF through a model. Uses test image if no PDF provided.
     */
    private void runPdfInference(VLMModelDownloader.VLMModel model) throws Exception {
        log.info("=== Testing {} ===", model.getName());

        // Download and import model
        VLMModelDownloader.DownloadResult result = VLMModelDownloader.download(model);
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff sd = importer.runImport(result.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Model loaded: {} variables, {} outputs", sd.variables().size(), sd.outputs().size());

        // Log model structure
        log.info("Model inputs:");
        for (var v : sd.inputs()) {
            log.info("  - {}", v);
        }
        log.info("Model outputs:");
        for (var v : sd.outputs()) {
            log.info("  - {}", v);
        }

        // Get input image (from PDF or generate test pattern)
        BufferedImage image = loadImageFromPdfOrGenerate(model);
        log.info("Image size: {}x{}", image.getWidth(), image.getHeight());

        // Preprocess
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray input = preprocessor.preprocess(image);
        log.info("Input tensor: {}", java.util.Arrays.toString(input.shape()));

        // Find input variable and run inference
        String inputName = findInputVariable(sd);
        if (inputName != null) {
            log.info("Running inference with input: {}", inputName);

            Map<String, INDArray> outputs = sd.output(Map.of(inputName, input), sd.outputs().toArray(new String[0]));

            log.info("=== Model Output ===");
            for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
                INDArray out = entry.getValue();
                log.info("Output '{}': shape={}, dtype={}",
                        entry.getKey(),
                        java.util.Arrays.toString(out.shape()),
                        out.dataType());

                // Show output statistics
                if (out.length() > 0) {
                    log.info("  min={}, max={}, mean={}",
                            out.minNumber(), out.maxNumber(), out.meanNumber());

                    // For classification outputs, show top predictions
                    if (out.rank() == 2 && out.size(1) > 1) {
                        INDArray probs = out.rank() == 2 ? out.getRow(0) : out;
                        int[] topK = findTopK(probs, 5);
                        log.info("  Top-5 class indices: {}", java.util.Arrays.toString(topK));
                    }
                }
            }
            log.info("=== End Model Output ===");
        } else {
            log.warn("Could not find input variable, skipping inference");
            log.info("Available variables:");
            for (var v : sd.variables()) {
                log.info("  - {} (placeholder={})", v.name(), v.isPlaceHolder());
            }
        }

        preprocessor.shutdown();
        log.info("=== {} complete ===", model.getName());
    }

    /**
     * Find top-K indices from a probability/logit array using SamplerUtils.
     * Uses the optimized argmax from the generation utilities.
     */
    private int[] findTopK(INDArray arr, int k) {
        int[] indices = new int[Math.min(k, (int) arr.length())];
        INDArray copy = arr.dup();
        for (int i = 0; i < indices.length; i++) {
            // Use SamplerUtils.argmax for optimized implementation
            indices[i] = SamplerUtils.argmax(copy);
            copy.putScalar(indices[i], Float.NEGATIVE_INFINITY);
        }
        return indices;
    }



    /**
     * Load a single image from PDF (first page or configured page) or generate a test pattern.
     * Automatically saves the rendered image to target/vlm-test-output/ for inspection.
     * For multi-page processing, use {@link #loadPagesToProcess()} instead.
     */
    private BufferedImage loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel model) throws IOException {
        if (pdfPath != null && new File(pdfPath).exists()) {
            log.info("Loading PDF: {}", pdfPath);
            try (PDDocument document = PDDocument.load(new File(pdfPath))) {
                log.info("PDF has {} pages", document.getNumberOfPages());
                PDFRenderer renderer = new PDFRenderer(document);
                int pageToLoad = specificPage >= 0 ? specificPage : 0;
                BufferedImage image = renderer.renderImageWithDPI(pageToLoad, renderDpi, ImageType.RGB);

                // Always save the rendered image for inspection
                String savedPath = savePageImage(image, "pdf_page", pageToLoad);
                log.info("Rendered page saved to: {}", savedPath);

                return image;
            }
        } else {
            log.info("No PDF provided (-Dvlm.test.pdf.path=...), using test pattern");
            return createTestImage(model.getInputWidth(), model.getInputHeight());
        }
    }

    /**
     * Load pages based on configuration:
     * - If vlm.test.pdf.page is set: returns just that single page
     * - If vlm.test.pdf.startPage + maxPages is set: returns pages from startPage
     * - Otherwise: returns all pages from startPage
     *
     * @return List of BufferedImage for pages to process
     * @throws IOException if PDF cannot be read
     */
    private List<BufferedImage> loadPagesToProcess() throws IOException {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            throw new IllegalStateException("No PDF configured. Set -Dvlm.test.pdf.path=/path/to/book.pdf");
        }

        List<BufferedImage> pages = new java.util.ArrayList<>();

        try (PDDocument document = PDDocument.load(new File(pdfPath))) {
            int totalPages = document.getNumberOfPages();
            PDFRenderer renderer = new PDFRenderer(document);

            // Determine which pages to load
            if (specificPage >= 0) {
                // Load single specific page
                if (specificPage >= totalPages) {
                    throw new IllegalArgumentException(
                            String.format("Page %d out of bounds. PDF has %d pages.", specificPage, totalPages));
                }
                log.info("Loading specific page {} of {} (DPI: {})", specificPage + 1, totalPages, renderDpi);
                pages.add(renderer.renderImageWithDPI(specificPage, renderDpi, ImageType.RGB));
            } else {
                // Load multiple pages starting from startPage
                int effectiveStart = Math.min(startPage, totalPages - 1);
                int remainingPages = totalPages - effectiveStart;
                int pagesToLoad = maxPages > 0 ? Math.min(maxPages, remainingPages) : remainingPages;
                log.info("Loading {} pages from PDF starting at page {} (DPI: {})", pagesToLoad, effectiveStart, renderDpi);

                for (int i = 0; i < pagesToLoad; i++) {
                    int pageIdx = effectiveStart + i;
                    log.info("Rendering page {}/{} (index {})", i + 1, pagesToLoad, pageIdx);
                    pages.add(renderer.renderImageWithDPI(pageIdx, renderDpi, ImageType.RGB));
                }
            }
        }

        log.info("Loaded {} page(s) from PDF", pages.size());
        return pages;
    }

    /**
     * Get page indices to process based on configuration.
     *
     * @return List of 0-based page indices to process
     * @throws IOException if PDF cannot be read
     */
    private List<Integer> getPageIndicesToProcess() throws IOException {
        List<Integer> indices = new java.util.ArrayList<>();
        int totalPages = getPdfPageCount();

        if (specificPage >= 0) {
            if (specificPage < totalPages) {
                indices.add(specificPage);
            }
        } else {
            int pagesToProcess = maxPages > 0 ? Math.min(maxPages, totalPages) : totalPages;
            for (int i = 0; i < pagesToProcess; i++) {
                indices.add(i);
            }
        }

        return indices;
    }

    /**
     * Load a specific page from a PDF.
     *
     * @param pageIndex Zero-based page index
     * @return BufferedImage of the page
     * @throws IOException if PDF cannot be read
     * @throws IllegalArgumentException if pageIndex is out of bounds
     */
    private BufferedImage loadPageFromPdf(int pageIndex) throws IOException {
        return loadPageFromPdf(pageIndex, renderDpi);
    }

    /**
     * Load a specific page from a PDF with custom DPI.
     * Automatically saves the rendered image to target/vlm-test-output/ for inspection.
     *
     * @param pageIndex Zero-based page index
     * @param dpi The DPI to render at
     * @return BufferedImage of the page
     * @throws IOException if PDF cannot be read
     * @throws IllegalArgumentException if pageIndex is out of bounds
     */
    private BufferedImage loadPageFromPdf(int pageIndex, int dpi) throws IOException {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            throw new IllegalStateException("No PDF configured. Set -Dvlm.test.pdf.path=/path/to/book.pdf");
        }

        try (PDDocument document = PDDocument.load(new File(pdfPath))) {
            int numPages = document.getNumberOfPages();
            if (pageIndex < 0 || pageIndex >= numPages) {
                throw new IllegalArgumentException(
                        String.format("Page index %d out of bounds. PDF has %d pages.", pageIndex, numPages));
            }

            log.info("Loading page {}/{} from PDF (DPI: {})", pageIndex + 1, numPages, dpi);
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(pageIndex, dpi, ImageType.RGB);

            // Always save the rendered image for inspection
            String savedPath = savePageImage(image, "pdf_page", pageIndex);
            log.info("Rendered page saved to: {}", savedPath);

            return image;
        }
    }

    /**
     * Get the number of pages in the configured PDF.
     *
     * @return Number of pages, or 0 if no PDF configured
     * @throws IOException if PDF cannot be read
     */
    private int getPdfPageCount() throws IOException {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            return 0;
        }

        try (PDDocument document = PDDocument.load(new File(pdfPath))) {
            return document.getNumberOfPages();
        }
    }

    /**
     * Check if a PDF is configured and exists.
     */
    private boolean hasPdf() {
        return pdfPath != null && new File(pdfPath).exists();
    }




    private void logDecoderInputUsage(SameDiff decoder, String inputName) {
        org.nd4j.autodiff.samediff.internal.Variable var = decoder.getVariables().get(inputName);
        if (var == null) {
            log.warn("Decoder input '{}' not found in variables", inputName);
            return;
        }
        log.info("Decoder input '{}' type={}, outputOfOp={}",
                inputName, var.getVariable().getVariableType(), var.getOutputOfOp());
    }

    private void logVariablesContaining(SameDiff decoder, String token) {
        int count = 0;
        for (String name : decoder.getVariables().keySet()) {
            if (name.contains(token)) {
                log.info("Decoder variable match: {}", name);
                count++;
                if (count >= 20) {
                    log.info("Decoder variable match: ... (truncated)");
                    break;
                }
            }
        }
        if (count == 0) {
            log.info("No decoder variables contain '{}'", token);
        }
    }

    private VLMImagePreprocessor createSmolDoclingPreprocessor(int targetSize, boolean normalize) {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(targetSize, targetSize));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(normalize);
        if (normalize) {
            config.setImageMean(new double[]{0.5, 0.5, 0.5});
            config.setImageStd(new double[]{0.5, 0.5, 0.5});
        }
        return VLMImagePreprocessor.fromConfig(config);
    }

    /**
     * Save a BufferedImage to disk for inspection.
     *
     * @param image The image to save
     * @param outputPath Path to save the image (supports .png, .jpg, .bmp)
     * @throws IOException if save fails
     */
    private void saveImage(BufferedImage image, String outputPath) throws IOException {
        File outputFile = new File(outputPath);
        String format = "png";
        if (outputPath.toLowerCase().endsWith(".jpg") || outputPath.toLowerCase().endsWith(".jpeg")) {
            format = "jpg";
        } else if (outputPath.toLowerCase().endsWith(".bmp")) {
            format = "bmp";
        }

        // Ensure parent directory exists
        if (outputFile.getParentFile() != null) {
            outputFile.getParentFile().mkdirs();
        }

        boolean written = ImageIO.write(image, format, outputFile);
        if (written) {
            log.info("Saved image to: {} ({}x{}, format={})",
                    outputFile.getAbsolutePath(), image.getWidth(), image.getHeight(), format);
        } else {
            log.error("Failed to write image to: {}", outputFile.getAbsolutePath());
        }
    }

    /**
     * Save a BufferedImage to the default output directory with auto-generated name.
     *
     * @param image The image to save
     * @param prefix Prefix for the filename
     * @param pageIndex Page index (for naming)
     * @return The saved file path
     * @throws IOException if save fails
     */
    private String savePageImage(BufferedImage image, String prefix, int pageIndex) throws IOException {
        // Use absolute path relative to project root
        String baseDir = System.getProperty("user.dir");
        String outputDir = System.getProperty("vlm.test.output.dir",
                baseDir + File.separator + "target" + File.separator + "vlm-test-output");
        File outputDirFile = new File(outputDir);
        outputDirFile.mkdirs();

        String filename = String.format("%s_page_%03d.png", prefix, pageIndex);
        String outputPath = outputDirFile.getAbsolutePath() + File.separator + filename;
        saveImage(image, outputPath);
        return outputPath;
    }

    /**
     * Save a preprocessed tensor as an image for visual inspection.
     * Reverses the normalization to convert back to viewable RGB.
     *
     * @param tensor The preprocessed tensor [batch, channels, H, W] or [batch, frames, channels, H, W]
     * @param mean Normalization mean that was applied
     * @param std Normalization std that was applied
     * @param filename Output filename (saved to vlm-test-output dir)
     * @return The saved file path
     */
    private String savePreprocessedTensor(INDArray tensor, double[] mean, double[] std, String filename) throws IOException {
        // Handle 5D tensor [batch, frames, channels, H, W] - take first frame
        INDArray img = tensor;
        if (tensor.rank() == 5) {
            img = tensor.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        } else if (tensor.rank() == 4) {
            img = tensor.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        }

        // img is now [channels, H, W]
        int channels = (int) img.size(0);
        int height = (int) img.size(1);
        int width = (int) img.size(2);

        log.info("Saving preprocessed tensor: channels={}, height={}, width={}", channels, height, width);

        // Reverse normalization: original = (normalized * std) + mean
        // Then scale back to 0-255
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r, g, b;

                if (channels >= 3) {
                    // Reverse normalize and rescale
                    double rNorm = img.getDouble(0, y, x);
                    double gNorm = img.getDouble(1, y, x);
                    double bNorm = img.getDouble(2, y, x);

                    // Reverse: pixel = (normalized * std) + mean, then * 255
                    r = (int) Math.max(0, Math.min(255, ((rNorm * std[0]) + mean[0]) * 255));
                    g = (int) Math.max(0, Math.min(255, ((gNorm * std[1]) + mean[1]) * 255));
                    b = (int) Math.max(0, Math.min(255, ((bNorm * std[2]) + mean[2]) * 255));
                } else {
                    // Grayscale
                    double val = img.getDouble(0, y, x);
                    int gray = (int) Math.max(0, Math.min(255, ((val * std[0]) + mean[0]) * 255));
                    r = g = b = gray;
                }

                int rgb = (r << 16) | (g << 8) | b;
                bufferedImage.setRGB(x, y, rgb);
            }
        }

        // Log some pixel values for debugging
        log.info("Preprocessed tensor stats: min={}, max={}, mean={}",
                img.minNumber(), img.maxNumber(), img.meanNumber());

        // Sample center pixel from tensor
        double centerR = img.getDouble(0, height/2, width/2);
        double centerG = img.getDouble(1, height/2, width/2);
        double centerB = img.getDouble(2, height/2, width/2);
        log.info("Center pixel (normalized): R={}, G={}, B={}", centerR, centerG, centerB);

        String baseDir = System.getProperty("user.dir");
        String outputDir = System.getProperty("vlm.test.output.dir",
                baseDir + File.separator + "target" + File.separator + "vlm-test-output");
        new File(outputDir).mkdirs();

        String outputPath = outputDir + File.separator + filename;
        saveImage(bufferedImage, outputPath);
        return outputPath;
    }

    private BufferedImage createTestImage(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (x * 255) / width;
                int g = (y * 255) / height;
                img.setRGB(x, y, (r << 16) | (g << 8) | 128);
            }
        }
        return img;
    }

    private String findInputVariable(SameDiff sd) {
        // Look for common input names
        for (String name : new String[]{"input", "pixel_values", "x", "input_ids", "images"}) {
            if (sd.hasVariable(name)) return name;
        }
        // Fall back to first placeholder
        for (var v : sd.variables()) {
            if (v.isPlaceHolder()) return v.name();
        }
        return null;
    }

    // ==================== Utility Tests ====================

    @Test
    @Order(90)
    @DisplayName("Test model cache management")
    public void testModelCacheManagement() {
        File cacheDir = VLMModelDownloader.getCacheDir();
        assertTrue(cacheDir.exists() || cacheDir.mkdirs(), "Cache directory should exist");

        log.info("Cache directory: {}", cacheDir.getAbsolutePath());

        File[] cachedModels = VLMModelDownloader.listCachedModels();
        log.info("Cached models: {}", cachedModels.length);
        for (File f : cachedModels) {
            log.info("  - {} ({} MB)", f.getName(), f.length() / (1024 * 1024));
        }
    }

    @Test
    @Order(91)
    @DisplayName("Test all model definitions are valid")
    public void testModelDefinitionsValid() {
        for (VLMModelDownloader.VLMModel model : VLMModelDownloader.VLMModel.values()) {
            assertNotNull(model.getName(), "Model name should not be null");
            assertNotNull(model.getUrl(), "Model URL should not be null");
            assertNotNull(model.getFormat(), "Model format should not be null");
            assertTrue(model.getInputWidth() > 0, "Input width should be positive");
            assertTrue(model.getInputHeight() > 0, "Input height should be positive");

            log.info("Model: {} ({}x{}, {})",
                    model.getName(),
                    model.getInputWidth(),
                    model.getInputHeight(),
                    model.getFormat());
        }
    }

    // ==================== Helper Methods ====================

    /**
     * Create a VLMImagePreprocessor configured for the given model.
     */
    private VLMImagePreprocessor createPreprocessor(VLMModelDownloader.VLMModel model) {
        PreprocessorConfig config = new PreprocessorConfig();
        config.setSize(new PreprocessorConfig.ImageSize(model.getInputHeight(), model.getInputWidth()));
        config.setDoRescale(true);
        config.setRescaleFactor(1.0 / 255.0);
        config.setDoNormalize(true);

        // Use ImageNet normalization for most vision models
        config.setImageMean(new double[]{0.485, 0.456, 0.406});
        config.setImageStd(new double[]{0.229, 0.224, 0.225});

        return VLMImagePreprocessor.fromConfig(config);
    }

    /**
     * Create a test input tensor for the given model.
     */
    private INDArray createTestInput(VLMModelDownloader.VLMModel model) {
        BufferedImage testImage = createTestImage(model.getInputWidth(), model.getInputHeight());
        VLMImagePreprocessor preprocessor = createPreprocessor(model);
        INDArray result = preprocessor.preprocess(testImage);
        preprocessor.shutdown();
        return result;
    }

    /**
     * Test that Java image loading + preprocessing matches Python/PIL exactly.
     * Loads the same PNG, resizes to 512x512, normalizes with mean=0.5 std=0.5,
     * and compares pixel values at known positions against Python reference values.
     *
     * Python reference values (PIL LANCZOS resize, (x/255 - 0.5)/0.5 normalize):
     *   R[0,0]=0.811765, G[0,0]=0.764706, B[0,0]=0.592157
     *   R[256,256]=-0.105882, G[256,256]=-0.600000, B[256,256]=-0.741176
     *   R[511,511]=0.333333, G[511,511]=0.176471, B[511,511]=-0.317647
     */
    @Test
    @DisplayName("Image preprocessing: Java vs Python pixel value comparison")
    public void testImagePreprocessingMatchesPython() throws Exception {
        log.info("=== Image Preprocessing Comparison: Java vs Python ===");

        // Load the same image used in the Python reference
        String pdfPath = System.getProperty(PDF_PATH_PROPERTY);
        assertNotNull(pdfPath, "Set -Dvlm.test.pdf.path to run this test");

        File pdfFile = new File(pdfPath);
        if (!pdfFile.isAbsolute()) {
            pdfFile = new File(System.getProperty("user.dir"), pdfPath);
        }
        assertTrue(pdfFile.exists(), "PDF not found: " + pdfFile);

        int pageIndex = Integer.parseInt(System.getProperty(PDF_PAGE_PROPERTY, "10"));
        int dpi = Integer.parseInt(System.getProperty(PDF_DPI_PROPERTY, "150"));

        // Render PDF page to BufferedImage (same as the main pipeline)
        PDDocument doc = PDDocument.load(pdfFile);
        PDFRenderer renderer = new PDFRenderer(doc);
        BufferedImage pdfImage = renderer.renderImageWithDPI(pageIndex, dpi, ImageType.RGB);
        doc.close();
        log.info("Rendered page {}: {}x{}", pageIndex, pdfImage.getWidth(), pdfImage.getHeight());

        // Save the rendered image so we can compare with Python
        File renderedFile = new File(System.getProperty("user.dir") + "/target/vlm-test-output/rendered_page.png");
        renderedFile.getParentFile().mkdirs();
        ImageIO.write(pdfImage, "PNG", renderedFile);
        log.info("Saved rendered page to: {}", renderedFile.getAbsolutePath());

        // Step 1: Resize to 512x512 (squish, no aspect ratio preservation)
        int targetSize = 512;
        BufferedImage resized = ImageTiler.resizeImage(pdfImage, targetSize, targetSize);
        log.info("Java resized: {}x{}", resized.getWidth(), resized.getHeight());

        // Check raw pixel values at corners BEFORE normalization
        int rgb00 = resized.getRGB(0, 0);
        int r00 = (rgb00 >> 16) & 0xFF;
        int g00 = (rgb00 >> 8) & 0xFF;
        int b00 = rgb00 & 0xFF;
        log.info("Java raw pixel [0,0] RGB: [{}, {}, {}]", r00, g00, b00);
        log.info("Python raw pixel [0,0] RGB: [231, 225, 203]");

        int rgb256 = resized.getRGB(256, 256);
        int r256 = (rgb256 >> 16) & 0xFF;
        int g256 = (rgb256 >> 8) & 0xFF;
        int b256 = rgb256 & 0xFF;
        log.info("Java raw pixel [256,256] RGB: [{}, {}, {}]", r256, g256, b256);
        log.info("Python raw pixel [256,256] RGB: [114, 51, 33]");

        int rgb511 = resized.getRGB(511, 511);
        int r511 = (rgb511 >> 16) & 0xFF;
        int g511 = (rgb511 >> 8) & 0xFF;
        int b511 = rgb511 & 0xFF;
        log.info("Java raw pixel [511,511] RGB: [{}, {}, {}]", r511, g511, b511);
        log.info("Python raw pixel [511,511] RGB: [170, 150, 87]");

        // Step 2: Run through the preprocessor (rescale + normalize)
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, true);
        INDArray tensor = preprocessor.preprocess(resized);
        preprocessor.shutdown();
        log.info("Preprocessed tensor: shape={}, dtype={}", java.util.Arrays.toString(tensor.shape()), tensor.dataType());
        log.info("  min={}, max={}, mean={}", tensor.minNumber(), tensor.maxNumber(), tensor.meanNumber());

        // Extract normalized values at the same positions as Python reference
        // tensor shape is [1, 3, 512, 512] = [batch, channel, h, w]
        float jR00 = tensor.getFloat(0, 0, 0, 0);
        float jG00 = tensor.getFloat(0, 1, 0, 0);
        float jB00 = tensor.getFloat(0, 2, 0, 0);
        log.info("Java normalized [0,0]: R={}, G={}, B={}", jR00, jG00, jB00);
        log.info("Python normalized [0,0]: R=0.811765, G=0.764706, B=0.592157");

        float jR256 = tensor.getFloat(0, 0, 256, 256);
        float jG256 = tensor.getFloat(0, 1, 256, 256);
        float jB256 = tensor.getFloat(0, 2, 256, 256);
        log.info("Java normalized [256,256]: R={}, G={}, B={}", jR256, jG256, jB256);
        log.info("Python normalized [256,256]: R=-0.105882, G=-0.600000, B=-0.741176");

        float jR511 = tensor.getFloat(0, 0, 511, 511);
        float jG511 = tensor.getFloat(0, 1, 511, 511);
        float jB511 = tensor.getFloat(0, 2, 511, 511);
        log.info("Java normalized [511,511]: R={}, G={}, B={}", jR511, jG511, jB511);
        log.info("Python normalized [511,511]: R=0.333333, G=0.176471, B=-0.317647");

        // Step 3: Check if the source images even match
        // The Python reference uses /tmp/page10-010.png which was rendered at 150 DPI
        // If that file exists, load it in Java and compare
        File pythonImageFile = new File("/tmp/page10-010.png");
        if (pythonImageFile.exists()) {
            BufferedImage pythonImage = ImageIO.read(pythonImageFile);
            log.info("Python source image: {}x{}", pythonImage.getWidth(), pythonImage.getHeight());
            log.info("Java rendered image: {}x{}", pdfImage.getWidth(), pdfImage.getHeight());

            if (pythonImage.getWidth() == pdfImage.getWidth() && pythonImage.getHeight() == pdfImage.getHeight()) {
                // Same size - compare pixels
                int diffPixels = 0;
                long totalDiff = 0;
                for (int y = 0; y < Math.min(100, pdfImage.getHeight()); y++) {
                    for (int x = 0; x < Math.min(100, pdfImage.getWidth()); x++) {
                        int pRgb = pythonImage.getRGB(x, y);
                        int jRgb = pdfImage.getRGB(x, y);
                        if (pRgb != jRgb) {
                            diffPixels++;
                            totalDiff += Math.abs(((pRgb >> 16) & 0xFF) - ((jRgb >> 16) & 0xFF));
                            totalDiff += Math.abs(((pRgb >> 8) & 0xFF) - ((jRgb >> 8) & 0xFF));
                            totalDiff += Math.abs((pRgb & 0xFF) - (jRgb & 0xFF));
                        }
                    }
                }
                log.info("Source image comparison (first 100x100): diffPixels={}, totalDiff={}", diffPixels, totalDiff);
            } else {
                log.warn("Source images have different sizes - Python: {}x{}, Java: {}x{}",
                        pythonImage.getWidth(), pythonImage.getHeight(), pdfImage.getWidth(), pdfImage.getHeight());
            }

            // Also resize the Python source image with Java's bilinear and compare
            BufferedImage pythonResizedByJava = ImageTiler.resizeImage(pythonImage, targetSize, targetSize);
            int pjR00 = (pythonResizedByJava.getRGB(0, 0) >> 16) & 0xFF;
            int pjG00 = (pythonResizedByJava.getRGB(0, 0) >> 8) & 0xFF;
            int pjB00 = pythonResizedByJava.getRGB(0, 0) & 0xFF;
            log.info("Python image resized by Java [0,0] RGB: [{}, {}, {}]", pjR00, pjG00, pjB00);
            log.info("Python image resized by Python [0,0] RGB: [231, 225, 203]");
            log.info("Java image resized by Java [0,0] RGB: [{}, {}, {}]", r00, g00, b00);
        }

        // Step 4: Compute overall statistics for comparison
        // Python reference: min=-1.000000, max=1.000000, mean=0.174427
        log.info("=== Overall Statistics Comparison ===");
        log.info("Java:   min={}, max={}, mean={}", tensor.minNumber(), tensor.maxNumber(), tensor.meanNumber());
        log.info("Python: min=-1.000000, max=1.000000, mean=0.174427");

        // Step 5: Also load the Python reference binary and compare element-wise
        File refBin = new File("/tmp/python_vision_input_3x512x512.bin");
        if (refBin.exists()) {
            log.info("Loading Python reference tensor from {}", refBin.getAbsolutePath());
            java.io.DataInputStream dis = new java.io.DataInputStream(
                    new java.io.BufferedInputStream(new java.io.FileInputStream(refBin)));
            float[] pythonData = new float[3 * 512 * 512];
            byte[] buf = new byte[4];
            for (int i = 0; i < pythonData.length; i++) {
                dis.readFully(buf);
                // numpy saves in little-endian
                int bits = (buf[0] & 0xFF) | ((buf[1] & 0xFF) << 8) | ((buf[2] & 0xFF) << 16) | ((buf[3] & 0xFF) << 24);
                pythonData[i] = Float.intBitsToFloat(bits);
            }
            dis.close();

            INDArray pythonTensor = Nd4j.create(pythonData, new long[]{1, 3, 512, 512}, 'c');
            log.info("Python tensor loaded: shape={}, min={}, max={}, mean={}",
                    java.util.Arrays.toString(pythonTensor.shape()),
                    pythonTensor.minNumber(), pythonTensor.maxNumber(), pythonTensor.meanNumber());

            // Element-wise difference
            INDArray diff = tensor.sub(pythonTensor);
            double maxAbsDiff = diff.amaxNumber().doubleValue();
            double meanAbsDiff = Nd4j.math.abs(diff).meanNumber().doubleValue();
            double l2Diff = diff.norm2Number().doubleValue();
            log.info("Element-wise difference: maxAbsDiff={}, meanAbsDiff={}, L2={}", maxAbsDiff, meanAbsDiff, l2Diff);

            // Per-channel difference
            for (int c = 0; c < 3; c++) {
                INDArray chanDiff = tensor.get(NDArrayIndex.point(0), NDArrayIndex.point(c), NDArrayIndex.all(), NDArrayIndex.all())
                        .sub(pythonTensor.get(NDArrayIndex.point(0), NDArrayIndex.point(c), NDArrayIndex.all(), NDArrayIndex.all()));
                log.info("Channel {} diff: max={}, mean={}, L2={}",
                        c, chanDiff.amaxNumber(), Nd4j.math.abs(chanDiff).meanNumber(), chanDiff.norm2Number());
            }

            pythonTensor.close();
        } else {
            log.warn("Python reference binary not found at {}. Run test_vision_fixed_input.py first.", refBin.getAbsolutePath());
        }

        tensor.close();
        log.info("=== Image Preprocessing Comparison Complete ===");
    }

    // ==================== Batch Processing Tests ====================

    /**
     * Test batch processing: process multiple PDF pages in parallel with proper tiling.
     *
     * Each page is:
     * 1. Resized and split into tiles (same as single-page pipeline)
     * 2. Vision encoder runs on each frame (tiles + global)
     * 3. Frame embeddings concatenated per page
     * 4. Proper tile-aware prompt built with grid layout
     * 5. Embeddings merged at <image> positions
     * 6. Batched autoregressive decoding across all pages
     *
     * Run with:
     *   -Dtest=TestVLMModelImportPipeline#testSmolDoclingBatchProcessing
     *   -Dvlm.test.pdf.path=/path/to/book.pdf
     *   -Dvlm.test.pdf.startPage=9      (first page, 0-indexed)
     *   -Dvlm.test.pdf.maxPages=2       (number of pages / batch size)
     *   -Dvlm.test.maxTokens=50
     *   -Dvlm.test.maxTiles=9           (max tiles per page, default 9)
     */
    @Test
    @DisplayName("SmolDocling batch: process multiple pages with tiling")
    public void testSmolDoclingBatchProcessing() throws Exception {
        // Skip if no PDF provided
        if (pdfPath == null || !new File(pdfPath).exists()) {
            log.info("Skipping batch test - no PDF provided. Use -Dvlm.test.pdf.path=/path/to/book.pdf");
            return;
        }

        // ==================== STEP 1: Download Models ====================
        log.info("=== BATCH PROCESSING TEST (WITH TILING) ===");
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE: All models downloaded.");

        // ==================== STEP 2: Load Tokenizer ====================
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("STEP 2 DONE: vocab_size={}", tokenizer.getVocabSize());

        // ==================== STEP 3: Import ONNX Models (with SDZ caching) ====================
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];
        log.info("  Vision encoder: {} variables", visionEncoder.variables().size());
        log.info("  Decoder: {} variables", decoder.variables().size());
        log.info("  Embed tokens: {} variables", embedTokens.variables().size());
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Load Pages ====================
        // batchSize = number of pages to decode concurrently. Works with any count >= 1.
        // Each page's frames are processed through the vision encoder in chunks of 2,
        // then all pages are decoded concurrently with batchSize = number of pages.
        if (maxPages <= 0) {
            maxPages = 1; // Default to 1 page
        }
        log.info("STEP 4: Loading pages from PDF (startPage={}, maxPages={})...", startPage, maxPages);
        List<BufferedImage> pages = loadPagesToProcess();
        int batchSize = pages.size();
        log.info("STEP 4 DONE: Loaded {} pages for batch processing (batchSize={})", batchSize, batchSize);

        if (batchSize < 1) {
            log.error("No pages loaded from PDF startPage={}. Check PDF path and page range.", startPage);
            return;
        }

        // ==================== STEP 5: Tile Each Page (CPU only) ====================
        long step5Start = System.currentTimeMillis();
        int targetSize = 512;
        int effectiveMaxTiles = maxTiles > 0 ? maxTiles : 9;
        log.info("STEP 5: Tiling {} pages (targetSize={}, maxTiles={})...", batchSize, targetSize, effectiveMaxTiles);

        // Per-page data: tiling results (CPU-only, no GPU arrays yet)
        // Use parallel tiling: each page's tile extraction and resize runs concurrently
        int tilingThreads = Math.min(batchSize, Runtime.getRuntime().availableProcessors());
        List<ImageTiler.SplitImageResult> pageSplitResults = new java.util.ArrayList<>();

        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            BufferedImage page = pages.get(pageIdx);
            log.info("  Page {}: raw {}x{}", pageIdx, page.getWidth(), page.getHeight());

            BufferedImage resizedForTiling = ImageTiler.resizeLongestEdge(page, 2048);
            ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLMParallel(
                    resizedForTiling, targetSize, effectiveMaxTiles, tilingThreads);
            pageSplitResults.add(splitResult);

            int numFrames = splitResult.getTotalFrames();
            log.info("  Page {}: {} frames ({} tiles + 1 global)", pageIdx, numFrames, splitResult.getTileCount());
        }
        log.info("STEP 5 DONE: [{}ms] (parallel tiling, {} threads)", System.currentTimeMillis() - step5Start, tilingThreads);

        // ==================== STEP 6: Batched Vision Encoding ====================
        // Process K frames per vision encoder forward pass instead of 1 at a time.
        // This reduces per-frame overhead (session reset, kernel launches, alloc/dealloc).
        int visionChunkSize = 1; // frames per forward pass (model expects batch=1)
        String vcsStr = System.getProperty("vlm.test.visionBatchSize");
        if (vcsStr != null && !vcsStr.isEmpty()) {
            visionChunkSize = Integer.parseInt(vcsStr);
        }

        long step6Start = System.currentTimeMillis();
        int totalFrames = 0;
        for (ImageTiler.SplitImageResult split : pageSplitResults) {
            totalFrames += split.getTotalFrames();
        }
        log.info("STEP 6: Batched vision encoding ({} pages, {} total frames, chunk size {})...",
                batchSize, totalFrames, visionChunkSize);

        String[] encOutputNames = visionEncoder.outputs().toArray(new String[0]);
        boolean hasMaskInput = visionEncoder.getVariable("pixel_attention_mask") != null;

        // Preprocess ALL frames across all pages in parallel, then batch-encode
        // Step 6a: Parallel CPU preprocessing
        int preprocessThreads = Math.min(totalFrames, Runtime.getRuntime().availableProcessors());
        List<INDArray> allFrameTensors = new java.util.ArrayList<>();
        List<INDArray> allFrameMasks = new java.util.ArrayList<>();
        int[] pageFrameOffsets = new int[batchSize + 1];
        int frameCount = 0;

        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            pageFrameOffsets[pageIdx] = frameCount;
            ImageTiler.SplitImageResult split = pageSplitResults.get(pageIdx);

            // Use parallel preprocessing for this page's frames
            final int fTargetSize = targetSize;
            INDArray frameTensor = VisionEncoderUtils.preprocessFramesParallel(
                    split.frames,
                    () -> createSmolDoclingPreprocessor(fTargetSize, true),
                    targetSize, preprocessThreads);

            // Extract individual frames and create masks
            for (int f = 0; f < split.getTotalFrames(); f++) {
                INDArray frame = frameTensor.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(f),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()
                ).reshape(1, 1, 3, targetSize, targetSize).dup();
                allFrameTensors.add(frame);

                ImageTiler.ContentRegion region = split.contentRegions.get(f);
                allFrameMasks.add(ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize));
                frameCount++;
            }
            frameTensor.close();
        }
        pageFrameOffsets[batchSize] = frameCount;
        log.info("  Preprocessed {} frames in parallel ({} threads)", frameCount, preprocessThreads);

        // Step 6b: Batched vision encoder - K frames per forward pass
        INDArray[] frameEmbeddings = new INDArray[totalFrames];
        int chunksProcessed = 0;

        for (int chunkStart = 0; chunkStart < totalFrames; chunkStart += visionChunkSize) {
            int chunkEnd = Math.min(chunkStart + visionChunkSize, totalFrames);
            int chunkSize = chunkEnd - chunkStart;
            long chunkStartMs = System.currentTimeMillis();

            // Stack frames for this chunk: [chunkSize, 1, 3, H, W]
            INDArray[] chunkFrames = new INDArray[chunkSize];
            INDArray[] chunkMasks = new INDArray[chunkSize];
            for (int i = 0; i < chunkSize; i++) {
                chunkFrames[i] = allFrameTensors.get(chunkStart + i);
                chunkMasks[i] = allFrameMasks.get(chunkStart + i);
            }
            INDArray batchedPixelValues = Nd4j.vstack(chunkFrames);
            INDArray batchedMasks = Nd4j.vstack(chunkMasks);

            Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
            visionInputMap.put("pixel_values", batchedPixelValues);
            if (hasMaskInput) {
                visionInputMap.put("pixel_attention_mask", batchedMasks);
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, encOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            if (selected == null) {
                throw new RuntimeException("Vision encoder produced no output for chunk " + chunksProcessed);
            }

            // Split chunk output back into individual frame embeddings
            INDArray chunkOutput = selected.tensor;
            for (int i = 0; i < chunkSize; i++) {
                frameEmbeddings[chunkStart + i] = chunkOutput.get(
                        NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()
                ).reshape(1, chunkOutput.size(1), chunkOutput.size(2)).dup();
            }

            log.info("  Chunk {}: frames [{}-{}), output shape={} [{}ms]",
                    chunksProcessed, chunkStart, chunkEnd,
                    java.util.Arrays.toString(chunkOutput.shape()),
                    System.currentTimeMillis() - chunkStartMs);

            // Cleanup chunk intermediates
            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            // directExecHelper poisons placeholders via setCloseable(false) → setConstant(true).
            // Must undo poisoning before close, otherwise close() is a no-op and the arrays
            // leak to GC → heap corruption → SIGABRT during constant cleanup.
            batchedPixelValues.setCloseable(true);
            batchedPixelValues.close();
            batchedMasks.setCloseable(true);
            batchedMasks.close();
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
            chunksProcessed++;
        }

        // Close individual preprocessed frames (no longer needed)
        for (INDArray ft : allFrameTensors) {
            if (ft != null && !ft.wasClosed()) { ft.setCloseable(true); ft.close(); }
        }
        for (INDArray fm : allFrameMasks) {
            if (fm != null && !fm.wasClosed()) { fm.setCloseable(true); fm.close(); }
        }

        // Concatenate frame embeddings per page: [1, pageSeqLen, hidden]
        List<INDArray> pageVisionEmbeddings = new java.util.ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            int start = pageFrameOffsets[pageIdx];
            int end = pageFrameOffsets[pageIdx + 1];
            int numPageFrames = end - start;

            INDArray pageEmbedding;
            if (numPageFrames == 1) {
                pageEmbedding = frameEmbeddings[start];
            } else {
                INDArray[] pageFrameEmbs = new INDArray[numPageFrames];
                System.arraycopy(frameEmbeddings, start, pageFrameEmbs, 0, numPageFrames);
                pageEmbedding = Nd4j.concat(1, pageFrameEmbs).dup();
                for (INDArray fe : pageFrameEmbs) { if (fe != null && !fe.wasClosed()) fe.close(); }
            }
            pageVisionEmbeddings.add(pageEmbedding);
            log.info("  Page {}: {} frames -> embedding shape={}",
                    pageIdx, numPageFrames, java.util.Arrays.toString(pageEmbedding.shape()));
        }

        long step6Time = System.currentTimeMillis() - step6Start;
        log.info("STEP 6 DONE: {} chunks, {} frames [{}ms total, {}ms/frame avg, {}ms/chunk avg]",
                chunksProcessed, totalFrames, step6Time,
                step6Time / Math.max(1, totalFrames),
                step6Time / Math.max(1, chunksProcessed));

        // Free vision encoder model constants to reclaim GPU memory for decode
        log.info("  Freeing vision encoder model constants...");
        int closedVisionArrays = 0;
        long closedBytes = 0;
        ArrayHolder constantHolder = visionEncoder.getConstantArrays();
        for (String name : new ArrayList<>(constantHolder.arrayNames())) {
            INDArray arr = constantHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedVisionArrays++;
            }
        }
        ArrayHolder varHolder = visionEncoder.getVariablesArrays();
        for (String name : new ArrayList<>(varHolder.arrayNames())) {
            INDArray arr = varHolder.removeArray(name);
            if (arr != null && !arr.wasClosed()) {
                closedBytes += arr.length() * arr.dataType().width();
                arr.data().setConstant(false);
                arr.close();
                closedVisionArrays++;
            }
        }
        Nd4j.getExecutioner().commit();
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPool(
                Nd4j.getAffinityManager().getDeviceForCurrentThread());
        log.info("  Freed {} vision encoder arrays (~{}MB)", closedVisionArrays, closedBytes / (1024 * 1024));
        visionEncoder = null;

        // ==================== STEP 7: Build Prompt and Merge Embeddings Per Page ====================
        long step7Start = System.currentTimeMillis();
        log.info("STEP 7: Building prompts and merging embeddings for each page...");

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);

        // All pages must have same tiling layout for batched processing
        // (otherwise prompts have different lengths -> can't batch)
        ImageTiler.SplitImageResult refSplit = pageSplitResults.get(0);
        int refNumRows = refSplit.numRows;
        int refNumCols = refSplit.numCols;
        int refNumFrames = refSplit.getTotalFrames();
        long refSeqLen = pageVisionEmbeddings.get(0).size(1);
        int imageSeqLenPerFrame = (int) (refSeqLen / refNumFrames);

        // Build prompt with proper tile grid layout
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(refNumRows, refNumCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";

        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("  Prompt: {} tokens, {} <image> tokens (grid: {}x{} + global)",
                promptTokenIds.length, ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId),
                refNumRows, refNumCols);

        // Get text embeddings (same prompt for all pages)
        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(Map.of(embedInputName, promptTokenIdsTensor), embedOutputNames);
        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();
        }

        // Merge vision + text embeddings for each page
        List<INDArray> batchedInputsEmbeds = new java.util.ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            // Verify all pages have compatible tiling
            ImageTiler.SplitImageResult pageSplit = pageSplitResults.get(pageIdx);
            if (pageSplit.numRows != refNumRows || pageSplit.numCols != refNumCols) {
                log.warn("Page {} has different tiling ({}x{}) than page 0 ({}x{}), skipping",
                        pageIdx, pageSplit.numRows, pageSplit.numCols, refNumRows, refNumCols);
                continue;
            }

            INDArray merged = EmbeddingMerger.mergeEmbeddings(textEmbeddings.dup(), pageVisionEmbeddings.get(pageIdx),
                    promptTokenIds, imageTokenId);
            batchedInputsEmbeds.add(merged);
            log.info("  Page {}: merged embeddings shape={}", pageIdx, java.util.Arrays.toString(merged.shape()));
        }

        // Update batch size (may have changed if some pages had incompatible tiling)
        batchSize = batchedInputsEmbeds.size();
        if (batchSize < 1) {
            log.error("No pages with compatible tiling - cannot batch");
            return;
        }

        log.info("STEP 7 DONE: {} pages ready for batched decoding [{}ms]", batchSize, System.currentTimeMillis() - step7Start);

        // ==================== STEP 8: Batched Autoregressive Decoding ====================
        long step8Start = System.currentTimeMillis();
        log.info("STEP 8: Batched decoding ({} pages, max {} tokens)...", batchSize, maxTokensConfig);

        // Stack embeddings for batch: [batchSize, seqLen, hidden]
        INDArray batchedEmbeddings = Nd4j.vstack(batchedInputsEmbeds.toArray(new INDArray[0]));
        log.info("  Batched embeddings shape: {}", java.util.Arrays.toString(batchedEmbeddings.shape()));

        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> presentKeyNames = kvNames.keyNames;
        List<String> presentValueNames = kvNames.valueNames;
        List<String> decoderInputNames = decoder.inputs();
        long hiddenSize = batchedEmbeddings.shape()[2];

        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceTokenId = tokenizer.getTokenId("<end_of_utterance>");
        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxTokensConfig).doSample(false).build());

        // Per-sequence state
        List<List<Integer>> generatedTokens = new java.util.ArrayList<>();
        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < batchSize; i++) {
            generatedTokens.add(new java.util.ArrayList<>());
        }

        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        // Helper thread for overlapping embed tokens computation with KV cache cleanup
        java.util.concurrent.ExecutorService embedExecutor = java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "EmbedTokens-Async");
            t.setDaemon(true);
            return t;
        });
        INDArray currentEmbeddings = batchedEmbeddings;
        long pastSeqLen = 0;

        int stepsCompleted = 0;
        for (int step = 0; step < maxTokensConfig; step++) {
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    INDArray attentionMask = Nd4j.ones(DataType.LONG, batchSize, totalSeqLen);
                    // Zero out attention for finished sequences so decoder ignores them
                    for (int i = 0; i < batchSize; i++) {
                        if (finished[i]) {
                            attentionMask.putRow(i, Nd4j.zeros(DataType.LONG, totalSeqLen));
                        }
                    }
                    decoderInputMap.put(inputName, attentionMask);
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, DecoderUtils.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, Nd4j.tile(posIds, batchSize, 1));
                } else if (inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize));
                    }
                }
            }

            // CRITICAL: Always ensure inputs_embeds is passed to the decoder
            if (!decoderInputMap.containsKey("inputs_embeds")) {
                log.warn("inputs_embeds not in decoder.inputs() - adding explicitly");
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            // Request logits + KV cache outputs
            List<String> allOutputs = new java.util.ArrayList<>();
            allOutputs.add(logitsOutputName);
            allOutputs.addAll(presentKeyNames);
            allOutputs.addAll(presentValueNames);
            // Track poolUsed around the decode call to isolate where growth happens
            long poolBefore = -1, poolAfterDecode = -1, poolAfterClose = -1;
            if (step < 5 || step % 50 == 0) {
                try {
                    org.nd4j.nativeblas.NativeOps nops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
                    org.bytedeco.javacpp.LongPointer uptr = new org.bytedeco.javacpp.LongPointer(1);
                    org.bytedeco.javacpp.LongPointer rptr = new org.bytedeco.javacpp.LongPointer(1);
                    nops.getMemoryPoolStats(0, uptr, rptr);
                    poolBefore = uptr.get() / (1024 * 1024);
                } catch (Exception ignore) {}
            }
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));
            if (step < 5 || step % 50 == 0) {
                try {
                    org.nd4j.nativeblas.NativeOps nops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
                    org.bytedeco.javacpp.LongPointer uptr = new org.bytedeco.javacpp.LongPointer(1);
                    org.bytedeco.javacpp.LongPointer rptr = new org.bytedeco.javacpp.LongPointer(1);
                    nops.getMemoryPoolStats(0, uptr, rptr);
                    poolAfterDecode = uptr.get() / (1024 * 1024);
                } catch (Exception ignore) {}
            }

            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits output at step {}", step);
                break;
            }
            INDArray logits = logitsRaw.dup();
            logitsRaw.setCloseable(true);
            logitsRaw.close();

            // Update KV cache — CRITICAL: must setCloseable(true) before close because
            // directExecHelper() poisoned these arrays with setCloseable(false) when they
            // were passed as placeholders in the previous step.
            int kvViewCount = 0, kvOwnerCount = 0;
            long kvLeakedBytes = 0;
            for (String presentName : presentKeyNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) {
                    if (pv.isView()) kvViewCount++; else kvOwnerCount++;
                    INDArray old = kvCache.put(presentName, pv);
                    if (old != null) {
                        boolean wasView = old.isView();
                        old.setCloseable(true);
                        if (wasView) {
                            kvLeakedBytes += old.data() != null ? old.data().length() * old.data().getElementSize() : 0;
                        }
                        old.close();
                    }
                }
            }
            for (String presentName : presentValueNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) {
                    if (pv.isView()) kvViewCount++; else kvOwnerCount++;
                    INDArray old = kvCache.put(presentName, pv);
                    if (old != null) {
                        boolean wasView = old.isView();
                        old.setCloseable(true);
                        if (wasView) {
                            kvLeakedBytes += old.data() != null ? old.data().length() * old.data().getElementSize() : 0;
                        }
                        old.close();
                    }
                }
            }
            if (step < 5 || step % 50 == 0) {
                try {
                    org.nd4j.nativeblas.NativeOps nops = org.nd4j.nativeblas.NativeOpsHolder.getInstance().getDeviceNativeOps();
                    org.bytedeco.javacpp.LongPointer uptr = new org.bytedeco.javacpp.LongPointer(1);
                    org.bytedeco.javacpp.LongPointer rptr = new org.bytedeco.javacpp.LongPointer(1);
                    nops.getMemoryPoolStats(0, uptr, rptr);
                    poolAfterClose = uptr.get() / (1024 * 1024);
                } catch (Exception ignore) {}
                log.info("  KV cache: views={}, owners={}; pool: before={}MB, afterDecode={}MB, afterClose={}MB (decode delta={}MB, close delta={}MB)",
                        kvViewCount, kvOwnerCount, poolBefore, poolAfterDecode, poolAfterClose,
                        poolAfterDecode - poolBefore, poolAfterClose - poolAfterDecode);
            }

            // Sample from last position for each sequence: [batchSize, seqLen, vocab] -> [batchSize, vocab]
            // CRITICAL: Use .dup() on CUDA views — views from .get() may have stale device buffers
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(), NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all()).dup();
            } else {
                lastLogits = logits;
            }

            // Batch sampling via sampler (uses bulk host transfer for greedy argmax)
            int[] nextTokenIds = sampler.sampleBatch(lastLogits);
            if (lastLogits != logits) lastLogits.close();

            // Record tokens, print as generated with running text, and check for EOS
            boolean allFinished = true;
            for (int i = 0; i < batchSize; i++) {
                if (!finished[i]) {
                    generatedTokens.get(i).add(nextTokenIds[i]);
                    int[] allTokensSoFar = generatedTokens.get(i).stream().mapToInt(Integer::intValue).toArray();
                    String textSoFar = tokenizer.decode(allTokensSoFar, false);
                    String tokenText = tokenizer.decode(new int[]{nextTokenIds[i]}, false);
                    log.info("  Step {}, page {}: '{}' (id={}) | text so far: {}", step, i, tokenText, nextTokenIds[i], textSoFar);
                    if (nextTokenIds[i] == eosTokenId ||
                            (endOfUtteranceTokenId != null && nextTokenIds[i] == endOfUtteranceTokenId)) {
                        finished[i] = true;
                        log.info("  Page {} finished at step {}", i, step);
                    }
                }
                if (!finished[i]) allFinished = false;
            }

            if (allFinished) {
                log.info("  All sequences finished at step {}", step);
                break;
            }

            logits.close();

            // Overlap: start embed tokens computation on helper thread while main thread
            // cleans up decoder inputs and old KV cache entries. embedTokens and decoder are
            // separate SameDiff instances so they can safely run on different threads.
            int[] batchTokenIds = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                batchTokenIds[i] = finished[i] ? eosTokenId : nextTokenIds[i];
            }
            INDArray tokenTensor = Nd4j.createFromArray(batchTokenIds).reshape(batchSize, 1).castTo(DataType.LONG);

            // Submit embed tokens to helper thread (GPU work on helper's CUDA context)
            final INDArray tokenTensorFinal = tokenTensor;
            java.util.concurrent.Future<INDArray> embedFuture = embedExecutor.submit(() -> {
                Map<String, INDArray> newEmbedOutputs = embedTokens.output(Map.of(embedInputName, tokenTensorFinal), embedOutputNames);
                return newEmbedOutputs.values().iterator().next().dup();
            });

            // Main thread: cleanup decoder inputs while embed tokens computes
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                INDArray arr = entry.getValue();
                if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                if (name.startsWith("past_key_values.")) continue;
                if (arr != null && !arr.wasClosed()) {
                    arr.setCloseable(true);
                    arr.close();
                }
            }
            decoder.clearPlaceholders(false);

            // Close prev embeddings (poisoned by decoder's directExecHelper)
            INDArray prevEmbeddings = currentEmbeddings;
            if (prevEmbeddings != batchedEmbeddings && prevEmbeddings != null && !prevEmbeddings.wasClosed()) {
                prevEmbeddings.setCloseable(true);
                prevEmbeddings.close();
            }

            // Wait for embed tokens result
            try {
                currentEmbeddings = embedFuture.get();
            } catch (Exception e) {
                throw new RuntimeException("Embed tokens failed", e);
            }
            // tokenTensor was poisoned by embedTokens.output() directExecHelper()
            if (tokenTensor != null && !tokenTensor.wasClosed()) {
                tokenTensor.setCloseable(true);
                tokenTensor.close();
            }
            embedTokens.clearPlaceholders(false);
            pastSeqLen += currentSeqLen;
            stepsCompleted = step + 1;
        }

        embedExecutor.shutdown();
        long step8End = System.currentTimeMillis();
        long step8Total = step8End - step8Start;

        // ==================== STEP 9: Output Results ====================
        log.info("========================================");
        log.info("BATCH PROCESSING RESULTS ({} pages, starting at page {}):", batchSize, startPage);
        log.info("========================================");

        int totalTokens = 0;
        DocTagsParser docTagsParser = new DocTagsParser();
        for (int i = 0; i < batchSize; i++) {
            int[] tokenIds = generatedTokens.get(i).stream().mapToInt(Integer::intValue).toArray();
            String rawText = tokenizer.decode(tokenIds, false);
            totalTokens += tokenIds.length;
            log.info("PAGE {} (actual page {}, {} tokens):", i, startPage + i, tokenIds.length);
            log.info("  RAW: {}", rawText);
            // Parse DocTags and convert to readable markdown
            DocumentStructure doc = docTagsParser.parse(rawText);
            String markdown = docTagsParser.toMarkdown(doc);
            log.info("  PARSED ({} elements):", doc.getElements().size());
            log.info("{}", markdown);
            log.info("---");
        }

        log.info("========================================");
        log.info("TIMING SUMMARY:");
        log.info("  Total decode time: {}ms", step8Total);
        log.info("  Steps completed: {}", stepsCompleted);
        log.info("  Total tokens generated: {}", totalTokens);
        log.info("  Effective ms/token: {} (batch amortized)", step8Total * batchSize / Math.max(1, totalTokens));
        log.info("  Throughput: {} tokens/sec", totalTokens * 1000.0 / Math.max(1, step8Total));
        log.info("========================================");

        // Cleanup — must restore closeable on all arrays that may have been poisoned by
        // directExecHelper() (sets setCloseable(false) on placeholders → marks buffer constant)
        for (INDArray arr : pageVisionEmbeddings) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        for (INDArray arr : batchedInputsEmbeds) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        // KV cache entries from the LAST decode step are poisoned (used as placeholders)
        for (INDArray arr : kvCache.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
        if (currentEmbeddings != null && currentEmbeddings != batchedEmbeddings && !currentEmbeddings.wasClosed()) {
            currentEmbeddings.setCloseable(true);
            currentEmbeddings.close();
        }
        if (textEmbeddings != null && !textEmbeddings.wasClosed()) { textEmbeddings.setCloseable(true); textEmbeddings.close(); }
        tokenizer.close();

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
        log.info("Batch processing test complete.");
    }

    /**
     * Batched VLM pipeline using new APIs and batched vision encoding.
     *
     * Key improvements over testSmolDoclingBatchProcessing:
     * 1. Vision encoder frames batched in chunks (K frames per forward pass instead of 1)
     * 2. Uses SameDiffMemoryUtils.safeClose() / freeModelArrays() for clean memory management
     * 3. Uses BatchGenerationState with multiple stop tokens
     * 4. Uses ImagePromptBuilder, EmbeddingMerger, DecoderUtils from library
     *
     * Run with:
     *   -Dtest=TestVLMModelImportPipeline#testBatchedVisionEncoderPipeline
     *   -Dvlm.test.pdf.path=/path/to/book.pdf
     *   -Dvlm.test.pdf.startPage=0
     *   -Dvlm.test.pdf.maxPages=2
     *   -Dvlm.test.maxTokens=50
     *   -Dvlm.test.maxTiles=9
     *   -Dvlm.test.visionBatchSize=4   (frames per vision encoder call, default 4)
     */
    @Test
    @DisplayName("Batched vision encoder pipeline with new APIs")
    public void testBatchedVisionEncoderPipeline() throws Exception {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            log.info("Skipping test - no PDF provided. Use -Dvlm.test.pdf.path=/path/to/book.pdf");
            return;
        }

        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);

        int visionBatchSize = 4;
        String vbsStr = System.getProperty("vlm.test.visionBatchSize");
        if (vbsStr != null && !vbsStr.isEmpty()) {
            visionBatchSize = Integer.parseInt(vbsStr);
        }

        // ==================== STEP 1: Download Models ====================
        log.info("=== BATCHED VISION ENCODER PIPELINE ===");
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE.");

        // ==================== STEP 2: Load Tokenizer ====================
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceId = tokenizer.getTokenId("<end_of_utterance>");
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        log.info("STEP 2 DONE: vocab={}, eos={}, endOfUtterance={}, imageToken={}",
                tokenizer.getVocabSize(), eosTokenId,
                endOfUtteranceId != null ? endOfUtteranceId : "N/A", imageTokenId);

        // ==================== STEP 3: Import ONNX Models (with SDZ caching) ====================
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Load & Tile Pages ====================
        if (maxPages <= 0) maxPages = 1;
        log.info("STEP 4: Loading and tiling pages (startPage={}, maxPages={})...", startPage, maxPages);
        List<BufferedImage> pages = loadPagesToProcess();
        int batchSize = pages.size();
        if (batchSize < 1) {
            log.error("No pages loaded.");
            return;
        }

        int targetSize = 512;
        int effectiveMaxTiles = maxTiles > 0 ? maxTiles : 9;

        List<ImageTiler.SplitImageResult> pageSplitResults = new ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            BufferedImage resized = ImageTiler.resizeLongestEdge(pages.get(pageIdx), 2048);
            ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLM(resized, targetSize, effectiveMaxTiles);
            pageSplitResults.add(split);
            log.info("  Page {}: {}x{} -> {} frames ({}x{} grid + global)",
                    pageIdx, pages.get(pageIdx).getWidth(), pages.get(pageIdx).getHeight(),
                    split.getTotalFrames(), split.numRows, split.numCols);
        }
        log.info("STEP 4 DONE.");

        // ==================== STEP 5: Preprocess All Frames ====================
        // Collect all frames from all pages into a flat list, tracking page boundaries
        log.info("STEP 5: Preprocessing frames...");
        long step5Start = System.currentTimeMillis();
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, true);

        // Flat list of all preprocessed frames + their masks, plus page boundary indices
        List<INDArray> allFrameTensors = new ArrayList<>();
        List<INDArray> allFrameMasks = new ArrayList<>();
        int[] pageFrameOffsets = new int[batchSize + 1]; // pageFrameOffsets[i] = start index for page i
        int totalFrames = 0;

        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            pageFrameOffsets[pageIdx] = totalFrames;
            ImageTiler.SplitImageResult split = pageSplitResults.get(pageIdx);

            for (int f = 0; f < split.getTotalFrames(); f++) {
                INDArray frameTensor = preprocessor.preprocess(split.frames.get(f)); // [1, 3, H, W]
                // Reshape to [1, 1, 3, H, W] as vision encoder expects [batch, numImages, C, H, W]
                frameTensor = frameTensor.reshape(1, 1, 3, targetSize, targetSize);
                allFrameTensors.add(frameTensor);

                ImageTiler.ContentRegion region = split.contentRegions.get(f);
                INDArray mask = ImageTiler.createPixelAttentionMask(region.width, region.height, targetSize);
                allFrameMasks.add(mask);
                totalFrames++;
            }
        }
        pageFrameOffsets[batchSize] = totalFrames;
        preprocessor.shutdown();
        log.info("STEP 5 DONE: {} total frames preprocessed [{}ms]",
                totalFrames, System.currentTimeMillis() - step5Start);

        // ==================== STEP 6: Batched Vision Encoding ====================
        long step6Start = System.currentTimeMillis();
        log.info("STEP 6: Batched vision encoding ({} frames, chunk size {})...",
                totalFrames, visionBatchSize);

        String[] encOutputNames = visionEncoder.outputs().toArray(new String[0]);
        boolean hasMaskInput = visionEncoder.getVariable("pixel_attention_mask") != null;

        // Encode frames in chunks of visionBatchSize
        INDArray[] frameEmbeddings = new INDArray[totalFrames];
        int chunksProcessed = 0;

        for (int chunkStart = 0; chunkStart < totalFrames; chunkStart += visionBatchSize) {
            int chunkEnd = Math.min(chunkStart + visionBatchSize, totalFrames);
            int chunkSize = chunkEnd - chunkStart;
            long chunkStartMs = System.currentTimeMillis();

            // Stack frames for this chunk: [chunkSize, 1, 3, H, W]
            INDArray[] chunkFrames = new INDArray[chunkSize];
            INDArray[] chunkMasks = new INDArray[chunkSize];
            for (int i = 0; i < chunkSize; i++) {
                chunkFrames[i] = allFrameTensors.get(chunkStart + i);
                chunkMasks[i] = allFrameMasks.get(chunkStart + i);
            }
            INDArray batchedPixelValues = Nd4j.vstack(chunkFrames);
            INDArray batchedMasks = Nd4j.vstack(chunkMasks);

            Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
            visionInputMap.put("pixel_values", batchedPixelValues);
            if (hasMaskInput) {
                visionInputMap.put("pixel_attention_mask", batchedMasks);
            }

            Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputMap, encOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            if (selected == null) {
                throw new RuntimeException("Vision encoder produced no output for chunk " + chunksProcessed);
            }

            // selected.tensor shape: [chunkSize, seqLenPerFrame, hiddenDim]
            INDArray chunkOutput = selected.tensor;
            log.info("  Chunk {}: frames [{}-{}), output shape={} [{}ms]",
                    chunksProcessed, chunkStart, chunkEnd,
                    java.util.Arrays.toString(chunkOutput.shape()),
                    System.currentTimeMillis() - chunkStartMs);

            // Split chunk output back into individual frame embeddings
            for (int i = 0; i < chunkSize; i++) {
                frameEmbeddings[chunkStart + i] = chunkOutput.get(
                        NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()
                ).reshape(1, chunkOutput.size(1), chunkOutput.size(2)).dup();
            }

            // Cleanup chunk intermediates
            for (var entry : visionOutputs.entrySet()) {
                INDArray arr = entry.getValue();
                if (arr != null && !arr.wasClosed() && arr.closeable()) arr.close();
            }
            SameDiffMemoryUtils.safeClose(batchedPixelValues);
            SameDiffMemoryUtils.safeClose(batchedMasks);
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            Nd4j.getExecutioner().commit();
            chunksProcessed++;
        }

        // Close individual frame tensors and masks (no longer needed)
        for (INDArray ft : allFrameTensors) SameDiffMemoryUtils.safeClose(ft);
        for (INDArray fm : allFrameMasks) SameDiffMemoryUtils.safeClose(fm);
        allFrameTensors.clear();
        allFrameMasks.clear();

        // Concatenate frame embeddings per page: [1, pageSeqLen, hidden]
        List<INDArray> pageVisionEmbeddings = new ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            int start = pageFrameOffsets[pageIdx];
            int end = pageFrameOffsets[pageIdx + 1];
            int numPageFrames = end - start;

            INDArray pageEmb;
            if (numPageFrames == 1) {
                pageEmb = frameEmbeddings[start];
            } else {
                INDArray[] pageFrameEmbs = new INDArray[numPageFrames];
                System.arraycopy(frameEmbeddings, start, pageFrameEmbs, 0, numPageFrames);
                pageEmb = Nd4j.concat(1, pageFrameEmbs);
                // Close individual frame embeddings after concat
                for (INDArray fe : pageFrameEmbs) SameDiffMemoryUtils.safeClose(fe);
            }
            pageVisionEmbeddings.add(pageEmb);
            log.info("  Page {}: {} frames -> embedding shape={}",
                    pageIdx, numPageFrames, java.util.Arrays.toString(pageEmb.shape()));
        }

        long step6Time = System.currentTimeMillis() - step6Start;
        log.info("STEP 6 DONE: {} chunks, {} frames [{}ms total, {}ms/frame avg, {}ms/chunk avg]",
                chunksProcessed, totalFrames, step6Time,
                step6Time / Math.max(1, totalFrames),
                step6Time / Math.max(1, chunksProcessed));

        // Free vision encoder to reclaim GPU memory
        log.info("  Freeing vision encoder...");
        int freedArrays = SameDiffMemoryUtils.freeModelArrays(visionEncoder);
        Nd4j.getExecutioner().commit();
        // Sync stream 0 (where RELEASE_SPECIAL frees land) + trim pool.
        // Without this, pool-reserved memory starves cudaStreamCreate() on new threads.
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPoolOnStream(
                Nd4j.getAffinityManager().getDeviceForCurrentThread(), null);
        log.info("  Freed {} arrays.", freedArrays);
        visionEncoder = null;

        // ==================== STEP 7: Build Prompt & Merge Embeddings ====================
        long step7Start = System.currentTimeMillis();
        log.info("STEP 7: Building prompt and merging embeddings...");

        // All pages must have same tiling for batched decode (same prompt length)
        ImageTiler.SplitImageResult refSplit = pageSplitResults.get(0);
        int refNumRows = refSplit.numRows;
        int refNumCols = refSplit.numCols;
        int refNumFrames = refSplit.getTotalFrames();
        long refVisionSeqLen = pageVisionEmbeddings.get(0).size(1);
        int imageSeqLenPerFrame = (int) (refVisionSeqLen / refNumFrames);

        // Build prompt using ImagePromptBuilder
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(refNumRows, refNumCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        int promptTokenCount = promptTokenIds.length;
        int imageTokenCount = ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId);
        log.info("  Prompt: {} tokens, {} <image> tokens (grid {}x{} + global, {}tokens/frame)",
                promptTokenCount, imageTokenCount, refNumRows, refNumCols, imageSeqLenPerFrame);

        // Get text embeddings (shared across all pages — same prompt)
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(Map.of(embedInputName, promptTokenIdsTensor), embedOutputNames);
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();

        // Merge vision + text embeddings per page using EmbeddingMerger
        List<INDArray> batchedInputsEmbeds = new ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            ImageTiler.SplitImageResult pageSplit = pageSplitResults.get(pageIdx);
            if (pageSplit.numRows != refNumRows || pageSplit.numCols != refNumCols) {
                log.warn("Page {} has different tiling ({}x{}) than page 0 ({}x{}), skipping",
                        pageIdx, pageSplit.numRows, pageSplit.numCols, refNumRows, refNumCols);
                continue;
            }

            INDArray merged = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings.dup(), pageVisionEmbeddings.get(pageIdx),
                    promptTokenIds, imageTokenId);
            batchedInputsEmbeds.add(merged);
            log.info("  Page {}: merged shape={}", pageIdx, java.util.Arrays.toString(merged.shape()));
        }

        batchSize = batchedInputsEmbeds.size();
        if (batchSize < 1) {
            log.error("No pages with compatible tiling.");
            tokenizer.close();
            return;
        }
        log.info("STEP 7 DONE: {} pages ready [{}ms]", batchSize, System.currentTimeMillis() - step7Start);

        // ==================== STEP 8: Batched Decode with BatchGenerationState ====================
        long step8Start = System.currentTimeMillis();
        log.info("STEP 8: Batched decoding ({} pages, max {} tokens)...", batchSize, maxTokensConfig);

        // Stack per-page embeddings: [batchSize, seqLen, hidden]
        INDArray batchedEmbeddings = Nd4j.vstack(batchedInputsEmbeds.toArray(new INDArray[0]));
        log.info("  Batched embeddings: {}", java.util.Arrays.toString(batchedEmbeddings.shape()));

        // Decoder metadata
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> presentKeyNames = kvNames.keyNames;
        List<String> presentValueNames = kvNames.valueNames;
        List<String> decoderInputNames = decoder.inputs();
        long hiddenSize = batchedEmbeddings.shape()[2];

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(presentKeyNames);
        allOutputNames.addAll(presentValueNames);

        // Initialize BatchGenerationState with multiple stop tokens
        int[] additionalStopTokens = endOfUtteranceId != null ? new int[]{endOfUtteranceId} : new int[0];
        BatchGenerationState state = new BatchGenerationState(batchSize, eosTokenId, additionalStopTokens);

        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        INDArray currentEmbeddings = batchedEmbeddings;
        long pastSeqLen = 0;

        for (int step = 0; step < maxTokensConfig; step++) {
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            // Build decoder inputs
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    INDArray attentionMask = Nd4j.ones(DataType.LONG, batchSize, totalSeqLen);
                    for (int i = 0; i < batchSize; i++) {
                        if (state.isFinished(i)) {
                            attentionMask.putRow(i, Nd4j.zeros(DataType.LONG, totalSeqLen));
                        }
                    }
                    decoderInputMap.put(inputName, attentionMask);
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName,
                            DecoderUtils.buildCausalMask(batchSize, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, Nd4j.tile(posIds, batchSize, 1));
                } else if (inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName,
                                DecoderUtils.createEmptyKvCache(decoder, inputName, batchSize, hiddenSize));
                    }
                }
            }

            if (!decoderInputMap.containsKey("inputs_embeds")) {
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            // Run decoder
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));

            // Extract and dup logits so we can close the raw output
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits at step {}", step);
                break;
            }
            INDArray logits = logitsRaw.dup();
            SameDiffMemoryUtils.safeClose(logitsRaw);

            // Update KV cache with safeClose on old entries
            for (String presentName : presentKeyNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) {
                    INDArray old = kvCache.put(presentName, pv);
                    SameDiffMemoryUtils.safeClose(old);
                }
            }
            for (String presentName : presentValueNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) {
                    INDArray old = kvCache.put(presentName, pv);
                    SameDiffMemoryUtils.safeClose(old);
                }
            }

            // Sample from last position: [batchSize, seqLen, vocab] -> [batchSize, vocab]
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(),
                        NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all()).dup();
            } else {
                lastLogits = logits;
            }

            int[] nextTokenIds = SamplerUtils.argmaxBatch(lastLogits);
            if (lastLogits != logits) SameDiffMemoryUtils.safeClose(lastLogits);

            // Record tokens using BatchGenerationState (handles EOS + custom stop tokens)
            long stepNanos = (System.currentTimeMillis() - step8Start) * 1_000_000L;
            state.recordTokens(nextTokenIds, stepNanos);

            // Log generated tokens
            for (int i = 0; i < batchSize; i++) {
                if (!state.isFinished(i) || state.getTokenCount(i) == state.getTokensForSequence(i).size()) {
                    String tokenText = tokenizer.decode(new int[]{nextTokenIds[i]}, false);
                    if (step < 5 || step % 50 == 0 || state.isFinished(i)) {
                        log.info("  Step {}, page {}: '{}' (id={}){}", step, i, tokenText, nextTokenIds[i],
                                state.isFinished(i) ? " [FINISHED]" : "");
                    }
                }
            }

            if (state.allFinished()) {
                log.info("  All sequences finished at step {}", step);
                break;
            }

            SameDiffMemoryUtils.safeClose(logits);

            // Close per-step inputs (not KV cache, not embeddings)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
            decoder.clearPlaceholders(false);

            // Embed next tokens — single batched call
            int[] batchTokenIds = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                batchTokenIds[i] = state.isFinished(i) ? eosTokenId : nextTokenIds[i];
            }
            INDArray tokenTensor = Nd4j.createFromArray(batchTokenIds)
                    .reshape(batchSize, 1).castTo(DataType.LONG);
            Map<String, INDArray> newEmbedOutputs = embedTokens.output(
                    Map.of(embedInputName, tokenTensor), embedOutputNames);

            INDArray prevEmbeddings = currentEmbeddings;
            currentEmbeddings = newEmbedOutputs.values().iterator().next().dup();

            if (prevEmbeddings != batchedEmbeddings) {
                SameDiffMemoryUtils.safeClose(prevEmbeddings);
            }
            SameDiffMemoryUtils.safeClose(tokenTensor);
            embedTokens.clearPlaceholders(false);
            pastSeqLen += currentSeqLen;
        }

        // Mark remaining as max tokens
        for (int i = 0; i < batchSize; i++) {
            state.markMaxTokens(i);
        }

        long step8Time = System.currentTimeMillis() - step8Start;

        // ==================== STEP 9: Output Results ====================
        log.info("========================================");
        log.info("BATCHED PIPELINE RESULTS ({} pages, starting at page {}):", batchSize, startPage);
        log.info("========================================");

        int totalTokens = 0;
        int[] promptTokenCounts = new int[batchSize];
        String[] texts = new String[batchSize];
        DocTagsParser docTagsParser = new DocTagsParser();

        for (int i = 0; i < batchSize; i++) {
            promptTokenCounts[i] = promptTokenCount;
            int[] tokenIds = state.getTokenArrayForSequence(i);
            texts[i] = tokenizer.decode(tokenIds, false);
            totalTokens += tokenIds.length;

            log.info("PAGE {} (actual page {}, {} tokens, reason={}):", i, startPage + i,
                    tokenIds.length, state.getFinishReasons()[i]);
            log.info("  RAW: {}", texts[i]);

            DocumentStructure doc = docTagsParser.parse(texts[i]);
            String markdown = docTagsParser.toMarkdown(doc);
            log.info("  PARSED ({} elements):", doc.getElements().size());
            log.info("{}", markdown);
            log.info("---");
        }

        GenerationResult[] results = state.buildResults(texts, promptTokenCounts,
                step8Time * 1_000_000L);

        log.info("========================================");
        log.info("TIMING SUMMARY:");
        log.info("  Vision encoding: {}ms ({} chunks of {} frames, {}ms/frame)",
                step6Time, chunksProcessed, visionBatchSize, step6Time / Math.max(1, totalFrames));
        log.info("  Decode time: {}ms", step8Time);
        log.info("  Total tokens generated: {}", totalTokens);
        log.info("  Effective ms/token: {} (batch amortized)",
                step8Time * batchSize / Math.max(1, totalTokens));
        log.info("  Throughput: {} tokens/sec",
                String.format("%.1f", totalTokens * 1000.0 / Math.max(1, step8Time)));
        for (int i = 0; i < batchSize; i++) {
            log.info("  Page {}: {} tokens, {} tok/s, reason={}",
                    i, results[i].getGeneratedTokenCount(),
                    String.format("%.1f", results[i].getTokensPerSecond()),
                    results[i].getFinishReason());
        }
        log.info("========================================");

        // ==================== Cleanup ====================
        for (INDArray arr : pageVisionEmbeddings) SameDiffMemoryUtils.safeClose(arr);
        for (INDArray arr : batchedInputsEmbeds) SameDiffMemoryUtils.safeClose(arr);
        for (INDArray arr : kvCache.values()) SameDiffMemoryUtils.safeClose(arr);
        if (currentEmbeddings != batchedEmbeddings) SameDiffMemoryUtils.safeClose(currentEmbeddings);
        SameDiffMemoryUtils.safeClose(textEmbeddings);
        SameDiffMemoryUtils.safeClose(batchedEmbeddings);
        tokenizer.close();

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
        log.info("Batched vision encoder pipeline test complete.");
    }

    /**
     * Optimized VLM pipeline using all new parallelization components:
     *
     * 1. PipelinedVisionEncoder: overlaps CPU preprocessing of page N+1 with GPU encoding of page N
     * 2. Parallel tiling: splitImageForVLMParallel uses thread pool for tile extraction
     * 3. BatchCompactor: removes finished sequences from batch to reduce computation
     * 4. SpeculativeDecodeLoop: n-gram based speculative decoding for structured outputs (DocTags)
     * 5. Embed/KV cleanup overlap: embedTokens runs on helper thread while main thread cleans KV cache
     * 6. Bulk host transfer: argmaxBatch uses toFloatVector() instead of per-element getFloat()
     *
     * Run with:
     *   -Dtest=TestVLMModelImportPipeline#testOptimizedPipeline
     *   -Dvlm.test.pdf.path=/path/to/book.pdf
     *   -Dvlm.test.pdf.startPage=0
     *   -Dvlm.test.pdf.maxPages=2
     *   -Dvlm.test.maxTokens=200
     *   -Dvlm.test.maxTiles=9
     *   -Dvlm.test.visionBatchSize=2   (frames per vision encoder call, default 2)
     */
    @Test
    @DisplayName("Optimized pipeline: pipelined vision, speculative decode, batch compaction")
    public void testOptimizedPipeline() throws Exception {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            log.info("Skipping test - no PDF provided. Use -Dvlm.test.pdf.path=/path/to/book.pdf");
            return;
        }


       /* Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);*/
        Nd4j.getEnvironment().setLogNativeNDArrayCreation(false);
        int visionChunkSize = 2;
        String vcsStr = System.getProperty("vlm.test.visionBatchSize");
        if (vcsStr != null && !vcsStr.isEmpty()) {
            visionChunkSize = Integer.parseInt(vcsStr);
        }

        // ==================== STEP 1: Download Models ====================
        log.info("=== OPTIMIZED PIPELINE (pipelined vision + speculative decode + batch compaction) ===");
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE.");

        // ==================== STEP 2: Load Tokenizer ====================
        log.info("STEP 2: Loading tokenizer...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceId = tokenizer.getTokenId("<end_of_utterance>");
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        log.info("STEP 2 DONE: vocab={}, eos={}, endOfUtterance={}, imageToken={}",
                tokenizer.getVocabSize(), eosTokenId,
                endOfUtteranceId != null ? endOfUtteranceId : "N/A", imageTokenId);

        // ==================== STEP 3: Import ONNX Models (with SDZ caching) ====================
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        // Enable native workspace so C++ op temporaries use bump allocation instead of
        // glibc malloc. This prevents heap metadata corruption from C++ op buffer overruns.
        visionEncoder.enableWorkspaceMode(8 * 1024 * 1024);
        decoder.enableWorkspaceMode(8 * 1024 * 1024);
        embedTokens.enableWorkspaceMode(8 * 1024 * 1024);
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Load & Tile Pages (Parallel) ====================
        if (maxPages <= 0) maxPages = 1;
        log.info("STEP 4: Loading and tiling pages (startPage={}, maxPages={})...", startPage, maxPages);
        List<BufferedImage> pages = loadPagesToProcess();
        int batchSize = pages.size();
        if (batchSize < 1) {
            log.error("No pages loaded.");
            return;
        }

        int targetSize = 512;
        int effectiveMaxTiles = maxTiles > 0 ? maxTiles : 9;
        int tilingThreads = Math.min(batchSize, Runtime.getRuntime().availableProcessors());

        long step4Start = System.currentTimeMillis();
        List<ImageTiler.SplitImageResult> pageSplitResults = new ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            BufferedImage resized = ImageTiler.resizeLongestEdge(pages.get(pageIdx), 2048);
            ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLMParallel(
                    resized, targetSize, effectiveMaxTiles, tilingThreads);
            pageSplitResults.add(split);
            log.info("  Page {}: {}x{} -> {} frames ({}x{} grid + global)",
                    pageIdx, pages.get(pageIdx).getWidth(), pages.get(pageIdx).getHeight(),
                    split.getTotalFrames(), split.numRows, split.numCols);
        }
        log.info("STEP 4 DONE: [{}ms] (parallel tiling, {} threads)",
                System.currentTimeMillis() - step4Start, tilingThreads);

        // ==================== STEP 5: Pipelined Vision Encoding ====================
        long step5Start = System.currentTimeMillis();
        int totalFrames = 0;
        for (ImageTiler.SplitImageResult split : pageSplitResults) {
            totalFrames += split.getTotalFrames();
        }
        log.info("STEP 5: Pipelined vision encoding ({} pages, {} total frames, chunk size {})...",
                batchSize, totalFrames, visionChunkSize);

        // Use PipelinedVisionEncoder: overlaps CPU preprocessing of page N+1
        // with GPU encoding of page N
        int preprocessThreads = Math.min(4, Runtime.getRuntime().availableProcessors());
        final int fTargetSize = targetSize;
        PipelinedVisionEncoder pipelinedEncoder = new PipelinedVisionEncoder(
                visionEncoder, visionChunkSize, preprocessThreads);

        // Debug/verbose disabled for performance - enable for canary checks
        // Nd4j.getEnvironment().setDebug(true);
        // Nd4j.getEnvironment().setVerbose(true);

        List<INDArray> pageVisionEmbeddings = pipelinedEncoder.encodePipelined(
                pageSplitResults,
                () -> createSmolDoclingPreprocessor(fTargetSize, true),
                targetSize);
        pipelinedEncoder.shutdown();

        long step5Time = System.currentTimeMillis() - step5Start;
        log.info("STEP 5 DONE: {} pages encoded [{}ms total, {}ms/frame avg]",
                batchSize, step5Time, step5Time / Math.max(1, totalFrames));

        // Free vision encoder to reclaim GPU memory
        log.info("  Freeing vision encoder...");
        int freedArrays = SameDiffMemoryUtils.freeModelArrays(visionEncoder);
        Nd4j.getExecutioner().commit();
        // Sync stream 0 (where RELEASE_SPECIAL frees land) + trim pool.
        // Without this, pool-reserved memory starves cudaStreamCreate() on new threads.
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPoolOnStream(
                Nd4j.getAffinityManager().getDeviceForCurrentThread(), null);
        log.info("  Freed {} arrays.", freedArrays);
        visionEncoder = null;

        // ==================== STEP 6: Build Prompt & Merge Embeddings ====================
        long step6Start = System.currentTimeMillis();
        log.info("STEP 6: Building prompt and merging embeddings...");

        // All pages must have same tiling for batched decode
        ImageTiler.SplitImageResult refSplit = pageSplitResults.get(0);
        int refNumRows = refSplit.numRows;
        int refNumCols = refSplit.numCols;
        int refNumFrames = refSplit.getTotalFrames();
        long refVisionSeqLen = pageVisionEmbeddings.get(0).size(1);
        int imageSeqLenPerFrame = (int) (refVisionSeqLen / refNumFrames);

        String imagePrompt = ImagePromptBuilder.buildImagePromptString(refNumRows, refNumCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("  Prompt: {} tokens, {} <image> tokens (grid {}x{} + global, {} tokens/frame)",
                promptTokenIds.length,
                ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId),
                refNumRows, refNumCols, imageSeqLenPerFrame);

        // Get text embeddings (shared across all pages)
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, promptTokenIdsTensor), embedOutputNames);
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        // Close original DSP output arrays to prevent GPU memory leak
        for (INDArray orig : embedOutputs.values()) SameDiffMemoryUtils.safeClose(orig);
        embedTokens.clearPlaceholders(true);

        // Merge vision + text per page
        List<INDArray> batchedInputsEmbeds = new ArrayList<>();
        for (int pageIdx = 0; pageIdx < batchSize; pageIdx++) {
            ImageTiler.SplitImageResult pageSplit = pageSplitResults.get(pageIdx);
            if (pageSplit.numRows != refNumRows || pageSplit.numCols != refNumCols) {
                log.warn("Page {} has different tiling ({}x{}) vs page 0 ({}x{}), skipping",
                        pageIdx, pageSplit.numRows, pageSplit.numCols, refNumRows, refNumCols);
                continue;
            }
            INDArray merged = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings.dup(), pageVisionEmbeddings.get(pageIdx),
                    promptTokenIds, imageTokenId);
            batchedInputsEmbeds.add(merged);
            log.info("  Page {}: merged shape={}", pageIdx, java.util.Arrays.toString(merged.shape()));
        }

        batchSize = batchedInputsEmbeds.size();
        if (batchSize < 1) {
            log.error("No pages with compatible tiling.");
            tokenizer.close();
            return;
        }
        log.info("STEP 6 DONE: {} pages ready [{}ms]", batchSize, System.currentTimeMillis() - step6Start);

        // ==================== STEP 7: Batched Decode with Speculative Decoding + Batch Compaction ====================
        long step7Start = System.currentTimeMillis();
        log.info("STEP 7: Optimized batched decoding ({} pages, max {} tokens)...", batchSize, maxTokensConfig);
        log.info("  Features: speculative decoding (n-gram), batch compaction, embed/KV overlap");

        INDArray batchedEmbeddings = Nd4j.vstack(batchedInputsEmbeds.toArray(new INDArray[0]));
        log.info("  Batched embeddings: {}", java.util.Arrays.toString(batchedEmbeddings.shape()));

        // Decoder metadata
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> presentKeyNames = kvNames.keyNames;
        List<String> presentValueNames = kvNames.valueNames;
        List<String> decoderInputNames = decoder.inputs();
        long hiddenSize = batchedEmbeddings.shape()[2];

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(presentKeyNames);
        allOutputNames.addAll(presentValueNames);

        // Initialize components
        java.util.Set<Integer> eosTokenIds = new java.util.HashSet<>();
        eosTokenIds.add(eosTokenId);
        if (endOfUtteranceId != null) eosTokenIds.add(endOfUtteranceId);

        NgramSpeculator speculator = new NgramSpeculator(3, 5);
        SpeculativeDecodeLoop specLoop = new SpeculativeDecodeLoop(speculator);
        // Probe whether the model supports multi-token decode (seqLen > 1 with KV cache).
        // Models with internal ONNX Expand ops that create [1,1,seqLen,seqLen] causal masks
        // can't broadcast to [1,1,seqLen,totalSeqLen] and will fail on every speculative attempt.
        // This detects the issue upfront with a single throwaway decode, avoiding 3 wasted attempts.
        specLoop.probeMultiTokenSupport(decoder, decoderInputNames, logitsOutputName,
                kvNames, 1, hiddenSize);
        BatchCompactor compactor = new BatchCompactor(batchSize, 0.25);
        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxTokensConfig).doSample(false).build());

        // Per-sequence state
        List<List<Integer>> generatedTokens = new ArrayList<>();
        boolean[] finished = new boolean[batchSize];
        for (int i = 0; i < batchSize; i++) {
            generatedTokens.add(new ArrayList<>());
        }

        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        INDArray currentEmbeddings = batchedEmbeddings;
        long pastSeqLen = 0;
        int activeBatchSize = batchSize;
        int stepsCompleted = 0;
        int totalTokensGenerated = 0;

        // ===== PHASE 1: PREFILL (Step 0) =====
        {
            long stepStart = System.currentTimeMillis();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            long inputPrepStart = System.currentTimeMillis();
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, activeBatchSize, totalSeqLen));
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName,
                            DecoderUtils.buildCausalMask(activeBatchSize, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, Nd4j.tile(posIds, activeBatchSize, 1));
                } else if (inputName.startsWith("past_key_values.")) {
                    decoderInputMap.put(inputName,
                            DecoderUtils.createEmptyKvCache(decoder, inputName, activeBatchSize, hiddenSize));
                }
            }
            if (!decoderInputMap.containsKey("inputs_embeds")) {
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            long inputPrepTime = System.currentTimeMillis() - inputPrepStart;
            long decoderStart = System.currentTimeMillis();
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));
            long decoderTime = System.currentTimeMillis() - decoderStart;

            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                throw new RuntimeException("No logits from prefill step");
            }
            INDArray logits = logitsRaw.dup();
            SameDiffMemoryUtils.safeClose(logitsRaw);
            log.info("  Step 0 decoder: {}ms, logits shape={} rank={}",
                    decoderTime, java.util.Arrays.toString(logits.shape()), logits.rank());

            // Store KV cache from prefill. MUST dup() since resetSession() below
            // closes all session node outputs including these decoder output arrays.
            for (String presentName : presentKeyNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) kvCache.put(presentName, pv.dup());
            }
            for (String presentName : presentValueNames) {
                INDArray pv = decoderOutputs.get(presentName);
                if (pv != null) kvCache.put(presentName, pv.dup());
            }

            // Sample from last position
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(),
                        NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all()).dup();
            } else {
                lastLogits = logits;
            }
            int[] nextTokenIds = sampler.sampleBatch(lastLogits);
            if (lastLogits != logits) SameDiffMemoryUtils.safeClose(lastLogits);
            SameDiffMemoryUtils.safeClose(logits);

            // Record tokens and check EOS
            for (int ci = 0; ci < activeBatchSize; ci++) {
                int origIdx = compactor.getOriginalIndex(ci);
                if (!finished[origIdx]) {
                    generatedTokens.get(origIdx).add(nextTokenIds[ci]);
                    totalTokensGenerated++;
                    if (eosTokenIds.contains(nextTokenIds[ci])) {
                        finished[origIdx] = true;
                    }
                }
            }

            StringBuilder sb = new StringBuilder();
            for (int ci = 0; ci < activeBatchSize; ci++) {
                int origIdx = compactor.getOriginalIndex(ci);
                String tokenText = tokenizer.decode(new int[]{nextTokenIds[ci]}, false);
                if (ci > 0) sb.append(" | ");
                sb.append(String.format("p%d:'%s'(%d)", origIdx, tokenText, nextTokenIds[ci]));
            }
            log.info("  Step 0 [{}ms total, prep={}ms, decoder={}ms, seqLen={}]: {}",
                    System.currentTimeMillis() - stepStart, inputPrepTime, decoderTime, totalSeqLen, sb);

            pastSeqLen += currentSeqLen;
            stepsCompleted = 1;

            // Embed first decode token for ALL batch items
            int[] firstTokenIds = new int[activeBatchSize];
            for (int ci = 0; ci < activeBatchSize; ci++) {
                firstTokenIds[ci] = nextTokenIds[ci];
            }
            INDArray prefillTokenTensor = Nd4j.createFromArray(firstTokenIds)
                    .reshape(activeBatchSize, 1).castTo(DataType.LONG);
            Map<String, INDArray> prefillEmbedOut = embedTokens.output(
                    Map.of(embedInputName, prefillTokenTensor), embedOutputNames);
            INDArray newEmbed = prefillEmbedOut.values().iterator().next().dup();
            for (INDArray orig : prefillEmbedOut.values()) SameDiffMemoryUtils.safeClose(orig);
            SameDiffMemoryUtils.safeClose(prefillTokenTensor);
            embedTokens.clearPlaceholders(false);
            if (currentEmbeddings != batchedEmbeddings) SameDiffMemoryUtils.safeClose(currentEmbeddings);
            currentEmbeddings = newEmbed;

            // Cleanup step 0 inputs
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
            decoder.clearPlaceholders(false);
        }

        // Check if all sequences finished during prefill
        boolean allFinishedAfterPrefill = true;
        for (boolean f : finished) if (!f) allFinishedAfterPrefill = false;

        if (!allFinishedAfterPrefill) {
            // ===== PHASE 2: DECODE LOOP =====
            // Use growing KV approach (same as batch test) for correctness.
            // The KV cache grows by 1 each step; attention mask grows correspondingly.
            long cachePos = pastSeqLen;

            log.info("  Decode setup: pastSeqLen={}, cachePos={}, kvCache entries={}",
                    pastSeqLen, cachePos, kvCache.size());

            // Reset the decoder session to clear prefill intermediates and compile
            // a fresh native plan for decode shapes. Without this, the slot array cache
            // from prefill (seqLen=679) causes allocation errors at decode (seqLen=1).
            decoder.resetSession();
            Nd4j.getMemoryManager().invokeGc();

            // Helper thread for overlapping embed tokens computation with KV cache cleanup
            java.util.concurrent.ExecutorService embedExecutor = java.util.concurrent.Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "EmbedTokens-Async");
                t.setDaemon(true);
                return t;
            });

            long decodeLoopStart = System.currentTimeMillis();
            for (int step = 1; step < maxTokensConfig; step++) {
                long stepStart = System.currentTimeMillis();

                // ── Speculative decode: try to generate multiple tokens in one decoder call ──
                // Requires batch=1 (per-sequence speculation) and enough history for ngram matching.
                if (activeBatchSize == 1 && step > 5 && !specLoop.isDisabled()) {
                    int origIdx = compactor.getOriginalIndex(0);
                    if (!finished[origIdx] && generatedTokens.get(origIdx).size() >= 6) {
                        List<Integer> tokenSeq = generatedTokens.get(origIdx);
                        int lastToken = tokenSeq.get(tokenSeq.size() - 1);

                        SpeculativeDecodeLoop.SpeculativeStepResult specResult = specLoop.step(
                                tokenSeq, lastToken,
                                embedTokens, embedInputName, embedOutputNames,
                                decoder, decoderInputNames, logitsOutputName,
                                kvNames, kvCache, cachePos,
                                activeBatchSize, hiddenSize, eosTokenIds);

                        if (specResult != null) {
                            int[] acceptedTokens = specResult.getAcceptedTokens();
                            long specStepTime = System.currentTimeMillis() - stepStart;

                            // Record accepted tokens
                            boolean specFinished = false;
                            for (int t : acceptedTokens) {
                                generatedTokens.get(origIdx).add(t);
                                totalTokensGenerated++;
                                if (eosTokenIds.contains(t)) {
                                    finished[origIdx] = true;
                                    specFinished = true;
                                    break;
                                }
                            }

                            // Update KV cache from speculative result
                            Map<String, INDArray> updatedKv = specResult.getUpdatedKvCache();
                            if (updatedKv != null) {
                                for (var entry : updatedKv.entrySet()) {
                                    INDArray old = kvCache.put(entry.getKey(), entry.getValue());
                                    if (old != null) {
                                        old.setCloseable(true);
                                        old.close();
                                    }
                                }
                            }

                            cachePos += specResult.getNewPositions();
                            stepsCompleted = step + 1;

                            if (step < 20 || step % 10 == 0) {
                                StringBuilder tokStr = new StringBuilder();
                                for (int t : acceptedTokens) {
                                    String text = tokenizer.decode(new int[]{t}, false);
                                    tokStr.append("'").append(text).append("'(").append(t).append(") ");
                                }
                                log.info("  Step {} [{}ms, SPECULATIVE {} tokens, cachePos={}]: {}",
                                        step, specStepTime, acceptedTokens.length, cachePos, tokStr.toString().trim());
                            }

                            if (specFinished || specResult.hitEos()) break;

                            // Embed last accepted token for next step
                            INDArray prevEmbed = currentEmbeddings;
                            int lastAccepted = acceptedTokens[acceptedTokens.length - 1];
                            INDArray embedTensor = Nd4j.createFromArray(new int[]{lastAccepted})
                                    .reshape(1, 1).castTo(DataType.LONG);
                            Map<String, INDArray> embedOut = embedTokens.output(
                                    Map.of(embedInputName, embedTensor), embedOutputNames);
                            currentEmbeddings = embedOut.values().iterator().next().dup();
                            for (INDArray orig : embedOut.values()) SameDiffMemoryUtils.safeClose(orig);
                            SameDiffMemoryUtils.safeClose(embedTensor);
                            embedTokens.clearPlaceholders(false);
                            if (prevEmbed != batchedEmbeddings && prevEmbed != null && !prevEmbed.wasClosed()) {
                                SameDiffMemoryUtils.safeClose(prevEmbed);
                            }
                            continue; // Skip normal single-token decode
                        }
                    }
                }

                // ── Normal single-token decode ──
                long currentSeqLen = 1;
                long totalSeqLen = currentSeqLen + cachePos;

                // Build input map fresh each step (growing KV, like batch test)
                Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
                decoderInputMap.put("inputs_embeds", currentEmbeddings);

                // Attention mask: all 1s, grows each step
                INDArray attentionMask = Nd4j.ones(DataType.LONG, activeBatchSize, totalSeqLen);
                for (int bi = 0; bi < activeBatchSize; bi++) {
                    if (finished[bi]) attentionMask.putRow(bi, Nd4j.zeros(DataType.LONG, totalSeqLen));
                }
                decoderInputMap.put("attention_mask", attentionMask);

                if (decoderInputNames.contains("_causal_mask")) {
                    decoderInputMap.put("_causal_mask",
                            DecoderUtils.buildCausalMask(activeBatchSize, currentSeqLen, totalSeqLen));
                }

                // Position IDs
                INDArray posIds = Nd4j.arange(cachePos, cachePos + currentSeqLen)
                        .reshape(1, currentSeqLen).castTo(DataType.LONG);
                if (activeBatchSize > 1) posIds = Nd4j.tile(posIds, activeBatchSize, 1);
                decoderInputMap.put("position_ids", posIds);

                // KV cache from previous step
                for (String inputName : decoderInputNames) {
                    if (inputName.startsWith("past_key_values.")) {
                        String presentName = inputName.replace("past_key_values", "present");
                        if (kvCache.containsKey(presentName)) {
                            decoderInputMap.put(inputName, kvCache.get(presentName));
                        } else {
                            decoderInputMap.put(inputName,
                                    DecoderUtils.createEmptyKvCache(decoder, inputName, activeBatchSize, hiddenSize));
                        }
                    }
                }

                // Run decoder
                long decoderStart = System.currentTimeMillis();
                Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap,
                        allOutputNames.toArray(new String[0]));
                long decoderTime = System.currentTimeMillis() - decoderStart;

                // Extract logits — dup and close original to free GPU memory
                INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
                if (logitsRaw == null) {
                    log.error("No logits at step {}", step);
                    break;
                }
                INDArray logits = logitsRaw.dup();
                logitsRaw.setCloseable(true);
                logitsRaw.close();

                // Update KV cache: close old entries, keep new ones
                for (String name : presentKeyNames) {
                    INDArray pv = decoderOutputs.get(name);
                    if (pv != null) {
                        INDArray old = kvCache.put(name, pv);
                        if (old != null) { old.setCloseable(true); old.close(); }
                    }
                }
                for (String name : presentValueNames) {
                    INDArray pv = decoderOutputs.get(name);
                    if (pv != null) {
                        INDArray old = kvCache.put(name, pv);
                        if (old != null) { old.setCloseable(true); old.close(); }
                    }
                }

                cachePos++;

                // Sample from logits
                INDArray lastLogits;
                if (logits.rank() == 3) {
                    lastLogits = logits.get(NDArrayIndex.all(),
                            NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all()).dup();
                } else if (logits.rank() == 2) {
                    lastLogits = logits;
                } else {
                    lastLogits = logits.reshape(1, logits.length());
                }
                int[] nextTokenIds = sampler.sampleBatch(lastLogits);
                if (lastLogits != logits) lastLogits.close();
                logits.close();

                boolean allFinished = true;
                for (int bi = 0; bi < activeBatchSize; bi++) {
                    if (finished[bi]) continue;
                    int nextTokenId = nextTokenIds[bi];
                    generatedTokens.get(bi).add(nextTokenId);
                    totalTokensGenerated++;

                    if (eosTokenIds.contains(nextTokenId)) {
                        finished[bi] = true;
                        log.info("  Sequence {} finished at step {} ({} tokens)", bi, step, generatedTokens.get(bi).size());
                    }
                }
                for (boolean f : finished) if (!f) allFinished = false;
                stepsCompleted = step + 1;
                if (allFinished) break;

                // Overlap: start embed tokens computation on helper thread while main
                // thread cleans up decoder inputs and old KV cache entries
                int[] tokenIdsForEmbed = new int[activeBatchSize];
                for (int bi = 0; bi < activeBatchSize; bi++) {
                    tokenIdsForEmbed[bi] = finished[bi] ? eosTokenId : nextTokenIds[bi];
                }
                INDArray stepTokenTensor = Nd4j.createFromArray(tokenIdsForEmbed)
                        .reshape(activeBatchSize, 1).castTo(DataType.LONG);

                final INDArray tokenTensorFinal = stepTokenTensor;
                java.util.concurrent.Future<INDArray> embedFuture = embedExecutor.submit(() -> {
                    Map<String, INDArray> newEmbedOutputs = embedTokens.output(
                            Map.of(embedInputName, tokenTensorFinal), embedOutputNames);
                    return newEmbedOutputs.values().iterator().next().dup();
                });

                // Main thread: cleanup decoder inputs while embed tokens computes
                for (var entry : decoderInputMap.entrySet()) {
                    String name = entry.getKey();
                    INDArray arr = entry.getValue();
                    if (name.equals("inputs_embeds") || name.equals("input_ids")) continue;
                    if (name.startsWith("past_key_values.")) continue;
                    if (arr != null && !arr.wasClosed()) {
                        arr.setCloseable(true);
                        arr.close();
                    }
                }
                decoder.clearPlaceholders(false);

                // Close prev embeddings
                INDArray prevEmbed = currentEmbeddings;
                if (prevEmbed != batchedEmbeddings && prevEmbed != null && !prevEmbed.wasClosed()) {
                    prevEmbed.setCloseable(true);
                    prevEmbed.close();
                }

                // Wait for embed tokens result
                try {
                    currentEmbeddings = embedFuture.get();
                } catch (Exception e) {
                    throw new RuntimeException("Embed tokens failed", e);
                }
                if (stepTokenTensor != null && !stepTokenTensor.wasClosed()) {
                    stepTokenTensor.setCloseable(true);
                    stepTokenTensor.close();
                }
                embedTokens.clearPlaceholders(false);

                // Log progress
                if (step < 5 || step % 10 == 0) {
                    StringBuilder tokInfo = new StringBuilder();
                    for (int bi = 0; bi < activeBatchSize; bi++) {
                        if (bi > 0) tokInfo.append(" | ");
                        String tokenText = tokenizer.decode(new int[]{nextTokenIds[bi]}, false);
                        tokInfo.append("b").append(bi).append("='").append(tokenText).append("'(").append(nextTokenIds[bi]).append(")");
                    }
                    log.info("  Step {} [{}ms, decoder={}ms, cachePos={}]: {}",
                            step, System.currentTimeMillis() - stepStart, decoderTime, cachePos, tokInfo);
                }
            }

            embedExecutor.shutdown();
            long decodeLoopTime = System.currentTimeMillis() - decodeLoopStart;
            int decodeSteps = stepsCompleted - 1; // exclude prefill step
            int decodeTokens = totalTokensGenerated; // all tokens come from decode
            log.info("  Decode loop: {}ms for {} steps ({} tokens), avg {}ms/step, {}ms/token (batch amortized)",
                    decodeLoopTime, decodeSteps, decodeTokens,
                    decodeSteps > 0 ? decodeLoopTime / decodeSteps : 0,
                    decodeTokens > 0 ? decodeLoopTime * batchSize / decodeTokens : 0);
        }
        long step7Time = System.currentTimeMillis() - step7Start;

        // ==================== STEP 8: Output Results ====================
        log.info("========================================");
        log.info("OPTIMIZED PIPELINE RESULTS ({} pages, starting at page {}):", batchSize, startPage);
        log.info("========================================");

        DocTagsParser docTagsParser = new DocTagsParser();
        for (int i = 0; i < batchSize; i++) {
            int[] tokenIds = generatedTokens.get(i).stream().mapToInt(Integer::intValue).toArray();
            String rawText = tokenizer.decode(tokenIds, false);
            log.info("PAGE {} (actual page {}, {} tokens):", i, startPage + i, tokenIds.length);
            log.info("  RAW: {}", rawText);
            DocumentStructure doc = docTagsParser.parse(rawText);
            String markdown = docTagsParser.toMarkdown(doc);
            log.info("  PARSED ({} elements):", doc.getElements().size());
            log.info("{}", markdown);
            log.info("---");
        }

        log.info("========================================");
        log.info("TIMING SUMMARY:");
        log.info("  Vision encoding (pipelined): {}ms ({} frames, {}ms/frame)",
                step5Time, totalFrames, step5Time / Math.max(1, totalFrames));
        log.info("  Decode time: {}ms", step7Time);
        log.info("  Steps completed: {}", stepsCompleted);
        log.info("  Total tokens generated: {}", totalTokensGenerated);
        log.info("  Effective ms/token: {} (batch amortized)",
                step7Time * batchSize / Math.max(1, totalTokensGenerated));
        log.info("  Throughput: {} tokens/sec",
                String.format("%.1f", totalTokensGenerated * 1000.0 / Math.max(1, step7Time)));
        log.info("  Speculation stats: {}", specLoop.getStats());
        log.info("  Batch compaction: {} -> {} (compacted={})",
                compactor.getOriginalBatchSize(), compactor.getCurrentBatchSize(), compactor.isCompacted());
        log.info("========================================");

        // ==================== Cleanup ====================
        for (INDArray arr : pageVisionEmbeddings) SameDiffMemoryUtils.safeClose(arr);
        for (INDArray arr : batchedInputsEmbeds) SameDiffMemoryUtils.safeClose(arr);
        for (INDArray arr : kvCache.values()) SameDiffMemoryUtils.safeClose(arr);
        if (currentEmbeddings != batchedEmbeddings) SameDiffMemoryUtils.safeClose(currentEmbeddings);
        SameDiffMemoryUtils.safeClose(textEmbeddings);
        SameDiffMemoryUtils.safeClose(batchedEmbeddings);
        tokenizer.close();

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
        log.info("Optimized pipeline test complete.");
    }

    // ==================== Fixed-Shape Pre-Allocated Decode Test ====================

    /**
     * Fixed-shape decode: pre-allocate all tensors at max sequence length, use causal
     * masking to control which positions are valid. Shapes never change between decode
     * steps, so DynamicShapePlan compiles once and is reused for every step.
     *
     * <p>Key idea: instead of growing the KV cache and input tensors each step,
     * pre-allocate everything at {@code prefill_tokens + max_new_tokens} and fill
     * positions progressively. The causal attention mask prevents attending to
     * unfilled future positions. No KV cache is fed back — the decoder recomputes
     * full attention each step (O(n²)), but all shapes are static.</p>
     *
     * <p>Benefits:</p>
     * <ul>
     *   <li>DynamicShapePlan compiles once → zero recompilation overhead</li>
     *   <li>All buffers pre-allocated → zero malloc during generation</li>
     *   <li>Predictable GPU memory footprint</li>
     *   <li>Can be a single combined graph (vision + embed + decoder)</li>
     * </ul>
     *
     * <p>Run with:</p>
     * <pre>
     *   -Dtest=TestVLMModelImportPipeline#testFixedShapeDecode
     *   -Dvlm.test.maxTokens=20
     * </pre>
     */
    @Test
    @DisplayName("Fixed-shape decode: pre-allocated tensors, causal masking, single plan compilation")
    public void testFixedShapeDecode() throws Exception {
        log.info("=== FIXED-SHAPE DECODE TEST ===");
        log.info("Pre-allocate all tensors at max size, use masking, compile plan once.");

        Nd4j.getEnvironment().setLogNativeNDArrayCreation(false);
        int maxNewTokens = maxTokensConfig > 0 ? maxTokensConfig : 20;

        // ==================== STEP 1: Download & load models ====================
        log.info("STEP 1: Loading models...");
        long step1Start = System.currentTimeMillis();
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        visionEncoder.enableWorkspaceMode(8 * 1024 * 1024);
        decoder.enableWorkspaceMode(8 * 1024 * 1024);
        embedTokens.enableWorkspaceMode(8 * 1024 * 1024);

        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("STEP 1 DONE: {}ms", System.currentTimeMillis() - step1Start);

        // ==================== STEP 2: Vision encode ====================
        log.info("STEP 2: Encoding test image...");
        long step2Start = System.currentTimeMillis();
        int targetSize = 512;
        BufferedImage testImage = createTestImage(targetSize, targetSize);
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, true);
        INDArray pixelValues = preprocessor.preprocess(testImage);
        // SmolDocling expects 5D: [batch, frames, C, H, W]
        if (pixelValues.rank() == 4) {
            pixelValues = pixelValues.reshape(1, pixelValues.size(0), pixelValues.size(1),
                    pixelValues.size(2), pixelValues.size(3));
        }
        log.info("  pixel_values shape: {}", java.util.Arrays.toString(pixelValues.shape()));

        Map<String, INDArray> visionInputs = new HashMap<>();
        visionInputs.put("pixel_values", pixelValues);
        // SmolDocling vision encoder requires pixel_attention_mask — all 1s for valid pixels
        if (visionEncoder.getVariable("pixel_attention_mask") != null) {
            // Shape matches pixel_values spatial dims: [batch, frames, H, W]
            long[] pvShape = pixelValues.shape();
            INDArray pixelMask = Nd4j.ones(DataType.LONG, pvShape[0], pvShape[1], pvShape[3], pvShape[4]);
            visionInputs.put("pixel_attention_mask", pixelMask);
        }
        String[] visionOutputNames = visionEncoder.outputs().toArray(new String[0]);
        Map<String, INDArray> visionOutputs = visionEncoder.output(visionInputs, visionOutputNames);

        // Select the first non-empty output as image embeddings
        INDArray imageEmbeds = null;
        for (String outName : visionOutputNames) {
            INDArray out = visionOutputs.get(outName);
            if (out != null && out.length() > 0) {
                imageEmbeds = out;
                log.info("  Vision output '{}': shape={}", outName, java.util.Arrays.toString(out.shape()));
                break;
            }
        }
        assertNotNull(imageEmbeds, "Vision encoder should produce output");

        // Ensure rank 3: [1, imageSeqLen, hidden]
        if (imageEmbeds.rank() == 2) {
            imageEmbeds = imageEmbeds.reshape(1, imageEmbeds.size(0), imageEmbeds.size(1));
        }
        long imageSeqLen = imageEmbeds.size(1);
        long hiddenSize = imageEmbeds.size(2);
        log.info("  imageEmbeds: [{}, {}, {}]", imageEmbeds.size(0), imageSeqLen, hiddenSize);

        // Detach imageEmbeds from vision encoder session, then free the vision encoder
        // to reclaim ~600MB GPU memory before decoder runs
        imageEmbeds = imageEmbeds.dup();
        visionEncoder.close();
        visionEncoder = null;
        Nd4j.getMemoryManager().purgeCaches();
        log.info("  Vision encoder closed, GPU memory reclaimed");
        log.info("STEP 2 DONE: {}ms", System.currentTimeMillis() - step2Start);

        // ==================== STEP 3: Embed prompt ====================
        log.info("STEP 3: Embedding prompt tokens...");
        long step3Start = System.currentTimeMillis();
        String prompt = "<|im_start|>User:Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(prompt, false).getIds();
        int promptSeqLen = promptTokenIds.length;
        log.info("  Prompt: {} tokens", promptSeqLen);

        // Embed prompt
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptIdsTensor = Nd4j.createFromArray(promptTokenIds).reshape(1, promptSeqLen).castTo(DataType.LONG);
        Map<String, INDArray> embedInputs = Map.of(embedInputName, promptIdsTensor);
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs, embedOutputNames);
        INDArray textEmbeds = embedOutputs.get(embedOutputNames[0]);
        if (textEmbeds.rank() == 2) {
            textEmbeds = textEmbeds.reshape(1, textEmbeds.size(0), textEmbeds.size(1));
        }
        log.info("  textEmbeds: [{}, {}, {}]", textEmbeds.size(0), textEmbeds.size(1), textEmbeds.size(2));
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Pre-allocate fixed-shape buffers ====================
        log.info("STEP 4: Pre-allocating fixed-shape buffers...");
        long step4Start = System.currentTimeMillis();

        long prefillLen = imageSeqLen + promptSeqLen;
        long maxTotalSeq = prefillLen + maxNewTokens;
        log.info("  prefillLen={} (image={} + prompt={}), maxNewTokens={}, maxTotalSeq={}",
                prefillLen, imageSeqLen, promptSeqLen, maxNewTokens, maxTotalSeq);

        // Combined embeddings buffer: [1, maxTotalSeq, hidden]
        // Build by concatenating prefill embeddings with zero padding
        INDArray prefillEmbeds = Nd4j.concat(1, imageEmbeds, textEmbeds).dup();
        INDArray paddingEmbeds = Nd4j.zeros(DataType.FLOAT, 1, maxNewTokens, hiddenSize);
        INDArray embedsBuffer = Nd4j.concat(1, prefillEmbeds, paddingEmbeds);
        SameDiffMemoryUtils.safeClose(prefillEmbeds);
        SameDiffMemoryUtils.safeClose(paddingEmbeds);
        // Free source embeddings now that they're copied into embedsBuffer
        SameDiffMemoryUtils.safeClose(imageEmbeds);
        SameDiffMemoryUtils.safeClose(textEmbeds);
        imageEmbeds = null;
        textEmbeds = null;
        Nd4j.getMemoryManager().purgeCaches();
        log.info("  embedsBuffer shape: {}", java.util.Arrays.toString(embedsBuffer.shape()));

        // Attention mask: [1, maxTotalSeq] — 1 for valid positions, 0 for future
        INDArray onesPrefix = Nd4j.ones(DataType.LONG, 1, prefillLen);
        INDArray zerosSuffix = Nd4j.zeros(DataType.LONG, 1, maxNewTokens);
        INDArray attentionMask = Nd4j.concat(1, onesPrefix, zerosSuffix);
        SameDiffMemoryUtils.safeClose(onesPrefix);
        SameDiffMemoryUtils.safeClose(zerosSuffix);

        // Position IDs: [1, maxTotalSeq] — 0, 1, 2, ...
        INDArray positionIds = Nd4j.arange(maxTotalSeq).reshape(1, maxTotalSeq).castTo(DataType.LONG);

        // Causal mask: [1, 1, maxTotalSeq, maxTotalSeq] — lower triangular
        // Use DecoderUtils.buildCausalMask which handles the mask correctly
        INDArray causalMask = DecoderUtils.buildCausalMask(maxTotalSeq, maxTotalSeq);
        log.info("  causalMask shape: {}", java.util.Arrays.toString(causalMask.shape()));

        // Empty KV cache (seq_len=0) for each layer — fixed shape, never changes
        List<String> decoderInputNames = decoder.inputs();
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);

        Map<String, INDArray> emptyKvCache = new HashMap<>();
        for (String inputName : decoderInputNames) {
            if (inputName.startsWith("past_key_values.")) {
                emptyKvCache.put(inputName, DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
            }
        }
        int numKvLayers = emptyKvCache.size() / 2;

        log.info("  Buffers allocated: embedsBuffer=[1,{},{}], causalMask=[1,1,{},{}], {} KV layers",
                maxTotalSeq, hiddenSize, maxTotalSeq, maxTotalSeq, numKvLayers);
        log.info("STEP 4 DONE: {}ms", System.currentTimeMillis() - step4Start);

        // ==================== STEP 5: Fixed-shape decode loop ====================
        log.info("STEP 5: Decoding with fixed shapes (max {} tokens)...", maxNewTokens);
        long step5Start = System.currentTimeMillis();

        // Only request logits (skip KV cache outputs since we don't use them)
        String logitsName = logitsOutputName != null ? logitsOutputName : "logits";
        String[] decoderOutputNames = new String[]{logitsName};

        List<Integer> generatedTokens = new ArrayList<>();
        long validLen = prefillLen;
        long firstTokenNanos = 0;
        long planCompilations = 0;

        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.nanoTime();

            // Build fixed-shape decoder inputs — shapes never change!
            Map<String, INDArray> decoderInputMap = new HashMap<>();
            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, embedsBuffer);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, attentionMask);
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName, causalMask);
                } else if (inputName.equals("position_ids")) {
                    decoderInputMap.put(inputName, positionIds);
                } else if (inputName.startsWith("past_key_values.")) {
                    decoderInputMap.put(inputName, emptyKvCache.get(inputName));
                }
            }

            // Run decoder — all shapes identical every step
            Map<String, INDArray> outputs = decoder.output(decoderInputMap, decoderOutputNames);
            INDArray logits = outputs.get(logitsName);

            // Extract logits at the last valid position: logits[0, validLen-1, :]
            INDArray lastLogits = logits.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point((int)(validLen - 1)),
                    NDArrayIndex.all());
            int nextTokenId = SamplerUtils.argmax(lastLogits);

            long stepNanos = System.nanoTime() - stepStart;
            if (step == 0) {
                firstTokenNanos = stepNanos;
                log.info("  Step 0 (prefill + first token): {}ms, token={} '{}'",
                        stepNanos / 1_000_000, nextTokenId, tokenizer.decode(new int[]{nextTokenId}, true));
            }

            generatedTokens.add(nextTokenId);

            // Check EOS
            if (nextTokenId == tokenizer.getEosTokenId()) {
                log.info("  EOS at step {}", step);
                break;
            }

            // Embed the new token and write into the buffer at position validLen
            INDArray newTokenIds = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> newEmbedOutputs = embedTokens.output(Map.of(embedInputName, newTokenIds), embedOutputNames);
            INDArray newEmbed = newEmbedOutputs.get(embedOutputNames[0]);
            if (newEmbed.rank() == 2) {
                newEmbed = newEmbed.reshape(1, 1, hiddenSize);
            }

            // Write new embedding into pre-allocated buffer via host copy
            // (avoids cross-device .dup() issues on non-peer GPUs)
            INDArray newEmbedView = newEmbed.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all());
            float[] embData = newEmbedView.toFloatVector();
            INDArray newEmbedFlat = Nd4j.createFromArray(embData);
            embedsBuffer.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point((int)validLen), NDArrayIndex.all()}, newEmbedFlat);
            SameDiffMemoryUtils.safeClose(newEmbedFlat);

            // Unmask the new position in attention mask
            attentionMask.putScalar(new long[]{0, validLen}, 1);

            validLen++;

            // Log progress
            if (step > 0 && step % 5 == 0) {
                String decoded = tokenizer.decode(generatedTokens.stream().mapToInt(Integer::intValue).toArray(), true);
                log.info("  Step {}: {}ms/token, generated so far: '{}'",
                        step, stepNanos / 1_000_000, decoded.length() > 60 ? decoded.substring(0, 60) + "..." : decoded);
            }
        }

        long step5Elapsed = System.currentTimeMillis() - step5Start;

        // ==================== STEP 6: Results ====================
        int[] tokenIdArray = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIdArray, true);
        int tokenCount = generatedTokens.size();
        double tokensPerSec = tokenCount > 0 ? (tokenCount * 1000.0) / step5Elapsed : 0;
        double msPerToken = tokenCount > 0 ? (double) step5Elapsed / tokenCount : 0;

        log.info("=== FIXED-SHAPE DECODE RESULTS ===");
        log.info("  Generated: {} tokens in {}ms ({} tokens/sec, {} ms/token)",
                tokenCount, step5Elapsed, String.format("%.1f", tokensPerSec), String.format("%.1f", msPerToken));
        log.info("  First token latency: {}ms", firstTokenNanos / 1_000_000);
        log.info("  Max total sequence: {} (prefill={}, generated={})", maxTotalSeq, prefillLen, tokenCount);
        log.info("  Text: '{}'", generatedText.length() > 200 ? generatedText.substring(0, 200) + "..." : generatedText);
        log.info("  Key: ALL tensor shapes fixed at maxTotalSeq={} — plan compiles once", maxTotalSeq);
        log.info("=================================");

        // Verify we generated something
        assertTrue(tokenCount > 0, "Should generate at least one token");
        assertNotNull(generatedText, "Generated text should not be null");

        // Cleanup
        SameDiffMemoryUtils.safeClose(embedsBuffer);
        SameDiffMemoryUtils.safeClose(attentionMask);
        SameDiffMemoryUtils.safeClose(positionIds);
        SameDiffMemoryUtils.safeClose(causalMask);
        SameDiffMemoryUtils.safeClose(imageEmbeds);
        SameDiffMemoryUtils.safeClose(textEmbeds);
        SameDiffMemoryUtils.safeClose(pixelValues);
        for (INDArray kv : emptyKvCache.values()) {
            SameDiffMemoryUtils.safeClose(kv);
        }
        preprocessor.shutdown();
        tokenizer.close();

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
        log.info("Fixed-shape decode test complete.");
    }

    // ==================== Full Page OCR Test ====================

    /**
     * Full page OCR: decode until &lt;/doctag&gt; stop token, not a fixed token count.
     *
     * <p>This test matches the Docling Python library's approach to page OCR:</p>
     * <ul>
     *   <li>Uses max_new_tokens=8192 (Docling's ceiling), NOT a small fixed count</li>
     *   <li>Stops on &lt;/doctag&gt; (49230), &lt;end_of_utterance&gt; (49279), or EOS</li>
     *   <li>Validates the output is a structurally complete DocTags document</li>
     *   <li>Parses the output into structured elements with bounding boxes</li>
     *   <li>Converts to markdown for human-readable validation</li>
     * </ul>
     *
     * <p>Run with:</p>
     * <pre>
     *   -Dtest=TestVLMModelImportPipeline#testFullPageOcr
     *   -Dvlm.test.pdf.path=/path/to/document.pdf
     *   -Dvlm.test.pdf.page=0
     * </pre>
     */
    @Test
    @DisplayName("Full page OCR: EOS-driven decode matching Docling pipeline")
    public void testFullPageOcr() throws Exception {
        if (pdfPath == null || !new File(pdfPath).exists()) {
            log.info("Skipping test - no PDF provided. Use -Dvlm.test.pdf.path=/path/to/book.pdf");
            return;
        }

        Nd4j.getEnvironment().setLogNativeNDArrayCreation(false);

        // Docling uses max_new_tokens=8192; override with vlm.test.maxTokens if set
        int maxNewTokens = maxTokensConfig > 50 ? maxTokensConfig : 8192;

        log.info("=== FULL PAGE OCR TEST (Docling-style, EOS-driven) ===");
        log.info("  Max new tokens: {} (Docling default: 8192)", maxNewTokens);

        // ==================== STEP 1: Download Models ====================
        log.info("STEP 1: Downloading models...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);
        log.info("STEP 1 DONE.");

        // ==================== STEP 2: Load Tokenizer + Resolve Stop Tokens ====================
        log.info("STEP 2: Loading tokenizer and resolving stop tokens...");
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());

        int eosTokenId = tokenizer.getEosTokenId();
        Integer endOfUtteranceId = tokenizer.getTokenId("<end_of_utterance>");
        Integer doctagCloseId = tokenizer.getTokenId("</doctag>");
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);

        // Build complete stop token set (matching Docling's stop_strings)
        java.util.Set<Integer> stopTokenIds = new java.util.HashSet<>();
        stopTokenIds.add(eosTokenId);
        if (endOfUtteranceId != null) stopTokenIds.add(endOfUtteranceId);
        if (doctagCloseId != null) stopTokenIds.add(doctagCloseId);

        log.info("STEP 2 DONE: vocab={}, eos={}, endOfUtterance={}, doctagClose={}, imageToken={}",
                tokenizer.getVocabSize(), eosTokenId,
                endOfUtteranceId != null ? endOfUtteranceId : "N/A",
                doctagCloseId != null ? doctagCloseId : "N/A",
                imageTokenId);
        log.info("  Stop token IDs: {}", stopTokenIds);

        // ==================== STEP 3: Import ONNX Models ====================
        log.info("STEP 3: Importing ONNX models (with SDZ cache)...");
        long step3Start = System.currentTimeMillis();
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionResult.getModelFile().getAbsolutePath(),
                decoderResult.getModelFile().getAbsolutePath(),
                embedTokensResult.getModelFile().getAbsolutePath()
        );
        SameDiff visionEncoder = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokens = models[2];

        visionEncoder.enableWorkspaceMode(8 * 1024 * 1024);
        decoder.enableWorkspaceMode(8 * 1024 * 1024);
        embedTokens.enableWorkspaceMode(8 * 1024 * 1024);
        log.info("STEP 3 DONE: {}ms", System.currentTimeMillis() - step3Start);

        // ==================== STEP 4: Load & Tile Page ====================
        log.info("STEP 4: Loading and tiling page...");
        BufferedImage pageImage = loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        int targetSize = 512;
        int effectiveMaxTiles = maxTiles > 0 ? maxTiles : 9;
        int tilingThreads = Math.min(Runtime.getRuntime().availableProcessors(), 4);

        BufferedImage resized = ImageTiler.resizeLongestEdge(pageImage, 2048);
        ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLMParallel(
                resized, targetSize, effectiveMaxTiles, tilingThreads);
        log.info("  Page: {}x{} -> {} frames ({}x{} grid + global)",
                pageImage.getWidth(), pageImage.getHeight(),
                split.getTotalFrames(), split.numRows, split.numCols);

        // ==================== STEP 5: Pipelined Vision Encoding ====================
        int totalFrames = split.getTotalFrames();
        log.info("STEP 5: Vision encoding ({} frames)...", totalFrames);
        long step5Start = System.currentTimeMillis();

        final int fTargetSize = targetSize;
        int visionChunkSize = 2;
        PipelinedVisionEncoder pipelinedEncoder = new PipelinedVisionEncoder(
                visionEncoder, visionChunkSize, tilingThreads);
        List<INDArray> pageVisionEmbeddings = pipelinedEncoder.encodePipelined(
                List.of(split),
                () -> createSmolDoclingPreprocessor(fTargetSize, true),
                targetSize);
        pipelinedEncoder.shutdown();

        INDArray visionEmbedsFlat = pageVisionEmbeddings.get(0); // [1, totalSeq, hidden]
        long hiddenSize = visionEmbedsFlat.size(2);
        long visionSeqLen = visionEmbedsFlat.size(1);
        int imageSeqLenPerFrame = (int) (visionSeqLen / totalFrames);

        long step5Time = System.currentTimeMillis() - step5Start;
        log.info("STEP 5 DONE: {}ms, vision shape={}", step5Time, java.util.Arrays.toString(visionEmbedsFlat.shape()));

        // Free vision encoder to reclaim GPU memory
        int freedArrays = SameDiffMemoryUtils.freeModelArrays(visionEncoder);
        Nd4j.getExecutioner().commit();
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPoolOnStream(
                Nd4j.getAffinityManager().getDeviceForCurrentThread(), null);
        log.info("  Freed {} vision encoder arrays.", freedArrays);
        visionEncoder = null;

        // ==================== STEP 6: Build Prompt & Merge Embeddings ====================
        log.info("STEP 6: Building prompt and merging embeddings...");
        long step6Start = System.currentTimeMillis();

        String imagePrompt = ImagePromptBuilder.buildImagePromptString(split.numRows, split.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("  Prompt: {} tokens, {} <image> tokens", promptTokenIds.length,
                ImagePromptBuilder.countOccurrences(promptTokenIds, imageTokenId));

        // Embed text tokens
        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        String[] embedOutputNames = embedTokens.outputs().toArray(new String[0]);
        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length).castTo(DataType.LONG);
        Map<String, INDArray> embedOutputs = embedTokens.output(
                Map.of(embedInputName, promptTokenIdsTensor), embedOutputNames);
        INDArray textEmbeddings = embedOutputs.values().iterator().next().dup();
        for (INDArray orig : embedOutputs.values()) SameDiffMemoryUtils.safeClose(orig);
        embedTokens.clearPlaceholders(true);

        // Merge vision + text embeddings
        INDArray mergedEmbeddings = EmbeddingMerger.mergeEmbeddings(
                textEmbeddings, visionEmbedsFlat, promptTokenIds, imageTokenId);
        log.info("  Merged embedding shape: {}", java.util.Arrays.toString(mergedEmbeddings.shape()));
        log.info("STEP 6 DONE: {}ms", System.currentTimeMillis() - step6Start);

        // ==================== STEP 7: Decode until </doctag> ====================
        log.info("STEP 7: Decoding until </doctag> (max {} tokens, greedy, temperature=0)...", maxNewTokens);
        long step7Start = System.currentTimeMillis();

        // Decoder metadata
        String logitsOutputName = DecoderUtils.findLogitsOutputName(decoder);
        DecoderUtils.KVCacheNames kvNames = DecoderUtils.findKVCacheOutputNames(decoder);
        List<String> presentKeyNames = kvNames.keyNames;
        List<String> presentValueNames = kvNames.valueNames;
        List<String> decoderInputNames = decoder.inputs();

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(logitsOutputName);
        allOutputNames.addAll(presentKeyNames);
        allOutputNames.addAll(presentValueNames);

        Sampler sampler = Sampler.fromConfig(SamplingConfig.builder()
                .temperature(0.0).topK(1).topP(1.0).maxNewTokens(maxNewTokens).doSample(false).build());

        List<Integer> generatedTokens = new ArrayList<>();
        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        INDArray currentEmbeddings = mergedEmbeddings;
        long pastSeqLen = 0;
        boolean reachedStopToken = false;
        String stopReason = "MAX_TOKENS";
        long firstTokenLatencyMs = 0;

        for (int step = 0; step < maxNewTokens; step++) {
            long stepStart = System.currentTimeMillis();
            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;

            // Build decoder inputs
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();
            for (String inputName : decoderInputNames) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    decoderInputMap.put(inputName, Nd4j.ones(DataType.LONG, 1, totalSeqLen));
                } else if (inputName.equals("_causal_mask")) {
                    decoderInputMap.put(inputName,
                            DecoderUtils.buildCausalMask(1, currentSeqLen, totalSeqLen));
                } else if (inputName.equals("position_ids")) {
                    decoderInputMap.put(inputName,
                            Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                                    .reshape(1, currentSeqLen).castTo(DataType.LONG));
                } else if (inputName.startsWith("past_key_values.")) {
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        decoderInputMap.put(inputName,
                                DecoderUtils.createEmptyKvCache(decoder, inputName, 1, hiddenSize));
                    }
                }
            }
            if (!decoderInputMap.containsKey("inputs_embeds")) {
                decoderInputMap.put("inputs_embeds", currentEmbeddings);
            }

            // Run decoder
            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap,
                    allOutputNames.toArray(new String[0]));

            // Extract logits
            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            INDArray logits = logitsRaw.dup();
            SameDiffMemoryUtils.safeClose(logitsRaw);

            // Update KV cache
            for (INDArray old : kvCache.values()) SameDiffMemoryUtils.safeClose(old);
            kvCache.clear();
            for (String name : presentKeyNames) {
                INDArray pv = decoderOutputs.get(name);
                if (pv != null) kvCache.put(name, pv);
            }
            for (String name : presentValueNames) {
                INDArray pv = decoderOutputs.get(name);
                if (pv != null) kvCache.put(name, pv);
            }

            // Sample: greedy argmax from last position
            INDArray lastLogits;
            if (logits.rank() == 3) {
                lastLogits = logits.get(NDArrayIndex.all(),
                        NDArrayIndex.point(logits.size(1) - 1), NDArrayIndex.all()).dup();
            } else {
                lastLogits = logits;
            }
            int[] nextTokenIds = sampler.sampleBatch(lastLogits);
            int nextTokenId = nextTokenIds[0];
            if (lastLogits != logits) SameDiffMemoryUtils.safeClose(lastLogits);
            SameDiffMemoryUtils.safeClose(logits);

            generatedTokens.add(nextTokenId);

            if (step == 0) {
                firstTokenLatencyMs = System.currentTimeMillis() - step7Start;
            }

            // Check stop conditions (matching Docling's stop_strings)
            if (stopTokenIds.contains(nextTokenId)) {
                String stopTokenName = "EOS";
                if (doctagCloseId != null && nextTokenId == doctagCloseId) {
                    stopTokenName = "</doctag>";
                } else if (endOfUtteranceId != null && nextTokenId == endOfUtteranceId) {
                    stopTokenName = "<end_of_utterance>";
                }
                stopReason = stopTokenName;
                log.info("  Stop token '{}' (id={}) at step {} ({} tokens generated)",
                        stopTokenName, nextTokenId, step, generatedTokens.size());
                reachedStopToken = true;
                break;
            }

            // Embed next token for next step
            pastSeqLen += currentSeqLen;
            INDArray stepTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId})
                    .reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> stepEmbedOut = embedTokens.output(
                    Map.of(embedInputName, stepTokenTensor), embedOutputNames);
            INDArray prevEmbed = currentEmbeddings;
            currentEmbeddings = stepEmbedOut.values().iterator().next().dup();
            for (INDArray orig : stepEmbedOut.values()) SameDiffMemoryUtils.safeClose(orig);
            SameDiffMemoryUtils.safeClose(stepTokenTensor);
            embedTokens.clearPlaceholders(false);
            if (prevEmbed != mergedEmbeddings) SameDiffMemoryUtils.safeClose(prevEmbed);

            // Cleanup step inputs (not KV cache or embeddings)
            for (var entry : decoderInputMap.entrySet()) {
                String name = entry.getKey();
                if (name.equals("inputs_embeds") || name.startsWith("past_key_values.")) continue;
                SameDiffMemoryUtils.safeClose(entry.getValue());
            }
            decoder.clearPlaceholders(false);

            // Log progress periodically
            if (step < 5 || step % 50 == 0 || (step % 10 == 0 && step < 100)) {
                String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);
                log.info("  Step {} [{}ms]: '{}' (id={})",
                        step, System.currentTimeMillis() - stepStart, tokenText, nextTokenId);
            }
        }

        long step7Time = System.currentTimeMillis() - step7Start;

        // ==================== STEP 8: Parse & Validate Output ====================
        log.info("========================================");
        log.info("FULL PAGE OCR RESULTS:");
        log.info("========================================");

        int[] tokenIdArray = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String rawDocTags = tokenizer.decode(tokenIdArray, false);
        int tokenCount = generatedTokens.size();
        double tokensPerSec = tokenCount > 0 ? (tokenCount * 1000.0) / step7Time : 0;
        double msPerToken = tokenCount > 0 ? (double) step7Time / tokenCount : 0;

        log.info("GENERATION STATS:");
        log.info("  Tokens generated: {}", tokenCount);
        log.info("  Stop reason: {}", stopReason);
        log.info("  First token latency: {}ms", firstTokenLatencyMs);
        log.info("  Total decode time: {}ms", step7Time);
        log.info("  Throughput: {} tokens/sec ({} ms/token)",
                String.format("%.1f", tokensPerSec), String.format("%.1f", msPerToken));

        log.info("RAW OUTPUT ({} chars):", rawDocTags.length());
        log.info("{}", rawDocTags);

        // Parse DocTags into structured document
        DocTagsParser docTagsParser = new DocTagsParser();
        boolean isComplete = docTagsParser.isComplete(rawDocTags);
        DocumentStructure doc = docTagsParser.parse(rawDocTags);
        String markdown = docTagsParser.toMarkdown(doc);
        String plainText = docTagsParser.extractPlainText(rawDocTags);

        log.info("STRUCTURAL ANALYSIS:");
        log.info("  DocTags complete (has <doctag>...</doctag>): {}", isComplete);
        log.info("  Elements parsed: {}", doc.getElementCount());
        for (DocTagsParser.DocumentElement elem : doc.getElements()) {
            String bboxStr = elem.getBoundingBox() != null ?
                    String.format(" [%d,%d,%d,%d]", elem.getBoundingBox().getX1(),
                            elem.getBoundingBox().getY1(), elem.getBoundingBox().getX2(),
                            elem.getBoundingBox().getY2()) : "";
            String contentPreview = elem.getContent().length() > 80 ?
                    elem.getContent().substring(0, 80) + "..." : elem.getContent();
            log.info("    <{}>{}: '{}'", elem.getTagType(), bboxStr, contentPreview);
        }

        log.info("MARKDOWN OUTPUT:");
        log.info("{}", markdown);

        log.info("PLAIN TEXT ({} chars):", plainText.length());
        log.info("{}", plainText.length() > 500 ? plainText.substring(0, 500) + "..." : plainText);

        // ==================== Assertions ====================
        // Must generate at least some tokens
        assertTrue(tokenCount > 0, "Should generate at least one token");

        // If generation hit a stop token, the document should be structurally complete
        if (reachedStopToken) {
            assertTrue(isComplete,
                    "When stop token is reached, output should be a complete <doctag>...</doctag> document");
        }

        // Should have parsed at least one document element
        assertTrue(doc.getElementCount() > 0,
                "Should parse at least one document element from the DocTags output. Raw: " +
                        (rawDocTags.length() > 200 ? rawDocTags.substring(0, 200) : rawDocTags));

        // Plain text content should be non-trivial
        assertTrue(plainText.length() > 10,
                "OCR'd text should be non-trivial (>10 chars). Got: '" + plainText + "'");

        log.info("========================================");
        log.info("VALIDATION: PASSED");
        log.info("  - Generated {} tokens (stop: {})", tokenCount, stopReason);
        log.info("  - {} document elements parsed", doc.getElementCount());
        log.info("  - {} chars of plain text extracted", plainText.length());
        log.info("========================================");

        // ==================== Cleanup ====================
        for (INDArray embed : pageVisionEmbeddings) SameDiffMemoryUtils.safeClose(embed);
        SameDiffMemoryUtils.safeClose(textEmbeddings);
        SameDiffMemoryUtils.safeClose(mergedEmbeddings);
        if (currentEmbeddings != mergedEmbeddings) SameDiffMemoryUtils.safeClose(currentEmbeddings);
        for (INDArray kv : kvCache.values()) SameDiffMemoryUtils.safeClose(kv);
        SameDiffMemoryUtils.safeClose(promptTokenIdsTensor);
        tokenizer.close();

        org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getShutdownInProgress().set(true);
        log.info("Full page OCR test complete.");
    }
}
