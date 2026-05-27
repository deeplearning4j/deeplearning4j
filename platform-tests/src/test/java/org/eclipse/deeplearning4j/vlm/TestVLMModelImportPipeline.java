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
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.eclipse.deeplearning4j.llm.generation.*;

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
 *
 * Adam Gibson
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestVLMModelImportPipeline {

    private static final String PDF_PATH_PROPERTY = "vlm.test.pdf.path";
    private static final String PDF_PAGE_PROPERTY = "vlm.test.pdf.page";       // Single page (0-based)
    private static final String PDF_MAX_PAGES_PROPERTY = "vlm.test.pdf.maxPages"; // Max pages to process
    private static final String PDF_DPI_PROPERTY = "vlm.test.pdf.dpi";         // Render DPI (default 150)
    private static final String PYTHON_REF_DIR = System.getProperty("vlm.python.ref.dir", "/tmp");

    private static String pdfPath;
    private static int specificPage = -1;   // -1 means process all/range
    private static int maxPages = -1;       // -1 means no limit
    private static int renderDpi = 150;

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

        String dpiStr = System.getProperty(PDF_DPI_PROPERTY);
        if (dpiStr != null && !dpiStr.isEmpty()) {
            renderDpi = Integer.parseInt(dpiStr);
        }

        log.info("VLM Model Import Pipeline Test Configuration:");
        log.info("  PDF Path: {}", pdfPath != null ? pdfPath : "(not set)");
        log.info("  Specific Page: {}", specificPage >= 0 ? specificPage : "(all pages)");
        log.info("  Max Pages: {}", maxPages > 0 ? maxPages : "(no limit)");
        log.info("  Render DPI: {}", renderDpi);
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
        // The Python reference uses page10-010.png which was rendered at 150 DPI
        // If that file exists, load it in Java and compare
        File pythonImageFile = new File(PYTHON_REF_DIR, "page10-010.png");
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
        File refBin = new File(PYTHON_REF_DIR, "python_vision_input_3x512x512.bin");
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

}
