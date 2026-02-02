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
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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
@NativeTag
@Tag(TagNames.FILE_IO)
@Tag("vlm")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestVLMModelImportPipeline {

    private static final String PDF_PATH_PROPERTY = "vlm.test.pdf.path";
    private static final String PDF_PAGE_PROPERTY = "vlm.test.pdf.page";       // Single page (0-based)
    private static final String PDF_MAX_PAGES_PROPERTY = "vlm.test.pdf.maxPages"; // Max pages to process
    private static final String PDF_DPI_PROPERTY = "vlm.test.pdf.dpi";         // Render DPI (default 150)
    private static final String MAX_TILES_PROPERTY = "vlm.test.maxTiles";      // Max tiles per image (default -1 = no limit)

    private static String pdfPath;
    private static int specificPage = -1;   // -1 means process all/range
    private static int maxPages = -1;       // -1 means no limit
    private static int renderDpi = 150;
    private static int maxTiles = -1;       // -1 means no limit

    @BeforeAll
    public static void setup() {
        pdfPath = System.getProperty(PDF_PATH_PROPERTY);

        // Parse page selection properties
        String pageStr = System.getProperty(PDF_PAGE_PROPERTY);
        if (pageStr != null) {
            specificPage = Integer.parseInt(pageStr);
        }

        String maxPagesStr = System.getProperty(PDF_MAX_PAGES_PROPERTY);
        if (maxPagesStr != null) {
            maxPages = Integer.parseInt(maxPagesStr);
        }

        String dpiStr = System.getProperty(PDF_DPI_PROPERTY);
        if (dpiStr != null) {
            renderDpi = Integer.parseInt(dpiStr);
        }

        String maxTilesStr = System.getProperty(MAX_TILES_PROPERTY);
        if (maxTilesStr != null) {
            maxTiles = Integer.parseInt(maxTilesStr);
        }

        log.info("VLM Model Import Pipeline Test Configuration:");
        log.info("  PDF Path: {}", pdfPath != null ? pdfPath : "(not set)");
        log.info("  Specific Page: {}", specificPage >= 0 ? specificPage : "(all pages)");
        log.info("  Max Pages: {}", maxPages > 0 ? maxPages : "(no limit)");
        log.info("  Render DPI: {}", renderDpi);
        log.info("  Max Tiles: {}", maxTiles > 0 ? maxTiles : "(no limit)");
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
        log.info("=== SmolDocling Full Pipeline ===");


        // Download vision encoder, decoder, embed_tokens, and tokenizer
        log.info("Downloading SmolDocling vision encoder...");
        var visionResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);

        log.info("Downloading SmolDocling decoder...");
        var decoderResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);

        log.info("Downloading SmolDocling embed_tokens...");
        var embedTokensResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);

        log.info("Downloading SmolDocling tokenizer...");
        var tokenizerResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        var tokenizerConfigResult = VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        // Load tokenizer for text decoding
        log.info("Loading tokenizer from: {}", tokenizerResult.getModelFile().getAbsolutePath());
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerResult.getModelFile());
        log.info("Tokenizer loaded: vocab_size={}, eos_token_id={}, bos_token_id={}",
                tokenizer.getVocabSize(), tokenizer.getEosTokenId(), tokenizer.getBosTokenId());

        // Import models - IMPORTANT: Use separate importer instances to avoid shared state
        // that can cause double free when native resources are shared across models
        log.info("Importing vision encoder...");
        OnnxFrameworkImporter visionImporter = new OnnxFrameworkImporter();
        SameDiff visionEncoder = visionImporter.runImport(visionResult.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Vision encoder: {} variables, inputs={}, outputs={}",
                visionEncoder.variables().size(), visionEncoder.inputs(), visionEncoder.outputs());

        // ===== COMPREHENSIVE VISION ENCODER DEBUGGING =====
        log.info("===== VISION ENCODER MODEL INSPECTION =====");

        // Check key weight tensors to ensure they were loaded correctly
        String[] keyWeights = {
                "vision_model.embeddings.patch_embedding.weight",
                "vision_model.embeddings.patch_embedding.bias",
                "vision_model.embeddings.position_embedding.weight",
                "vision_model.encoder.layers.0.layer_norm1.weight",
                "vision_model.encoder.layers.0.self_attn.q_proj.bias",
                "vision_model.post_layernorm.weight",
                "onnx::MatMul_3025"  // Connector linear weight - critical for image_features output
        };
        for (String weightName : keyWeights) {
            SDVariable var = visionEncoder.getVariable(weightName);
            if (var != null) {
                INDArray arr = var.getArr();
                if (arr != null) {
                    double min = arr.minNumber().doubleValue();
                    double max = arr.maxNumber().doubleValue();
                    boolean allZero = (min == 0.0 && max == 0.0);
                    log.info("Weight '{}': shape={}, dtype={}, min={}, max={}, mean={}, allZero={}",
                            weightName, java.util.Arrays.toString(arr.shape()), arr.dataType(),
                            min, max, arr.meanNumber(), allZero);
                    if (allZero && arr.length() > 1) {
                        log.error("CRITICAL: Weight '{}' is ALL ZEROS! This will cause broken output.", weightName);
                    }
                } else {
                    log.warn("Weight '{}': array is NULL", weightName);
                }
            } else {
                log.warn("Weight '{}': variable NOT FOUND", weightName);
            }
        }

        // Also check all onnx::MatMul weights for zeros
        log.info("Checking all MatMul weights for zero values...");
        int zeroWeightCount = 0;
        for (String varName : visionEncoder.getVariables().keySet()) {
            if (varName.startsWith("onnx::MatMul_")) {
                SDVariable var = visionEncoder.getVariable(varName);
                if (var != null && var.getArr() != null) {
                    INDArray arr = var.getArr();
                    double min = arr.minNumber().doubleValue();
                    double max = arr.maxNumber().doubleValue();
                    if (min == 0.0 && max == 0.0) {
                        log.error("MatMul weight '{}' is ALL ZEROS! shape={}", varName, java.util.Arrays.toString(arr.shape()));
                        zeroWeightCount++;
                    }
                }
            }
        }
        if (zeroWeightCount > 0) {
            log.error("Found {} MatMul weights that are ALL ZEROS - model weights may not have loaded correctly!", zeroWeightCount);
        } else {
            log.info("All MatMul weights have non-zero values.");
        }

        // List all operations in the vision encoder
        log.info("Vision encoder has {} operations", visionEncoder.getOps().size());
        int opCount = 0;
        for (String opName : visionEncoder.getOps().keySet()) {
            if (opCount < 20 || opName.contains("patch_embedding") || opName.contains("layer_norm") || opName.contains("connector")) {
                var op = visionEncoder.getOps().get(opName);
                log.info("  Op[{}]: {} -> inputs={}, outputs={}",
                        opCount, opName, op.getInputsToOp(), op.getOutputsOfOp());
            }
            opCount++;
        }
        if (opCount > 20) {
            log.info("  ... and {} more operations", opCount - 20);
        }

        // Check the output variable
        for (String outputName : visionEncoder.outputs()) {
            SDVariable outVar = visionEncoder.getVariable(outputName);
            if (outVar != null) {
                log.info("Output '{}': type={}, shape={}", outputName, outVar.getVariableType(),
                        java.util.Arrays.toString(outVar.getShape()));
            }
        }
        log.info("===== END VISION ENCODER INSPECTION =====");

        log.info("Importing decoder...");
        OnnxFrameworkImporter decoderImporter = new OnnxFrameworkImporter();
        SameDiff decoder = decoderImporter.runImport(decoderResult.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Decoder: {} variables, inputs={}, outputs={}",
                decoder.variables().size(), decoder.inputs(), decoder.outputs());

        log.info("Importing embed_tokens...");
        OnnxFrameworkImporter embedImporter = new OnnxFrameworkImporter();
        SameDiff embedTokens = embedImporter.runImport(embedTokensResult.getModelFile().getAbsolutePath(), Map.of(), false, false);
        log.info("Embed tokens: {} variables, inputs={}, outputs={}",
                embedTokens.variables().size(), embedTokens.inputs(), embedTokens.outputs());

        // Fix the decoder model: the ONNX model has input_ids baked in as a constant
        // We need to add input_ids as a dynamic placeholder and fix the attention mask computation
        fixDecoderInputIds(decoder, tokenizer);
        log.info("Fixed decoder input_ids: now has inputs={}", decoder.inputs());

        // Diagnose and potentially fix the repeat_kv Reshape operations
        fixRepeatKVReshape(decoder);
        // Ensure inputs_embeds is wired into the decoder graph
        fixDecoderInputsEmbeds(decoder);
        boolean debugGraph = Boolean.parseBoolean(System.getProperty("vlm.test.debugGraph", "true"));
        if (debugGraph) {
            logDecoderInputUsage(decoder, "inputs_embeds");
            logDecoderInputUsage(decoder, "attention_mask");
            logDecoderInputUsage(decoder, "position_ids");
            logVariablesContaining(decoder, "inputs_embeds");
        }

        // ==================== SMOLDOCLING PROMPT TEMPLATE ====================
        // SmolDocling expects a chat-formatted prompt with <image> token that gets replaced
        // with vision embeddings. Format: <|im_start|>User:<image>PROMPT<end_of_utterance>\nAssistant:
        int imageTokenId = resolveImageTokenId(tokenizer);
        log.info("Resolved <image> token id: {}", imageTokenId);
        String promptText = "Convert this page to docling.";

        // Load real image from PDF or generate test pattern
        BufferedImage pdfImage = loadImageFromPdfOrGenerate(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        log.info("Loaded image: {}x{}", pdfImage.getWidth(), pdfImage.getHeight());

        // ==================== IMAGE SPLITTING FOR SMOLDOCLING ====================
        // SmolDocling/Idefics3 resizes longest edge to 2048, then splits into 512x512 tiles + a global thumbnail.
        // This is critical for document understanding - shrinking to 512x512 loses all text detail.
        int longestEdge = 2048;
        int targetSize = 512;

        BufferedImage resizedForTiling = resizeLongestEdge(pdfImage, longestEdge);
        log.info("Resized for tiling: {}x{}", resizedForTiling.getWidth(), resizedForTiling.getHeight());

        // Split image into tiles
        SplitImageResult splitResult = splitImageForVLM(resizedForTiling, targetSize);
        int numFrames = splitResult.getTotalFrames();
        log.info("Split image into {} frames ({} tiles + 1 global)", numFrames, splitResult.getTileCount());

        // Save each tile for inspection
        String outputDir = System.getProperty("user.dir") + File.separator + "target" + File.separator + "vlm-test-output";
        new File(outputDir).mkdirs();
        for (int i = 0; i < splitResult.frames.size(); i++) {
            BufferedImage frame = splitResult.frames.get(i);
            String tileName = i < splitResult.getTileCount()
                    ? String.format("tile_%02d_r%d_c%d.png", i, i / Math.max(1, splitResult.numCols), i % Math.max(1, splitResult.numCols))
                    : "tile_global.png";
            saveImage(frame, outputDir + File.separator + tileName);
        }
        log.info("Saved {} tile images to {}", splitResult.frames.size(), outputDir);

        boolean disableNormalize = Boolean.parseBoolean(System.getProperty("vlm.test.disableNormalize", "false"));
        VLMImagePreprocessor preprocessor = createSmolDoclingPreprocessor(targetSize, !disableNormalize);

        // Preprocess all frames into a 5D tensor [batch, numFrames, channels, H, W]
        INDArray imageInput = preprocessFramesForSmolDocling(splitResult.frames, preprocessor, targetSize);
        log.info("Preprocessed {} frames into tensor: {}", numFrames, java.util.Arrays.toString(imageInput.shape()));

        // Save first preprocessed tile for visual inspection
        savePreprocessedTensor(imageInput,
                disableNormalize ? new double[]{0.0, 0.0, 0.0} : new double[]{0.5, 0.5, 0.5},
                disableNormalize ? new double[]{1.0, 1.0, 1.0} : new double[]{0.5, 0.5, 0.5},
                "preprocessed_tile_0.png");

        preprocessor.shutdown();

        // ==================== PROCESS EACH FRAME THROUGH VISION ENCODER ====================
        // The vision encoder processes one frame at a time, then we concatenate embeddings
        List<String> visionInputNames = visionEncoder.inputs();
        log.info("Vision encoder inputs: {}", visionInputNames);
        if (debugGraph) {
            for (String inputName : visionInputNames) {
                SDVariable var = visionEncoder.getVariable(inputName);
                if (var != null) {
                    log.info("Vision input '{}' shape={}, dtype={}", inputName,
                            java.util.Arrays.toString(var.getShape()), var.dataType());
                }
            }
        }
        boolean disablePixelMask = Boolean.parseBoolean(System.getProperty("vlm.test.disablePixelMask", "false"));
        if (disablePixelMask) {
            log.warn("Pixel attention mask disabled via -Dvlm.test.disablePixelMask=true");
        }

        List<INDArray> frameEmbeddings = new java.util.ArrayList<>();

        // ===== DEBUG: Test vision encoder with a simple forward pass first =====
        log.info("===== VISION ENCODER DEBUG FORWARD PASS =====");
        // Enable ND4J debug and verbose mode to see all op executions
        Nd4j.getEnvironment().setDebug(true);
        Nd4j.getEnvironment().setVerbose(true);
        {
            // Create a simple test input
            INDArray testPixelValues = Nd4j.rand(DataType.FLOAT, 1, 1, 3, targetSize, targetSize).muli(2).subi(1); // [-1, 1]
            INDArray testMask = Nd4j.ones(DataType.BOOL, 1, 1, targetSize, targetSize);

            log.info("Test input pixel_values: shape={}, dtype={}, min={}, max={}, mean={}",
                    java.util.Arrays.toString(testPixelValues.shape()), testPixelValues.dataType(),
                    testPixelValues.minNumber(), testPixelValues.maxNumber(), testPixelValues.meanNumber());
            log.info("Test input mask: shape={}, dtype={}, all_true={}",
                    java.util.Arrays.toString(testMask.shape()), testMask.dataType(), testMask.all());

            // Find intermediate outputs to trace - capture ALL LayerNorm ops to find where NaN starts
            List<String> intermediateOutputs = new java.util.ArrayList<>();
            for (String varName : visionEncoder.getVariables().keySet()) {
                if (varName.contains("patch_embedding") && varName.contains("output")) {
                    intermediateOutputs.add(varName);
                } else if (varName.contains("layer_norm1") && varName.contains("output") && varName.contains("layers.0")) {
                    // Capture ALL layer_norm1 ops: Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul
                    intermediateOutputs.add(varName);
                } else if (varName.contains("connector") && varName.contains("output")) {
                    intermediateOutputs.add(varName);
                }
            }
            // Log all layer_norm1 variables found for debugging
            log.info("LayerNorm1 intermediate vars: {}", intermediateOutputs.stream()
                    .filter(v -> v.contains("layer_norm1")).collect(java.util.stream.Collectors.toList()));
            intermediateOutputs.addAll(visionEncoder.outputs());
            log.info("Will trace {} intermediate outputs: {}", intermediateOutputs.size(),
                    intermediateOutputs.size() <= 10 ? intermediateOutputs : intermediateOutputs.subList(0, 10) + "...");

            // Run forward pass with intermediate outputs
            Map<String, INDArray> testInputs = new java.util.HashMap<>();
            testInputs.put("pixel_values", testPixelValues);
            testInputs.put("pixel_attention_mask", testMask);

            try {
                Map<String, INDArray> testOutputs = visionEncoder.output(testInputs, intermediateOutputs.toArray(new String[0]));
                for (var entry : testOutputs.entrySet()) {
                    INDArray arr = entry.getValue();
                    boolean allZero = arr.minNumber().doubleValue() == 0.0 && arr.maxNumber().doubleValue() == 0.0;
                    log.info("  Intermediate '{}': shape={}, dtype={}, min={}, max={}, mean={}, allZero={}",
                            entry.getKey(), java.util.Arrays.toString(arr.shape()), arr.dataType(),
                            arr.minNumber(), arr.maxNumber(), arr.meanNumber(), allZero);
                }
            } catch (Exception e) {
                log.error("Error running intermediate outputs: {}", e.getMessage());
                // Fall back to just the final output
                Map<String, INDArray> testOutputs = visionEncoder.output(testInputs, visionEncoder.outputs().toArray(new String[0]));
                for (var entry : testOutputs.entrySet()) {
                    INDArray arr = entry.getValue();
                    log.info("  Final output '{}': shape={}, min={}, max={}, mean={}",
                            entry.getKey(), java.util.Arrays.toString(arr.shape()),
                            arr.minNumber(), arr.maxNumber(), arr.meanNumber());
                }
            }
        }
        log.info("===== END VISION ENCODER DEBUG =====");

        boolean retryNoNormalize = Boolean.parseBoolean(System.getProperty("vlm.test.retryNoNormalize", "true"));
        boolean retryDisablePixelMask = Boolean.parseBoolean(System.getProperty("vlm.test.retryDisablePixelMask", "true"));
        boolean needsRetry = false;

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            log.info("Processing frame {}/{}", frameIdx + 1, numFrames);

            // Extract single frame: [1, 1, 3, H, W]
            INDArray singleFrame = imageInput.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(frameIdx),
                    NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all()
            ).dup().reshape(1, 1, 3, targetSize, targetSize);

            // Diagnostic: Log input stats
            log.info("  Frame {} input pixel_values: shape={}, dtype={}, min={}, max={}, mean={}",
                    frameIdx, java.util.Arrays.toString(singleFrame.shape()), singleFrame.dataType(),
                    singleFrame.minNumber(), singleFrame.maxNumber(), singleFrame.meanNumber());

            // Create input map for this frame
            Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
            for (String inputName : visionInputNames) {
                if (inputName.equals("pixel_values")) {
                    visionInputMap.put(inputName, singleFrame);
                } else if (inputName.equals("pixel_attention_mask")) {
                    INDArray mask;
                    if (disablePixelMask) {
                        mask = Nd4j.ones(DataType.BOOL, 1, 1, targetSize, targetSize);
                    } else {
                        ContentRegion region = splitResult.contentRegions.get(frameIdx);
                        mask = createPixelAttentionMask(region.width, region.height, targetSize);
                    }
                    if (frameIdx == 0) {
                        log.info("  Frame {} mask: dtype={}, all_true={}",
                                frameIdx, mask.dataType(), mask.all());
                    }
                    visionInputMap.put(inputName, mask);
                }
            }

            // Run vision encoder for this frame
            Map<String, INDArray> visionOutputs = visionEncoder.output(
                    visionInputMap,
                    visionEncoder.outputs().toArray(new String[0]));

            // Log available outputs for debugging with value stats
            for (var entry : visionOutputs.entrySet()) {
                INDArray out = entry.getValue();
                log.info("  Frame {} output '{}': shape={}, min={}, max={}, mean={}",
                        frameIdx, entry.getKey(), java.util.Arrays.toString(out.shape()),
                        out.minNumber(), out.maxNumber(), out.meanNumber());
            }

            // Select the correct vision embeddings (usually last_hidden_state)
            VisionOutput selected = selectVisionOutput(visionOutputs);
            if (selected == null) {
                throw new RuntimeException("Vision encoder produced no usable outputs");
            }
            INDArray out = selected.tensor.dup();  // IMPORTANT: dup to avoid workspace reuse issues
            log.info("  Frame {} selected output '{}': shape={}, min={}, max={}, mean={}",
                    frameIdx, selected.name, java.util.Arrays.toString(out.shape()),
                    out.minNumber(), out.maxNumber(), out.meanNumber());
            if (frameIdx == 0 && isAllZeroOrNaN(out) && retryNoNormalize) {
                needsRetry = true;
            }
            frameEmbeddings.add(out);

            // CRITICAL: Release memory from forward pass to prevent OOM
            // Close all output arrays except the one we dup'd
            for (var entry : visionOutputs.entrySet()) {
                if (entry.getValue() != selected.tensor) {
                    entry.getValue().close();
                }
            }
            // Close the original selected tensor (we have a dup)
            selected.tensor.close();
            // Close input arrays
            singleFrame.close();
            for (var entry : visionInputMap.entrySet()) {
                if (!entry.getKey().equals("pixel_values")) {
                    entry.getValue().close();
                }
            }
            // Clear placeholder arrays and op inputs from the graph
            visionEncoder.clearPlaceholders(false);
            visionEncoder.clearOpInputs();
        }

        if (needsRetry) {
            log.warn("Vision output is all-zero/NaN. Retrying with no normalization and full mask.");
            if (retryDisablePixelMask) {
                disablePixelMask = true;
            }
            VLMImagePreprocessor retryPreprocessor = createSmolDoclingPreprocessor(targetSize, false);
            INDArray retryInput = preprocessFramesForSmolDocling(splitResult.frames, retryPreprocessor, targetSize);
            retryPreprocessor.shutdown();
            imageInput = retryInput;
            frameEmbeddings.clear();
            for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
                INDArray singleFrame = imageInput.get(
                        NDArrayIndex.point(0),
                        NDArrayIndex.point(frameIdx),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(),
                        NDArrayIndex.all()
                ).dup().reshape(1, 1, 3, targetSize, targetSize);

                Map<String, INDArray> visionInputMap = new java.util.HashMap<>();
                for (String inputName : visionInputNames) {
                    if (inputName.equals("pixel_values")) {
                        visionInputMap.put(inputName, singleFrame);
                    } else if (inputName.equals("pixel_attention_mask")) {
                        INDArray mask = Nd4j.ones(DataType.BOOL, 1, 1, targetSize, targetSize);
                        visionInputMap.put(inputName, mask);
                    }
                }

                Map<String, INDArray> visionOutputs = visionEncoder.output(
                        visionInputMap,
                        visionEncoder.outputs().toArray(new String[0]));
                VisionOutput selected = selectVisionOutput(visionOutputs);
                if (selected == null) {
                    throw new RuntimeException("Vision encoder produced no usable outputs on retry");
                }
                INDArray out = selected.tensor.dup();
                log.info("  Retry frame {} output '{}': shape={}, min={}, max={}, mean={}",
                        frameIdx, selected.name, java.util.Arrays.toString(out.shape()),
                        out.minNumber(), out.maxNumber(), out.meanNumber());
                frameEmbeddings.add(out);

                // CRITICAL: Release memory from forward pass to prevent OOM
                for (var entry : visionOutputs.entrySet()) {
                    if (entry.getValue() != selected.tensor) {
                        entry.getValue().close();
                    }
                }
                selected.tensor.close();
                singleFrame.close();
                for (var entry : visionInputMap.entrySet()) {
                    if (!entry.getKey().equals("pixel_values")) {
                        entry.getValue().close();
                    }
                }
                visionEncoder.clearPlaceholders(false);
                visionEncoder.clearOpInputs();
            }
        }

        // Concatenate all frame embeddings along sequence dimension
        // Each frame gives [1, 64, 576], concatenate to [1, numFrames*64, 576]
        INDArray visionEmbeddings;
        if (frameEmbeddings.size() == 1) {
            visionEmbeddings = frameEmbeddings.get(0);
        } else {
            // Stack along sequence dimension (dim 1)
            visionEmbeddings = Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
        }
        log.info("Combined vision embeddings: shape={}", java.util.Arrays.toString(visionEmbeddings.shape()));
        if (visionEmbeddings.rank() != 3) {
            throw new IllegalStateException("Expected vision embeddings rank 3, got " + visionEmbeddings.rank());
        }

        int imageSeqLenPerImage = (int) frameEmbeddings.get(0).size(1);
        for (int i = 1; i < frameEmbeddings.size(); i++) {
            if (frameEmbeddings.get(i).size(1) != imageSeqLenPerImage) {
                log.warn("Frame {} has seq_len {} (expected {})",
                        i, frameEmbeddings.get(i).size(1), imageSeqLenPerImage);
            }
        }

        String imagePrompt = buildImagePromptString(splitResult.numRows, splitResult.numCols, imageSeqLenPerImage);
        String chatPrompt = "<|im_start|>User:" + imagePrompt + promptText + "<end_of_utterance>\nAssistant:";
        log.info("Chat prompt length: {} chars", chatPrompt.length());

        // Tokenize the prompt
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
        log.info("Tokenized prompt: {} tokens", promptTokenIds.length);
        int promptImageTokens = countOccurrences(promptTokenIds, imageTokenId);
        log.info("Prompt <image> tokens: {} (vision tokens: {})", promptImageTokens, visionEmbeddings.size(1));
        if (promptImageTokens == 0) {
            log.warn("Prompt contains no <image> tokens after expansion");
        }

        // === Autoregressive Text Generation with Decoder ===
        log.info("=== Starting Text Generation ===");
        log.info("Decoder inputs required: {}", decoder.inputs());

        // Debug: Print decoder inputs and outputs
        log.info("Decoder has {} inputs, {} outputs", decoder.inputs().size(), decoder.outputs().size());
        log.info("Decoder inputs: {}", decoder.inputs());
        for (String inputName : decoder.inputs()) {
            if (!inputName.startsWith("past_key_values")) {
                log.info("  Input '{}': var exists={}", inputName, decoder.hasVariable(inputName));
            }
        }

        // Get vision output dimensions
        long batchSize = visionEmbeddings.shape()[0];
        long visionSeqLen = visionEmbeddings.shape()[1];
        long hiddenSize = visionEmbeddings.shape()[2];
        log.info("Vision embeddings: batch={}, seq_len={}, hidden={}", batchSize, visionSeqLen, hiddenSize);
        if (visionSeqLen <= 0) {
            throw new IllegalStateException("Vision embeddings sequence length is <= 0");
        }

        // Get text embeddings for the prompt tokens
        log.info("Getting text embeddings for {} prompt tokens", promptTokenIds.length);
        INDArray promptTokenIdsTensor = Nd4j.createFromArray(promptTokenIds)
                .reshape(1, promptTokenIds.length)
                .castTo(DataType.LONG);

        String embedInputName = embedTokens.inputs().isEmpty() ? "input_ids" : embedTokens.inputs().get(0);
        Map<String, INDArray> embedInputMap = Map.of(embedInputName, promptTokenIdsTensor);
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputMap, embedTokens.outputs().toArray(new String[0]));

        INDArray textEmbeddings = null;
        for (var entry : embedOutputs.entrySet()) {
            textEmbeddings = entry.getValue().dup();  // IMPORTANT: dup to avoid workspace reuse issues
            log.info("Text embeddings: shape={}", java.util.Arrays.toString(textEmbeddings.shape()));
        }

        if (textEmbeddings == null) {
            throw new RuntimeException("embed_tokens produced no output");
        }

        // Check dimensions - vision and text must have matching hidden size
        long visionHiddenSize = visionEmbeddings.shape()[2];
        long textHiddenSize = textEmbeddings.shape()[2];
        log.info("Vision hidden size: {}, Text hidden size: {}", visionHiddenSize, textHiddenSize);

        if (visionHiddenSize != textHiddenSize) {
            throw new IllegalStateException("Hidden size mismatch: vision=" + visionHiddenSize + " text=" + textHiddenSize);
        }

        // Replace <image> token embeddings with vision embeddings (ONNX reference behavior)
        INDArray inputsEmbeds = textEmbeddings.dup();
        int imageSlots = countOccurrences(promptTokenIds, imageTokenId);
        if (imageSlots != visionSeqLen) {
            log.warn("Image token slots ({}) != vision tokens ({}). Will fill {} slots.",
                    imageSlots, visionSeqLen, Math.min(imageSlots, (int) visionSeqLen));
        }

        // Diagnostic: Check vision embedding values
        log.info("Vision embeddings stats: min={}, max={}, mean={}",
                visionEmbeddings.minNumber(), visionEmbeddings.maxNumber(), visionEmbeddings.meanNumber());
        log.info("Text embeddings stats: min={}, max={}, mean={}",
                textEmbeddings.minNumber(), textEmbeddings.maxNumber(), textEmbeddings.meanNumber());

        INDArray visionFlat = visionEmbeddings.reshape((int) visionSeqLen, (int) visionHiddenSize);
        int fillCount = Math.min(imageSlots, (int) visionSeqLen);
        int fillIdx = 0;
        for (int pos = 0; pos < promptTokenIds.length && fillIdx < fillCount; pos++) {
            if (promptTokenIds[pos] == imageTokenId) {
                inputsEmbeds.put(
                        new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.point(pos), NDArrayIndex.all()},
                        visionFlat.getRow(fillIdx)
                );
                fillIdx++;
            }
        }
        log.info("Filled {} of {} image token positions", fillIdx, imageSlots);
        if (fillIdx < visionSeqLen) {
            log.warn("Only filled {} of {} vision tokens", fillIdx, visionSeqLen);
        }

        // Diagnostic: Check final inputsEmbeds
        log.info("Final inputsEmbeds stats: min={}, max={}, mean={}",
                inputsEmbeds.minNumber(), inputsEmbeds.maxNumber(), inputsEmbeds.meanNumber());

        // Configure sampling - use greedy decoding (temperature=0) for deterministic output
        SamplingConfig samplingConfig = SamplingConfig.builder()
                .temperature(0.0)  // Greedy decoding
                .topK(1)           // Only consider top token
                .topP(1.0)
                .maxNewTokens(50)
                .doSample(false)   // Disable sampling for greedy
                .build();
        Sampler sampler = Sampler.fromConfig(samplingConfig);
        log.info("Using sampler with temp={}, topK={}, topP={}",
                samplingConfig.getTemperature(), samplingConfig.getTopK(), samplingConfig.getTopP());
        boolean debugEmbeds = Boolean.parseBoolean(System.getProperty("vlm.test.debugEmbeds", "true"));

        // Track generated tokens
        List<Integer> generatedTokens = new java.util.ArrayList<>();
        int eosTokenId = tokenizer.getEosTokenId();
        log.info("EOS token ID: {}", eosTokenId);

        // Find the logits output name and KV cache output names
        String logitsOutputName = null;
        List<String> presentKeyNames = new java.util.ArrayList<>();
        List<String> presentValueNames = new java.util.ArrayList<>();

        for (String outputName : decoder.outputs()) {
            if (outputName.contains("logit") || outputName.equals("logits")) {
                logitsOutputName = outputName;
            } else if (outputName.contains("present") && outputName.contains("key")) {
                presentKeyNames.add(outputName);
            } else if (outputName.contains("present") && outputName.contains("value")) {
                presentValueNames.add(outputName);
            }
        }
        if (logitsOutputName == null && !decoder.outputs().isEmpty()) {
            logitsOutputName = decoder.outputs().get(0);
        }
        log.info("Using logits output: {}", logitsOutputName);
        log.info("Found {} present key outputs, {} present value outputs", presentKeyNames.size(), presentValueNames.size());

        // Sort to ensure consistent ordering
        java.util.Collections.sort(presentKeyNames);
        java.util.Collections.sort(presentValueNames);

        // For first step, use full prompt embeddings (with vision tokens injected)
        INDArray currentEmbeddings = inputsEmbeds;
        INDArray currentInputIds = promptTokenIdsTensor;
        int maxTokens = Math.min(20, samplingConfig.getMaxNewTokens());

        // KV cache storage - maps from layer index to cached tensors
        Map<String, INDArray> kvCache = new java.util.HashMap<>();
        long pastSeqLen = 0;

        for (int step = 0; step < maxTokens; step++) {
            // Build decoder inputs for this step
            Map<String, INDArray> decoderInputMap = new java.util.HashMap<>();

            long currentSeqLen = currentEmbeddings.shape()[1];
            long totalSeqLen = currentSeqLen + pastSeqLen;
            log.info("Step {}: currentSeqLen={}, pastSeqLen={}, totalSeqLen={}", step, currentSeqLen, pastSeqLen, totalSeqLen);

            for (String inputName : decoder.inputs()) {
                if (inputName.equals("inputs_embeds")) {
                    decoderInputMap.put(inputName, currentEmbeddings);
                } else if (inputName.equals("attention_mask")) {
                    INDArray mask = Nd4j.ones(DataType.LONG, batchSize, totalSeqLen);
                    decoderInputMap.put(inputName, mask);
                } else if (inputName.equals("input_ids")) {
                    decoderInputMap.put(inputName, currentInputIds);
                } else if (inputName.equals("position_ids")) {
                    INDArray posIds = Nd4j.arange(pastSeqLen, pastSeqLen + currentSeqLen)
                            .reshape(1, currentSeqLen).castTo(DataType.LONG);
                    decoderInputMap.put(inputName, posIds);
                } else if (inputName.startsWith("past_key_values.")) {
                    // Use cached KV if available, otherwise create empty tensor
                    // Map past_key_values.X.key -> present.X.key from cache
                    String presentName = inputName.replace("past_key_values", "present");
                    if (kvCache.containsKey(presentName)) {
                        decoderInputMap.put(inputName, kvCache.get(presentName));
                    } else {
                        // First step: empty KV cache with shape [batch, num_heads, 0, head_dim]
                        INDArray emptyKv = createEmptyKvCache(decoder, inputName, batchSize, visionHiddenSize);
                        decoderInputMap.put(inputName, emptyKv);
                    }
                }
            }

            // Run decoder - get ALL outputs including present KV values
            List<String> allOutputs = new java.util.ArrayList<>();
            allOutputs.add(logitsOutputName);
            allOutputs.addAll(presentKeyNames);
            allOutputs.addAll(presentValueNames);
            String layernormOutName = "/model/layers.0/input_layernorm/output_0";
            if (debugEmbeds && decoder.hasVariable(layernormOutName)) {
                allOutputs.add(layernormOutName);
            }

            Map<String, INDArray> decoderOutputs = decoder.output(decoderInputMap, allOutputs.toArray(new String[0]));

            INDArray logitsRaw = decoderOutputs.get(logitsOutputName);
            if (logitsRaw == null) {
                log.error("No logits output found");
                break;
            }
            INDArray logits = logitsRaw.dup();  // IMPORTANT: dup to avoid workspace reuse issues
            if (debugEmbeds && decoderOutputs.containsKey("/model/layers.0/input_layernorm/output_0")) {
                INDArray ln = decoderOutputs.get("/model/layers.0/input_layernorm/output_0");
                if (ln != null) {
                    log.info("Step {} layernorm0 stats: min={}, max={}, mean={}",
                            step, ln.minNumber(), ln.maxNumber(), ln.meanNumber());
                }
            }

            // Store present KV values for next step
            for (String presentName : presentKeyNames) {
                INDArray presentVal = decoderOutputs.get(presentName);
                if (presentVal != null) {
                    kvCache.put(presentName, presentVal.dup());
                }
            }
            for (String presentName : presentValueNames) {
                INDArray presentVal = decoderOutputs.get(presentName);
                if (presentVal != null) {
                    kvCache.put(presentName, presentVal.dup());
                }
            }
            if (step == 0) {
                log.info("Stored {} KV cache entries", kvCache.size());
            }

            // Get logits for the last position: [batch, seq, vocab] -> [vocab]
            INDArray lastLogits;
            if (logits.rank() == 3) {
                long lastPos = logits.size(1) - 1;
                lastLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPos), NDArrayIndex.all());
            } else if (logits.rank() == 2) {
                lastLogits = logits.getRow(0);
            } else {
                log.error("Unexpected logits shape: {}", java.util.Arrays.toString(logits.shape()));
                break;
            }

            // IMPORTANT: Sample BEFORE any debug decoder calls that may corrupt workspace memory
            // Make a safe copy of logits for sampling since debug code below may run another decoder.output()
            INDArray logitsForSampling = lastLogits.dup();
            int nextTokenId = sampler.sample(logitsForSampling);

            if (step == 0) {
                // Log top-k tokens and basic stats for debugging
                log.info("Step 0 logits stats: min={}, max={}, mean={}",
                        logitsForSampling.minNumber(), logitsForSampling.maxNumber(), logitsForSampling.meanNumber());
                INDArray probs = SamplerUtils.softmax(logitsForSampling.dup());
                INDArray[] topK = SamplerUtils.topK(logitsForSampling, 5);
                log.info("Step 0 top-5 tokens:");
                for (int i = 0; i < 5; i++) {
                    int idx = topK[0].getInt(i);
                    double logitVal = topK[1].getDouble(i);
                    double probVal = probs.getDouble(idx);
                    String tok = tokenizer.decode(new int[]{idx}, false);
                    log.info("  #{}: id={}, logit={}, prob={}, text='{}'", i + 1, idx, logitVal, probVal, tok);
                }
                log.info("Sampled token: id={}", nextTokenId);

                if (debugEmbeds) {
                    // Compare logits with zeroed inputs_embeds to verify embeddings are used
                    // NOTE: This runs another decoder.output() which corrupts workspace memory
                    // That's why we sample BEFORE this block using the dup'd logitsForSampling

                    Map<String, INDArray> zeroInputMap = new java.util.HashMap<>(decoderInputMap);
                    zeroInputMap.put("inputs_embeds", Nd4j.zerosLike(currentEmbeddings));
                    Map<String, INDArray> zeroOutputs = decoder.output(zeroInputMap, new String[]{logitsOutputName});
                    INDArray zeroLogits = zeroOutputs.get(logitsOutputName);
                    if (zeroLogits != null) {
                        INDArray zeroLast;
                        if (zeroLogits.rank() == 3) {
                            long lastPos = zeroLogits.size(1) - 1;
                            zeroLast = zeroLogits.get(NDArrayIndex.point(0), NDArrayIndex.point(lastPos), NDArrayIndex.all()).dup();
                        } else if (zeroLogits.rank() == 2) {
                            zeroLast = zeroLogits.getRow(0).dup();
                        } else {
                            zeroLast = zeroLogits.dup();
                        }
                        double diff = logitsForSampling.sub(zeroLast).norm2Number().doubleValue();
                        log.info("Step 0 logits diff vs zero-embed (L2): {}", diff);
                    } else {
                        log.warn("Zero-embed logits not produced");
                    }
                }
            }
            generatedTokens.add(nextTokenId);

            // Decode and log the token
            String tokenText = tokenizer.decode(new int[]{nextTokenId}, false);  // Keep special tokens for DocTags
            log.info("Step {}: token_id={}, text='{}'", step, nextTokenId, tokenText);

            // Check for EOS
            if (nextTokenId == eosTokenId) {
                log.info("EOS token generated at step {}", step);
                break;
            }

            // CRITICAL: Release memory from decoder forward pass
            // Close the raw logits (we have a dup)
            logitsRaw.close();
            // Close decoder outputs that we already dup'd into kvCache
            for (var entry : decoderOutputs.entrySet()) {
                String key = entry.getKey();
                // Don't close if we're still using it
                if (!key.equals(logitsOutputName)) {
                    entry.getValue().close();
                }
            }
            decoder.clearPlaceholders(false);
            decoder.clearOpInputs();

            // Update for next step: get embedding for the new token
            INDArray newTokenTensor = Nd4j.createFromArray(new int[]{nextTokenId}).reshape(1, 1).castTo(DataType.LONG);
            Map<String, INDArray> newEmbedInputMap = Map.of(embedInputName, newTokenTensor);
            Map<String, INDArray> newEmbedOutputs = embedTokens.output(newEmbedInputMap, embedTokens.outputs().toArray(new String[0]));

            // Get the new token embedding for next step
            INDArray prevEmbeddings = currentEmbeddings;
            for (var entry : newEmbedOutputs.entrySet()) {
                currentEmbeddings = entry.getValue().dup();  // Shape: [1, 1, hidden_size]
                entry.getValue().close();  // Close the original
            }
            // Close previous embeddings
            if (prevEmbeddings != null) {
                prevEmbeddings.close();
            }
            currentInputIds = newTokenTensor;
            embedTokens.clearPlaceholders(false);
            embedTokens.clearOpInputs();

            // Update past sequence length for attention mask
            pastSeqLen += currentSeqLen;
        }

        // Decode all generated tokens to text
        int[] tokenIds = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String generatedText = tokenizer.decode(tokenIds, false);  // Keep special tokens for DocTags

        log.info("=== Generated DocTags Text ===");
        log.info("Token count: {}", generatedTokens.size());
        log.info("Token IDs: {}", generatedTokens);
        log.info("Generated text: {}", generatedText);

        // Basic validation
        assertNotNull(generatedText, "Generated text should not be null");
        assertTrue(generatedTokens.size() > 0, "Should have generated at least one token");

        // Clean up tokenizer
        tokenizer.close();

        log.info("=== SmolDocling Pipeline Complete ===");
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

            // Check for EOS
            if (nextToken == eosToken) {
                log.info("EOS token generated at step {}", step);
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
     * Expand a single <image> token into a sequence that matches the vision token count.
     */
    private int[] expandImageTokens(int[] promptTokenIds, int imageTokenId, int imageTokenCount) {
        if (imageTokenCount <= 0) {
            return promptTokenIds;
        }
        int firstImagePos = -1;
        int occurrences = 0;
        for (int i = 0; i < promptTokenIds.length; i++) {
            if (promptTokenIds[i] == imageTokenId) {
                occurrences++;
                if (firstImagePos < 0) {
                    firstImagePos = i;
                }
            }
        }
        if (firstImagePos < 0) {
            log.warn("No <image> token found for expansion");
            return promptTokenIds;
        }

        int newLen = promptTokenIds.length - 1 + imageTokenCount;
        int[] expanded = new int[newLen];
        int outIdx = 0;
        for (int i = 0; i < promptTokenIds.length; i++) {
            int id = promptTokenIds[i];
            if (id == imageTokenId && i == firstImagePos) {
                for (int k = 0; k < imageTokenCount; k++) {
                    expanded[outIdx++] = imageTokenId;
                }
            } else {
                expanded[outIdx++] = id;
            }
        }

        if (occurrences > 1) {
            log.warn("Found {} <image> tokens; only expanded the first one", occurrences);
        }
        return expanded;
    }

    private int countOccurrences(int[] ids, int targetId) {
        int count = 0;
        for (int id : ids) {
            if (id == targetId) {
                count++;
            }
        }
        return count;
    }

    private int resolveImageTokenId(Tokenizer tokenizer) {
        Integer id = tokenizer.getTokenId("<image>");
        if (id != null && id >= 0) {
            return id;
        }
        int[] encoded = tokenizer.encode("<image>", false).getIds();
        if (encoded.length == 1) {
            return encoded[0];
        }
        log.warn("Could not resolve <image> token id from tokenizer; using fallback 49190");
        return 49190;
    }

    /**
     * Build the expanded image prompt string following Idefics3 processor logic.
     * Uses row/col tokens and a global image token when split into tiles.
     */
    private String buildImagePromptString(int imageRows, int imageCols, int imageSeqLen) {
        String fake = "<fake_token_around_image>";
        String image = "<image>";
        String global = "<global-img>";

        if (imageRows <= 0 || imageCols <= 0) {
            StringBuilder sb = new StringBuilder();
            sb.append(fake).append(global);
            for (int i = 0; i < imageSeqLen; i++) {
                sb.append(image);
            }
            sb.append(fake);
            return sb.toString();
        }

        StringBuilder sb = new StringBuilder();
        for (int r = 1; r <= imageRows; r++) {
            for (int c = 1; c <= imageCols; c++) {
                sb.append(fake);
                sb.append("<row_").append(r).append("_col_").append(c).append(">");
                for (int i = 0; i < imageSeqLen; i++) {
                    sb.append(image);
                }
                sb.append(fake);
            }
            sb.append("\n");
        }

        sb.append("\n");
        sb.append(fake).append(global);
        for (int i = 0; i < imageSeqLen; i++) {
            sb.append(image);
        }
        sb.append(fake);

        return sb.toString();
    }

    private INDArray createEmptyKvCache(SameDiff decoder, String inputName, long batchSize, long hiddenSize) {
        long numHeads = -1;
        long headDim = -1;
        DataType kvType = DataType.FLOAT;

        SDVariable inputVar = decoder.getVariable(inputName);
        if (inputVar != null && inputVar.getShape() != null && inputVar.getShape().length >= 4) {
            long[] shape = inputVar.getShape();
            if (inputVar.dataType() != null) {
                kvType = inputVar.dataType();
            }
            if (shape[1] > 0) {
                numHeads = shape[1];
            }
            if (shape[3] > 0) {
                headDim = shape[3];
            }
        }

        if (numHeads <= 0 || headDim <= 0) {
            String presentName = inputName.replace("past_key_values", "present");
            SDVariable presentVar = decoder.getVariable(presentName);
            if (presentVar != null && presentVar.getShape() != null && presentVar.getShape().length >= 4) {
                long[] shape = presentVar.getShape();
                if (numHeads <= 0 && shape[1] > 0) {
                    numHeads = shape[1];
                }
                if (headDim <= 0 && shape[3] > 0) {
                    headDim = shape[3];
                }
            }
        }

        if (headDim <= 0 && numHeads > 0 && hiddenSize > 0) {
            headDim = Math.max(1, hiddenSize / numHeads);
        }
        if (numHeads <= 0 && headDim > 0 && hiddenSize > 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }
        if (headDim <= 0) {
            headDim = 64;
        }
        if (numHeads <= 0) {
            numHeads = Math.max(1, hiddenSize / headDim);
        }

        return Nd4j.zeros(kvType, batchSize, numHeads, 0, headDim);
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
     * - If vlm.test.pdf.maxPages is set: returns up to that many pages
     * - Otherwise: returns all pages
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
                // Load multiple pages
                int pagesToLoad = maxPages > 0 ? Math.min(maxPages, totalPages) : totalPages;
                log.info("Loading {} pages from PDF (DPI: {})", pagesToLoad, renderDpi);

                for (int i = 0; i < pagesToLoad; i++) {
                    log.info("Rendering page {}/{}", i + 1, pagesToLoad);
                    pages.add(renderer.renderImageWithDPI(i, renderDpi, ImageType.RGB));
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

    // ==================== Image Splitting for SmolDocling ====================

    /**
     * Split an image into tiles for SmolDocling/Idefics3 processing.
     * This implements the split_image logic from Idefics3ImageProcessor.
     *
     * The algorithm:
     * 1. If image is larger than maxSize in either dimension, split into tiles
     * 2. Each tile is padded to maxSize x maxSize (preserving scale)
     * 3. Also include a global resized+pad version of the full image
     *
     * @param image The input image
     * @param maxSize The maximum tile size (512 for SmolDocling)
     * @return SplitImageResult containing the frames and metadata
     */
    private SplitImageResult splitImageForVLM(BufferedImage image, int maxSize) {
        int width = image.getWidth();
        int height = image.getHeight();

        log.info("Splitting image {}x{} into {}x{} tiles (maxTiles={})", width, height, maxSize, maxSize,
                maxTiles > 0 ? maxTiles : "unlimited");

        List<BufferedImage> frames = new java.util.ArrayList<>();
        List<ContentRegion> contentRegions = new java.util.ArrayList<>();
        int numSplitsH = 0;
        int numSplitsW = 0;

        if (height > maxSize || width > maxSize) {
            // Calculate number of splits needed
            numSplitsH = (int) Math.ceil((double) height / maxSize);
            numSplitsW = (int) Math.ceil((double) width / maxSize);

            // If maxTiles is set, reduce the grid to fit within the limit
            // Reserve 1 slot for global image, so limit tiles to maxTiles - 1
            if (maxTiles > 0) {
                int maxTilesForGrid = Math.max(1, maxTiles - 1);  // At least 1 tile
                int totalTiles = numSplitsH * numSplitsW;
                if (totalTiles > maxTilesForGrid) {
                    // Find the best grid configuration that maximizes coverage
                    // For tall images, prefer more rows; for wide images, prefer more columns
                    double imageAspect = (double) height / width;  // > 1 for tall images

                    int bestH = 1, bestW = 1;
                    int bestCount = 1;
                    double bestAspectMatch = Double.MAX_VALUE;

                    // Try all valid grid configurations
                    for (int h = 1; h <= Math.min(numSplitsH, maxTilesForGrid); h++) {
                        int maxW = maxTilesForGrid / h;
                        for (int w = 1; w <= Math.min(numSplitsW, maxW); w++) {
                            int count = h * w;
                            if (count <= maxTilesForGrid) {
                                double gridAspect = (double) h / w;
                                double aspectMatch = Math.abs(Math.log(gridAspect) - Math.log(imageAspect));

                                // Prefer more tiles, with tie-breaker on aspect ratio match
                                if (count > bestCount ||
                                    (count == bestCount && aspectMatch < bestAspectMatch)) {
                                    bestH = h;
                                    bestW = w;
                                    bestCount = count;
                                    bestAspectMatch = aspectMatch;
                                }
                            }
                        }
                    }

                    numSplitsH = bestH;
                    numSplitsW = bestW;
                    log.info("Reduced grid from {}x{} to {}x{} ({} tiles) to fit maxTiles={}, imageAspect={}",
                            (int) Math.ceil((double) height / maxSize),
                            (int) Math.ceil((double) width / maxSize),
                            numSplitsH, numSplitsW, numSplitsH * numSplitsW, maxTiles, imageAspect);
                }
            }

            // Idefics3 row/col tokens are defined for up to 6x6 grid
            int maxGrid = 6;
            if (numSplitsH > maxGrid || numSplitsW > maxGrid) {
                double scale = Math.min((double) maxGrid / numSplitsH, (double) maxGrid / numSplitsW);
                int newH = Math.max(1, (int) Math.floor(numSplitsH * scale));
                int newW = Math.max(1, (int) Math.floor(numSplitsW * scale));
                newH = Math.min(maxGrid, newH);
                newW = Math.min(maxGrid, newW);
                log.info("Reducing grid {}x{} to {}x{} to fit row/col token limits",
                        numSplitsH, numSplitsW, newH, newW);
                numSplitsH = newH;
                numSplitsW = newW;
            }

            // Calculate optimal tile size to evenly divide the image
            int optimalHeight = (int) Math.ceil((double) height / numSplitsH);
            int optimalWidth = (int) Math.ceil((double) width / numSplitsW);

            log.info("Splitting into {}x{} grid ({} tiles), optimal tile size: {}x{}",
                    numSplitsH, numSplitsW, numSplitsH * numSplitsW, optimalHeight, optimalWidth);

            // Create tiles
            for (int r = 0; r < numSplitsH; r++) {
                for (int c = 0; c < numSplitsW; c++) {
                    int startX = c * optimalWidth;
                    int startY = r * optimalHeight;
                    int endX = Math.min(startX + optimalWidth, width);
                    int endY = Math.min(startY + optimalHeight, height);

                    int tileWidth = endX - startX;
                    int tileHeight = endY - startY;

                    // Crop the tile
                    BufferedImage tile = image.getSubimage(startX, startY, tileWidth, tileHeight);

                    int contentW = tileWidth;
                    int contentH = tileHeight;

                    // If tile is larger than maxSize (possible when maxTiles limits the grid), downscale to fit
                    if (tileWidth > maxSize || tileHeight > maxSize) {
                        ResizeResult resized = resizeToFit(tile, maxSize, maxSize);
                        tile = resized.image;
                        contentW = resized.width;
                        contentH = resized.height;
                    }

                    // Pad to maxSize x maxSize (preserve content scale)
                    if (contentW != maxSize || contentH != maxSize) {
                        tile = padToSize(tile, maxSize, maxSize);
                    }

                    frames.add(tile);
                    contentRegions.add(new ContentRegion(contentW, contentH));
                    log.info("  Tile [{},{}]: crop ({},{}) to ({},{}), content {}x{}, padded to {}x{}",
                            r, c, startX, startY, endX, endY, contentW, contentH, maxSize, maxSize);
                }
            }
        }

        // Always add the global resized image at the end (keep aspect ratio, pad to square)
        ResizeResult globalResize = resizeToFit(image, maxSize, maxSize);
        BufferedImage globalImage = padToSize(globalResize.image, maxSize, maxSize);
        frames.add(globalImage);
        contentRegions.add(new ContentRegion(globalResize.width, globalResize.height));
        log.info("  Added global resized image ({}x{})", maxSize, maxSize);

        log.info("Total frames: {} ({} tiles + 1 global)", frames.size(), frames.size() - 1);

        return new SplitImageResult(frames, contentRegions, numSplitsH, numSplitsW);
    }

    /**
     * Resize image so that the longest edge matches the target length.
     */
    private BufferedImage resizeLongestEdge(BufferedImage image, int longestEdge) {
        int width = image.getWidth();
        int height = image.getHeight();
        int maxDim = Math.max(width, height);
        if (maxDim == longestEdge) {
            return image;
        }
        double scale = (double) longestEdge / (double) maxDim;
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));
        return resizeImage(image, newW, newH);
    }

    /**
     * Resize an image to fit within the target size while preserving aspect ratio.
     */
    private ResizeResult resizeToFit(BufferedImage image, int targetWidth, int targetHeight) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width <= targetWidth && height <= targetHeight) {
            return new ResizeResult(image, width, height);
        }
        double scale = Math.min((double) targetWidth / (double) width, (double) targetHeight / (double) height);
        int newW = Math.max(1, (int) Math.round(width * scale));
        int newH = Math.max(1, (int) Math.round(height * scale));
        return new ResizeResult(resizeImage(image, newW, newH), newW, newH);
    }

    /**
     * Pad an image to the target size (top-left aligned).
     */
    private BufferedImage padToSize(BufferedImage image, int targetWidth, int targetHeight) {
        if (image.getWidth() == targetWidth && image.getHeight() == targetHeight) {
            return image;
        }
        BufferedImage padded = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = padded.createGraphics();
        g2d.setColor(Color.BLACK);
        g2d.fillRect(0, 0, targetWidth, targetHeight);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return padded;
    }

    /**
     * Resize an image to the specified dimensions using high-quality interpolation.
     */
    private BufferedImage resizeImage(BufferedImage original, int targetWidth, int targetHeight) {
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();

        // Use high-quality rendering hints
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();

        return resized;
    }

    private static class ResizeResult {
        final BufferedImage image;
        final int width;
        final int height;

        ResizeResult(BufferedImage image, int width, int height) {
            this.image = image;
            this.width = width;
            this.height = height;
        }
    }

    /**
     * Result of splitting an image into tiles.
     */
    private static class SplitImageResult {
        final List<BufferedImage> frames;
        final List<ContentRegion> contentRegions;
        final int numRows;
        final int numCols;

        SplitImageResult(List<BufferedImage> frames, List<ContentRegion> contentRegions, int numRows, int numCols) {
            this.frames = frames;
            this.contentRegions = contentRegions;
            this.numRows = numRows;
            this.numCols = numCols;
        }

        int getTileCount() {
            return numRows * numCols;
        }

        int getTotalFrames() {
            return frames.size();
        }
    }

    private static class ContentRegion {
        final int width;
        final int height;

        ContentRegion(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }

    private static class VisionOutput {
        final String name;
        final INDArray tensor;

        VisionOutput(String name, INDArray tensor) {
            this.name = name;
            this.tensor = tensor;
        }
    }

    /**
     * Preprocess multiple image frames for SmolDocling.
     * Each frame is normalized and converted to a tensor.
     *
     * @param frames List of BufferedImage frames
     * @param preprocessor The VLM preprocessor to use
     * @param targetSize The target size (512)
     * @return INDArray with shape [batch, numFrames, channels, height, width]
     */
    private INDArray preprocessFramesForSmolDocling(List<BufferedImage> frames,
                                                     VLMImagePreprocessor preprocessor,
                                                     int targetSize) {
        int numFrames = frames.size();
        INDArray result = Nd4j.create(DataType.FLOAT, 1, numFrames, 3, targetSize, targetSize);

        for (int f = 0; f < numFrames; f++) {
            BufferedImage frame = frames.get(f);

            // Preprocess this frame (resize + normalize)
            INDArray frameTensor = preprocessor.preprocess(frame);  // [1, 3, H, W]

            // Copy into the 5D tensor
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < targetSize; y++) {
                    for (int x = 0; x < targetSize; x++) {
                        float val = frameTensor.getFloat(0, c, y, x);
                        result.putScalar(new long[]{0, f, c, y, x}, val);
                    }
                }
            }

            log.info("Preprocessed frame {}/{}: min={}, max={}, mean={}",
                    f + 1, numFrames,
                    frameTensor.minNumber(), frameTensor.maxNumber(), frameTensor.meanNumber());
        }

        return result;
    }

    /**
     * Create a pixel attention mask for a padded frame.
     */
    private INDArray createPixelAttentionMask(int contentWidth, int contentHeight, int targetSize) {
        if (contentWidth >= targetSize && contentHeight >= targetSize) {
            return Nd4j.ones(DataType.BOOL, 1, 1, targetSize, targetSize);
        }
        INDArray mask = Nd4j.zeros(DataType.BOOL, 1, 1, targetSize, targetSize);
        if (contentWidth > 0 && contentHeight > 0) {
            mask.get(
                    NDArrayIndex.point(0),
                    NDArrayIndex.point(0),
                    NDArrayIndex.interval(0, contentHeight),
                    NDArrayIndex.interval(0, contentWidth)
            ).assign(1);
        }
        return mask;
    }

    /**
     * Select the best vision output tensor (prefer last_hidden_state).
     */
    private VisionOutput selectVisionOutput(Map<String, INDArray> outputs) {
        if (outputs == null || outputs.isEmpty()) {
            return null;
        }

        for (var entry : outputs.entrySet()) {
            String name = entry.getKey();
            INDArray tensor = entry.getValue();
            if (name.contains("last_hidden_state") && tensor != null && tensor.rank() == 3) {
                return new VisionOutput(name, tensor);
            }
        }

        VisionOutput best = null;
        for (var entry : outputs.entrySet()) {
            INDArray tensor = entry.getValue();
            if (tensor != null && tensor.rank() == 3) {
                if (best == null || tensor.size(1) > best.tensor.size(1)) {
                    best = new VisionOutput(entry.getKey(), tensor);
                }
            }
        }

        return best;
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

    private boolean isAllZeroOrNaN(INDArray arr) {
        double min = arr.minNumber().doubleValue();
        double max = arr.maxNumber().doubleValue();
        if (Double.isNaN(min) || Double.isNaN(max) || Double.isInfinite(min) || Double.isInfinite(max)) {
            return true;
        }
        return min == 0.0 && max == 0.0;
    }

    private void fixDecoderInputsEmbeds(SameDiff decoder) {
        String targetOpName = null;
        for (String opName : decoder.getOps().keySet()) {
            if (opName.contains("/model/layers.0/input_layernorm") && opName.contains("LayerNorm")) {
                targetOpName = opName;
                break;
            }
        }
        if (targetOpName == null) {
            for (String opName : decoder.getOps().keySet()) {
                if (opName.contains("input_layernorm")) {
                    targetOpName = opName;
                    break;
                }
            }
        }
        if (targetOpName == null) {
            log.warn("Could not locate input_layernorm op to wire inputs_embeds");
            return;
        }

        org.nd4j.autodiff.samediff.internal.SameDiffOp op = decoder.getOps().get(targetOpName);
        if (op == null || op.getInputsToOp() == null || op.getInputsToOp().isEmpty()) {
            log.warn("input_layernorm op '{}' has no inputs", targetOpName);
            return;
        }

        List<String> inputs = new java.util.ArrayList<>(op.getInputsToOp());
        String oldInput = inputs.get(0);
        if ("inputs_embeds".equals(oldInput)) {
            log.info("inputs_embeds already wired into '{}'", targetOpName);
            return;
        }

        log.info("Rewiring '{}' input[0] from '{}' to 'inputs_embeds'", targetOpName, oldInput);
        inputs.set(0, "inputs_embeds");
        op.setInputsToOp(inputs);

        org.nd4j.autodiff.samediff.internal.Variable inputVar = decoder.getVariables().get("inputs_embeds");
        if (inputVar != null) {
            List<String> inputsForOp = inputVar.getInputsForOp();
            if (inputsForOp == null) {
                inputsForOp = new java.util.ArrayList<>();
                inputVar.setInputsForOp(inputsForOp);
            }
            if (!inputsForOp.contains(targetOpName)) {
                inputsForOp.add(targetOpName);
            }
        }

        org.nd4j.autodiff.samediff.internal.Variable oldVar = decoder.getVariables().get(oldInput);
        if (oldVar != null && oldVar.getInputsForOp() != null) {
            oldVar.getInputsForOp().remove(targetOpName);
        }
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
     * Fix the decoder model's baked-in input_ids constant.
     *
     * The ONNX model has input_ids baked in as a constant during export.
     * This method rewires the graph so that the input_ids_subgraph output
     * dynamically matches the attention_mask shape.
     *
     * The issue: /model/attn_mask_reformat/Add fails because:
     * - attn_mask_subgraph/Where_2/output_0 has shape [1, 1, 1, N] (from attention_mask)
     * - input_ids_subgraph/Expand/output_0 has shape [1, 1, 1, 4] (baked-in constant)
     *
     * The fix: Replace the input to the Add operation with a dynamic zeros tensor
     * that matches the attention_mask derived shape.
     *
     * @param decoder The SameDiff decoder model to fix
     * @param tokenizer The tokenizer (used to get special token IDs)
     */
    private void fixDecoderInputIds(SameDiff decoder, Tokenizer tokenizer) {
        log.info("=== Fixing decoder input_ids ===");

        // Find the Add operation that combines attention masks
        String addOpName = "/model/attn_mask_reformat/Add";
        org.nd4j.autodiff.samediff.internal.SameDiffOp addOp = decoder.getOps().get(addOpName);

        if (addOp == null) {
            log.warn("Could not find Add operation at {}, listing available ops with 'Add':", addOpName);
            for (String opName : decoder.getOps().keySet()) {
                if (opName.contains("Add") && opName.contains("attn_mask")) {
                    log.info("  Found op: {}", opName);
                    addOp = decoder.getOps().get(opName);
                    addOpName = opName;
                }
            }
        }

        if (addOp == null) {
            log.error("Could not find the Add operation for attention mask reformatting");
            return;
        }

        log.info("Found Add operation: {}", addOpName);
        log.info("  Inputs: {}", addOp.getInputsToOp());
        log.info("  Outputs: {}", addOp.getOutputsOfOp());

        // Find which input is the input_ids_subgraph output (the one with wrong shape)
        String inputIdsSubgraphOutput = null;
        String attnMaskSubgraphOutput = null;
        int inputIdsIndex = -1;

        for (int i = 0; i < addOp.getInputsToOp().size(); i++) {
            String inputName = addOp.getInputsToOp().get(i);
            if (inputName.contains("input_ids_subgraph")) {
                inputIdsSubgraphOutput = inputName;
                inputIdsIndex = i;
                log.info("  Input[{}] is input_ids_subgraph: {}", i, inputName);
            } else if (inputName.contains("attn_mask_subgraph")) {
                attnMaskSubgraphOutput = inputName;
                log.info("  Input[{}] is attn_mask_subgraph: {}", i, inputName);
            }
        }

        if (inputIdsSubgraphOutput == null || attnMaskSubgraphOutput == null) {
            log.error("Could not identify input_ids_subgraph and attn_mask_subgraph outputs");
            log.error("  inputIdsSubgraphOutput: {}", inputIdsSubgraphOutput);
            log.error("  attnMaskSubgraphOutput: {}", attnMaskSubgraphOutput);
            return;
        }

        // Get the attn_mask_subgraph output variable to match its shape
        SDVariable attnMaskOutput = decoder.getVariable(attnMaskSubgraphOutput);
        if (attnMaskOutput == null) {
            log.error("Could not find variable: {}", attnMaskSubgraphOutput);
            return;
        }

        log.info("attn_mask_subgraph output shape: {}", java.util.Arrays.toString(attnMaskOutput.getShape()));

        // Create a zeros tensor that matches the attn_mask_subgraph output shape
        // Use zerosLike which dynamically computes the shape at runtime
        SDVariable dynamicZeros = decoder.zerosLike("_fix_input_ids_zeros", attnMaskOutput);

        // Replace the input in the Add operation
        log.info("Replacing input[{}] '{}' with '{}'", inputIdsIndex, inputIdsSubgraphOutput, dynamicZeros.name());

        // Update the inputsToOp list
        List<String> newInputs = new java.util.ArrayList<>(addOp.getInputsToOp());
        newInputs.set(inputIdsIndex, dynamicZeros.name());
        addOp.setInputsToOp(newInputs);

        // Update the variable's inputsForOp tracking
        org.nd4j.autodiff.samediff.internal.Variable zerosVar = decoder.getVariables().get(dynamicZeros.name());
        if (zerosVar != null) {
            List<String> inputsForOp = zerosVar.getInputsForOp();
            if (inputsForOp == null) {
                inputsForOp = new java.util.ArrayList<>();
                zerosVar.setInputsForOp(inputsForOp);
            }
            inputsForOp.add(addOpName);
        }

        // Remove the old variable from being an input for this op
        org.nd4j.autodiff.samediff.internal.Variable oldVar = decoder.getVariables().get(inputIdsSubgraphOutput);
        if (oldVar != null && oldVar.getInputsForOp() != null) {
            oldVar.getInputsForOp().remove(addOpName);
        }

        log.info("=== Fix complete ===");
        log.info("Add operation now has inputs: {}", addOp.getInputsToOp());
    }

    /**
     * Fix the repeat_kv Reshape operations in the decoder model.
     *
     * The ONNX model uses shape [0, 0, 3, -1] for reshaping k/v projections,
     * where 0 means "copy from input". The Reshape import hook handles this
     * by creating dynamic operations, but there may be issues with how these
     * execute at runtime.
     *
     * This method diagnoses the reshape constants and fixes them if needed.
     */
    private void fixRepeatKVReshape(SameDiff decoder) {
        log.info("=== Checking repeat_kv Reshape operations ===");

        // Find Reshape operations in repeat_kv paths
        for (String opName : decoder.getOps().keySet()) {
            if (opName.contains("repeat_kv") && opName.contains("Reshape_1")) {
                org.nd4j.autodiff.samediff.internal.SameDiffOp op = decoder.getOps().get(opName);
                log.info("Found Reshape: {}", opName);
                log.info("  Inputs: {}", op.getInputsToOp());
                log.info("  Outputs: {}", op.getOutputsOfOp());

                // Check the shape input
                if (op.getInputsToOp().size() >= 2) {
                    String shapeInputName = op.getInputsToOp().get(1);
                    SDVariable shapeVar = decoder.getVariable(shapeInputName);
                    if (shapeVar != null) {
                        log.info("  Shape variable: {}", shapeInputName);
                        log.info("  Shape var type: {}", shapeVar.getVariableType());
                        INDArray shapeArr = shapeVar.getArr();
                        if (shapeArr != null) {
                            log.info("  Shape values: {}", shapeArr);
                        } else {
                            log.info("  Shape array is null (computed at runtime)");
                        }
                    }
                }

                // Only check first layer for now
                break;
            }
        }

        // Check the shape constants
        log.info("\n=== Checking shape constants ===");
        for (String varName : decoder.getVariables().keySet()) {
            if (varName.contains("0, 0, 3, -1") || varName.contains("constants") && varName.contains("INT64")) {
                SDVariable var = decoder.getVariable(varName);
                log.info("Constant: {}", varName);
                log.info("  Type: {}", var.getVariableType());
                INDArray arr = var.getArr();
                if (arr != null) {
                    log.info("  Values: {}", arr);
                }
            }
        }

        log.info("=== repeat_kv check complete ===");
    }

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
}
