/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipelineConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.output.DocTagsParser;
import org.eclipse.deeplearning4j.vlm.output.DocumentStructure;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * VLM (SmolDocling document extraction) facade for the SDX AOT exports.
 *
 * <p>Contract consumed by {@link SdxLlmCli} ({@code vlm} subcommand) and
 * {@link SdxLlmCApi} ({@code sdxVlmExtract}): keep the signature stable.</p>
 *
 * <p>Implements the canonical SmolDocling pipeline as exercised by the validated
 * examples (SmolDoclingVLMExample, SmolDoclingPdfToMarkdownExample):
 * <ol>
 *   <li>Resolve model files (ONNX or cached SDZ) from modelPath directory</li>
 *   <li>Resolve tokenizer.json (from tokenizerPath or the model directory)</li>
 *   <li>Import models via OnnxModelCache (SDZ-caching: skip ONNX parse on repeat runs)</li>
 *   <li>Load each input page as a BufferedImage (JPEG/PNG directly; PDF via PDFBox)</li>
 *   <li>Tile each page with ImageTiler, preprocess frames with VisionEncoderUtils</li>
 *   <li>Encode tiles with VisionEncoder (pixel_values + pixel_attention_mask)</li>
 *   <li>Assemble chat prompt with ImagePromptBuilder and merge embeddings</li>
 *   <li>Decode with GenerationPipeline (reused across pages)</li>
 *   <li>Format output: doctags (raw), markdown (via DocTagsParser), or text (stripped)</li>
 * </ol>
 * </p>
 *
 * <p>Options JSON keys (all optional):
 * <ul>
 *   <li>{@code maxNewTokens} int — decode budget per page (default 1024)</li>
 *   <li>{@code prompt} String — instruction text (default "Convert this page to docling.")</li>
 *   <li>{@code format} String — "doctags" | "markdown" | "text" (default "doctags")</li>
 * </ul>
 * </p>
 *
 * <p>PDF support: requires Apache PDFBox on the classpath (it is a transitive dependency
 * of samediff-vlm via the PDF example; if absent a clear exception is thrown).</p>
 */
final class SdxVlm {

    private SdxVlm() {
    }

    // Standard SmolDocling tile size (matches preprocessor_config.json target_height/width).
    private static final int DEFAULT_TILE_SIZE = 512;

    /**
     * Run VLM document extraction on an image or PDF.
     *
     * @param modelPath     VLM model directory (contains vision_encoder.onnx / .sdz,
     *                      decoder_model_merged.onnx / .sdz, embed_tokens.onnx / .sdz).
     *                      Also accepts a path to any one of the component files —
     *                      the parent directory is then used.
     * @param tokenizerPath optional tokenizer path (tokenizer.json file or directory
     *                      containing tokenizer.json); null tries the model dir
     * @param inputPath     input image (png/jpg) or PDF
     * @param optionsJson   optional JSON: {"maxNewTokens":N,"prompt":"...","format":"doctags|markdown|text"}
     * @return the extracted document text in the requested format
     */
    static String extract(String modelPath, String tokenizerPath, String inputPath, String optionsJson)
            throws Exception {

        // ---- 1. Parse options (shaded Jackson tree API only — no POJO databind) ----
        int maxNewTokens = 1024;
        String userPrompt = "Convert this page to docling.";
        String format = "doctags";

        if (optionsJson != null && !optionsJson.isEmpty()) {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(optionsJson);
            if (root.has("maxNewTokens")) {
                maxNewTokens = root.get("maxNewTokens").asInt(maxNewTokens);
            }
            if (root.has("prompt") && !root.get("prompt").isNull()) {
                userPrompt = root.get("prompt").asText(userPrompt);
            }
            if (root.has("format") && !root.get("format").isNull()) {
                format = root.get("format").asText(format).toLowerCase();
            }
        }

        // ---- 2. Resolve model directory ----
        File modelDir = resolveModelDir(modelPath);

        // ---- 3. Locate component files ----
        File visionEncoderFile = findModelFile(modelDir,
                "vision_encoder.onnx", "vision_encoder.sdz",
                "vision_encoder.opt.sdz");
        File decoderFile = findModelFile(modelDir,
                "decoder_model_merged.onnx", "decoder_model_merged.sdz",
                "decoder_model_merged.opt.sdz",
                "decoder.onnx", "decoder.sdz");
        File embedTokensFile = findModelFile(modelDir,
                "embed_tokens.onnx", "embed_tokens.sdz",
                "embed_tokens.opt.sdz");

        // ---- 4. Resolve tokenizer ----
        File tokenizerFile = resolveTokenizerFile(tokenizerPath, modelDir);

        // ---- 5. Resolve preprocessor config (tile size) ----
        int tileSize = DEFAULT_TILE_SIZE;
        File ppConfigFile = findFileOrNull(modelDir,
                "preprocessor_config.json", "smoldocling-preprocessor-config.json");
        if (ppConfigFile != null) {
            try {
                PreprocessorConfig ppConfig = PreprocessorConfig.fromFile(ppConfigFile);
                tileSize = ppConfig.getTargetHeight();
            } catch (Exception e) {
                // Fall back to default tile size; non-fatal
            }
        }

        // ---- 6. Load models (SDZ-cached on repeat runs) ----
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionEncoderFile.getAbsolutePath(),
                decoderFile.getAbsolutePath(),
                embedTokensFile.getAbsolutePath());
        SameDiff visionEncoderSd = models[0];
        SameDiff decoder = models[1];
        SameDiff embedTokensSd = models[2];

        // ---- 7. Load tokenizer ----
        Tokenizer tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile);

        // ---- 8. Render input to page images ----
        List<BufferedImage> pages = renderInputToPages(inputPath);
        if (pages.isEmpty()) {
            throw new IllegalArgumentException("No pages could be rendered from: " + inputPath);
        }

        // ---- 9. Create a single GenerationPipeline reused across all pages ----
        final int finalMaxNewTokens = maxNewTokens;
        StringBuilder documentOutput = new StringBuilder();
        DocTagsParser parser = new DocTagsParser();

        GenerationPipelineConfig pipelineConfig = GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokensSd)
                .tokenizer(tokenizer)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxNewTokens)
                .build();

        try (GenerationPipeline pipeline = GenerationPipeline.create(pipelineConfig)) {
            for (int pageIdx = 0; pageIdx < pages.size(); pageIdx++) {
                BufferedImage pageImage = pages.get(pageIdx);

                // 9a. Tile the page image
                ImageTiler.SplitImageResult split = ImageTiler.splitImageForVLM(pageImage, tileSize);
                int frames = split.getTotalFrames();

                // 9b. Preprocess all frames into [1, frames, 3, tileSize, tileSize]
                VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(
                        ppConfigFile != null ? PreprocessorConfig.fromFile(ppConfigFile)
                                : defaultPreprocessorConfig(tileSize));
                INDArray imageInput = VisionEncoderUtils.preprocessFrames(split.frames, preprocessor, tileSize);
                preprocessor.shutdown();

                // 9c. Encode tiles through the SigLIP vision encoder
                VisionEncoder visionEncoder = VisionEncoder.builder()
                        .model(visionEncoderSd)
                        .targetSize(tileSize)
                        .build();
                VisionEncoder.Result visionResult = visionEncoder.encode(imageInput, frames, split);
                INDArray visionEmbeddings = visionResult.getEmbeddings();
                imageInput.close();

                // 9d. Build chat prompt with tile grid tokens
                int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
                int seqPerFrame = (int) (visionEmbeddings.size(1) / frames);
                String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                        split.numRows, split.numCols, seqPerFrame);
                ChatTemplate.Message promptMessage = ChatTemplate.Message.userMultimodal(
                        List.of(
                                ChatTemplate.ContentPart.image(imagePrompt),
                                ChatTemplate.ContentPart.text(userPrompt)));
                String chatPrompt = tokenizer.applyChatTemplate(
                        List.of(promptMessage), true);

                // 9e. Encode prompt tokens and merge embeddings
                int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();
                INDArray textEmbeddings = pipeline.embedTokens(promptTokenIds);
                INDArray mergedEmbeddings = EmbeddingMerger.mergeEmbeddings(
                        textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);

                // 9f. Decode
                GenerationResult result = pipeline.generate(mergedEmbeddings, promptTokenIds, finalMaxNewTokens);
                String rawDocTags = result.getText();

                // 9g. Format output
                String pageOutput = formatOutput(rawDocTags, format, parser);

                if (pageIdx > 0 && pages.size() > 1) {
                    documentOutput.append("\n\n");
                }
                documentOutput.append(pageOutput);
            }
        } finally {
            tokenizer.close();
        }

        return documentOutput.toString();
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /**
     * Resolve the model directory from a path that may point to either a
     * directory or one of the component files inside the directory.
     */
    private static File resolveModelDir(String modelPath) {
        File f = new File(modelPath);
        if (f.isDirectory()) {
            return f;
        }
        // If it's a file, use its parent directory
        if (f.isFile()) {
            return f.getParentFile();
        }
        // Doesn't exist yet — treat as directory path
        return f;
    }

    /**
     * Find the first existing file among the given candidate names in modelDir.
     * Tries names in order; SDZ caches are preferred for load speed but ONNX is
     * the authoritative fallback because OnnxModelCache handles the SDZ caching.
     */
    private static File findModelFile(File dir, String... candidates) throws Exception {
        for (String name : candidates) {
            File f = new File(dir, name);
            if (f.isFile()) {
                return f;
            }
        }
        // Also check in the dl4j-vlm-models flat-file cache (separate file naming)
        // e.g. ~/.cache/dl4j-vlm-models/smoldocling-vision-encoder.onnx
        String homeCache = System.getProperty("user.home") + "/.cache/dl4j-vlm-models";
        for (String name : candidates) {
            // Prefix common name patterns with "smoldocling-" for the flat cache
            String flatName = name.replace("vision_encoder", "smoldocling-vision-encoder")
                    .replace("decoder_model_merged", "smoldocling-decoder")
                    .replace("embed_tokens", "smoldocling-embed-tokens");
            File f = new File(homeCache, flatName);
            if (f.isFile()) {
                return f;
            }
        }
        throw new IllegalArgumentException(
                "Could not find any of " + java.util.Arrays.toString(candidates)
                        + " in " + dir.getAbsolutePath()
                        + " or ~/.cache/dl4j-vlm-models/");
    }

    /** Like findModelFile but returns null if not found (for optional files). */
    private static File findFileOrNull(File dir, String... candidates) {
        for (String name : candidates) {
            File f = new File(dir, name);
            if (f.isFile()) {
                return f;
            }
        }
        String homeCache = System.getProperty("user.home") + "/.cache/dl4j-vlm-models";
        for (String name : candidates) {
            File f = new File(homeCache, name);
            if (f.isFile()) {
                return f;
            }
        }
        return null;
    }

    /**
     * Resolve the tokenizer.json file.
     * Priority: explicit tokenizerPath (file or dir) > modelDir/tokenizer.json >
     * ~/.cache/dl4j-vlm-models/smoldocling-tokenizer.json.
     */
    private static File resolveTokenizerFile(String tokenizerPath, File modelDir) throws Exception {
        if (tokenizerPath != null && !tokenizerPath.isEmpty()) {
            File tf = new File(tokenizerPath);
            if (tf.isFile()) {
                return tf;
            }
            // Might be a directory
            File inDir = new File(tf, "tokenizer.json");
            if (inDir.isFile()) {
                return inDir;
            }
        }
        // Try model dir
        File inModel = new File(modelDir, "tokenizer.json");
        if (inModel.isFile()) {
            return inModel;
        }
        // Try flat cache
        File cached = new File(System.getProperty("user.home")
                + "/.cache/dl4j-vlm-models/smoldocling-tokenizer.json");
        if (cached.isFile()) {
            return cached;
        }
        throw new IllegalArgumentException(
                "Could not find tokenizer.json in: " + modelDir.getAbsolutePath()
                        + "; pass --tokenizer <path> or ensure the file exists in the model directory.");
    }

    /**
     * Render the input file to a list of page images.
     *
     * <p>Supported inputs:
     * <ul>
     *   <li>JPEG / PNG / BMP / GIF / TIFF — returned as a single-element list</li>
     *   <li>PDF — each page rendered at 150 DPI via Apache PDFBox (loaded reflectively
     *       so the binary compiles even when PDFBox is absent at build time — it IS
     *       present at runtime because samediff-vlm depends on it transitively)</li>
     * </ul>
     * </p>
     */
    private static List<BufferedImage> renderInputToPages(String inputPath) throws Exception {
        File input = new File(inputPath);
        if (!input.exists()) {
            throw new IllegalArgumentException("Input file does not exist: " + inputPath);
        }

        String lower = inputPath.toLowerCase();
        List<BufferedImage> pages = new ArrayList<>();

        if (lower.endsWith(".pdf")) {
            pages.addAll(renderPdfPages(input));
        } else {
            // Standard image formats handled by javax.imageio.ImageIO
            BufferedImage img = ImageIO.read(input);
            if (img == null) {
                throw new IllegalArgumentException(
                        "Could not read image from: " + inputPath
                                + " (unsupported format or corrupt file)");
            }
            pages.add(img);
        }
        return pages;
    }

    /**
     * Render all pages of a PDF to BufferedImages using Apache PDFBox.
     *
     * <p>PDFBox is loaded reflectively so that this class compiles against the
     * sdx-aot classpath even if PDFBox is excluded at image build time. In practice
     * PDFBox IS present (samediff-vlm pulls it in for the PDF example). If it is
     * missing the reflective load throws a clear ClassNotFoundException.</p>
     */
    private static List<BufferedImage> renderPdfPages(File pdfFile) throws Exception {
        List<BufferedImage> pages = new ArrayList<>();
        // Use reflection to avoid a hard compile-time dep on PDFBox classes.
        // If PDFBox is absent on the classpath a clear ClassNotFoundException is thrown.
        Class<?> pdDocClass = Class.forName("org.apache.pdfbox.pdmodel.PDDocument");
        Class<?> rendererClass = Class.forName("org.apache.pdfbox.rendering.PDFRenderer");
        Class<?> imageTypeClass = Class.forName("org.apache.pdfbox.rendering.ImageType");
        Object rgbConst = imageTypeClass.getField("RGB").get(null);

        // PDDocument.load(File)
        Object doc = pdDocClass.getMethod("load", File.class).invoke(null, pdfFile);
        try {
            int numPages = (int) pdDocClass.getMethod("getNumberOfPages").invoke(doc);
            Object renderer = rendererClass.getConstructor(pdDocClass).newInstance(doc);
            for (int i = 0; i < numPages; i++) {
                // renderImageWithDPI(int, float, ImageType) -> BufferedImage
                BufferedImage page = (BufferedImage) rendererClass
                        .getMethod("renderImageWithDPI", int.class, float.class, imageTypeClass)
                        .invoke(renderer, i, 150f, rgbConst);
                pages.add(page);
            }
        } finally {
            pdDocClass.getMethod("close").invoke(doc);
        }
        return pages;
    }

    /**
     * Format raw DocTags output into the requested output format.
     */
    private static String formatOutput(String rawDocTags, String format, DocTagsParser parser) {
        if (rawDocTags == null) {
            return "";
        }
        switch (format) {
            case "markdown": {
                DocumentStructure structure = parser.parse(rawDocTags);
                return parser.toMarkdown(structure);
            }
            case "text": {
                // Strip all XML-like tags — return plain text content only
                DocumentStructure structure = parser.parse(rawDocTags);
                return parser.toMarkdown(structure)
                        .replaceAll("#+\\s*", "")        // remove markdown headers
                        .replaceAll("\\*\\*|\\*|__", "") // remove bold/italic markers
                        .replaceAll("\\|", " ")          // flatten table pipes
                        .replaceAll("-{3,}", "")         // remove horizontal rules
                        .replaceAll("\\n{3,}", "\n\n")   // collapse excess blank lines
                        .trim();
            }
            case "doctags":
            default:
                return rawDocTags;
        }
    }

    /**
     * Build a minimal PreprocessorConfig when no preprocessor_config.json is available.
     * Uses SmolDocling's standard settings (mean/std from SigLIP, target 512x512).
     */
    private static PreprocessorConfig defaultPreprocessorConfig(int tileSize) {
        // Build via the static fromJson factory if available, otherwise use minimal defaults.
        // PreprocessorConfig is a simple data holder — we construct it via ObjectMapper
        // with a minimal JSON blob (same format PreprocessorConfig.fromFile parses).
        try {
            String json = "{\"image_mean\":[0.5,0.5,0.5],\"image_std\":[0.5,0.5,0.5],"
                    + "\"size\":{\"height\":" + tileSize + ",\"width\":" + tileSize + "}}";
            return PreprocessorConfig.fromJson(json);
        } catch (Exception e) {
            // Last resort: return null and let the preprocessor use its own defaults.
            return null;
        }
    }
}
