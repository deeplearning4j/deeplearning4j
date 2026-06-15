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

package org.nd4j.ggml.format;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.GGMLModelImport;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/**
 * Loader for multimodal GGUF models that are distributed across multiple GGUF files.
 *
 * <p>Many multimodal models (such as Qwen3-VL, LLaVA, and BakLLaVA) are released as two
 * separate GGUF files:
 * <ol>
 *   <li>The main language model (LLM) GGUF — contains the transformer decoder weights.</li>
 *   <li>The multimodal projector GGUF ({@code mmproj}) — contains the vision encoder
 *       weights and the projection layer that maps vision features into the LLM's embedding
 *       space.</li>
 * </ol>
 *
 * <h2>Qwen3-VL split GGUF structure</h2>
 * <p>Qwen3-VL follows the standard llama.cpp split convention:
 * <pre>
 * qwen3-vl-7b/
 * ├── qwen3-vl-7b-q4_k_m.gguf          &lt;-- LLM (decoder weights)
 * └── qwen3-vl-7b-mmproj-f16.gguf       &lt;-- mmproj (vision encoder + projector)
 * </pre>
 *
 * <p>The mmproj file contains the vision encoder (a Vision Transformer, e.g.
 * {@code clip.vision.*} tensors in older models, or {@code v.*} tensors in newer ones) plus
 * the multi-modal projector ({@code mm.*}) that bridges visual and language spaces.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Load from explicit files
 * MultimodalGGUFLoader.MultimodalModel model = MultimodalGGUFLoader.loadMultimodal(
 *     new File("qwen3-vl-7b-q4_k_m.gguf"),
 *     new File("qwen3-vl-7b-mmproj-f16.gguf")
 * );
 *
 * // Auto-detect from a directory
 * MultimodalGGUFLoader.MultimodalModel model =
 *     MultimodalGGUFLoader.loadFromDirectory(new File("/path/to/qwen3-vl-7b/"));
 *
 * // Access the components
 * SameDiff llm           = model.getLlm();
 * SameDiff visionEncoder = model.getVisionEncoder();
 * GGMLMetadata llmMeta   = model.getLlmMetadata();
 * GGMLMetadata visMeta   = model.getVisionMetadata();
 * }</pre>
 */
@Slf4j
public class MultimodalGGUFLoader {

    // --- mmproj file name patterns (checked against the lowercase file name) ---------------------

    /** Substrings that identify a vision/mmproj GGUF file by name. */
    private static final String[] MMPROJ_PATTERNS = {
        "mmproj", "mm-proj", "vision", "encoder", "clip"
    };

    // Utility class — no instances.
    private MultimodalGGUFLoader() {}

    // ========== Public API ==========

    /**
     * Load a multimodal model from explicitly specified LLM and mmproj GGUF files.
     *
     * <p>Both files are loaded in parallel using {@link CompletableFuture} to minimise
     * wall-clock load time on multi-core machines.
     *
     * @param llmGguf    the language model GGUF file
     * @param mmprojGguf the multimodal projector / vision encoder GGUF file
     * @return the loaded {@link MultimodalModel} containing both components and their metadata
     * @throws IOException         if either file cannot be read
     * @throws GGMLImportException if GGUF parsing or SameDiff conversion fails
     */
    public static MultimodalModel loadMultimodal(File llmGguf, File mmprojGguf)
            throws IOException, GGMLImportException {

        validateFile(llmGguf, "LLM GGUF");
        validateFile(mmprojGguf, "mmproj GGUF");

        log.info("Loading multimodal model — LLM: {}, mmproj: {}",
                llmGguf.getName(), mmprojGguf.getName());

        long startMs = System.currentTimeMillis();

        // Inspect metadata first (lightweight, not parallelised — establishes log context).
        GGMLMetadata llmMetadata   = GGMLModelImport.inspectModel(llmGguf);
        GGMLMetadata visionMetadata = GGMLModelImport.inspectModel(mmprojGguf);

        log.info("LLM architecture: {}, params: {}M",
                llmMetadata.getArchitecture(),
                llmMetadata.getTotalParameters() / 1_000_000L);
        log.info("Vision encoder architecture: {}, params: {}M",
                visionMetadata.getArchitecture(),
                visionMetadata.getTotalParameters() / 1_000_000L);

        // Load both models in parallel — each is an independent, read-only operation.
        CompletableFuture<SameDiff> llmFuture =
                CompletableFuture.supplyAsync(() -> importQuiet(llmGguf, "LLM"));
        CompletableFuture<SameDiff> visionFuture =
                CompletableFuture.supplyAsync(() -> importQuiet(mmprojGguf, "vision encoder"));

        SameDiff llm;
        SameDiff visionEncoder;
        try {
            CompletableFuture.allOf(llmFuture, visionFuture).join();
            llm           = llmFuture.get();
            visionEncoder = visionFuture.get();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof GGMLImportException) throw (GGMLImportException) cause;
            if (cause instanceof IOException)         throw (IOException) cause;
            throw new IOException("Parallel GGUF loading failed", cause);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("GGUF loading interrupted", e);
        }

        log.info("Multimodal model loaded in {}ms", System.currentTimeMillis() - startMs);

        return MultimodalModel.builder()
                .llm(llm)
                .visionEncoder(visionEncoder)
                .llmMetadata(llmMetadata)
                .visionMetadata(visionMetadata)
                .build();
    }

    /**
     * Auto-detect and load a multimodal model from a directory.
     *
     * <p>Detection logic:
     * <ol>
     *   <li>All {@code *.gguf} files in {@code modelDir} are enumerated.</li>
     *   <li>Files whose (lowercased) names contain any of {@code mmproj}, {@code mm-proj},
     *       {@code vision}, {@code encoder}, or {@code clip} are treated as the vision /
     *       mmproj component.  If multiple candidates are found the first match is used.</li>
     *   <li>Among the remaining files the largest one (by file size) is selected as the LLM.
     *       This heuristic is reliable because the decoder is always significantly larger than
     *       any projector.</li>
     * </ol>
     *
     * @param modelDir directory containing the GGUF files
     * @return the loaded {@link MultimodalModel}
     * @throws IOException         if the directory cannot be read or required files are missing
     * @throws GGMLImportException if GGUF parsing or SameDiff conversion fails
     */
    public static MultimodalModel loadFromDirectory(File modelDir)
            throws IOException, GGMLImportException {

        if (!modelDir.exists() || !modelDir.isDirectory()) {
            throw new IOException("Model directory does not exist: " + modelDir.getAbsolutePath());
        }

        File[] allGgufs = modelDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".gguf"));
        if (allGgufs == null || allGgufs.length == 0) {
            throw new IOException("No .gguf files found in: " + modelDir.getAbsolutePath());
        }

        log.info("Found {} GGUF file(s) in {}", allGgufs.length, modelDir.getAbsolutePath());

        // Separate mmproj candidates from LLM candidates.
        List<File> mmprojCandidates = new ArrayList<>();
        List<File> llmCandidates    = new ArrayList<>();

        for (File f : allGgufs) {
            if (isMmprojFile(f)) {
                mmprojCandidates.add(f);
                log.info("  mmproj candidate: {}", f.getName());
            } else {
                llmCandidates.add(f);
                log.info("  LLM candidate:    {}", f.getName());
            }
        }

        if (mmprojCandidates.isEmpty()) {
            throw new IOException(
                "No mmproj/vision GGUF file found in: " + modelDir.getAbsolutePath() +
                ". Expected a file whose name contains one of: " + Arrays.toString(MMPROJ_PATTERNS));
        }
        if (llmCandidates.isEmpty()) {
            throw new IOException(
                "No LLM GGUF file found in: " + modelDir.getAbsolutePath() +
                " (every .gguf file was classified as mmproj).");
        }

        // For mmproj: take the first (alphabetically consistent) candidate.
        File mmprojGguf = mmprojCandidates.get(0);

        // For LLM: take the largest file — the decoder always dominates.
        File llmGguf = llmCandidates.stream()
                .max(Comparator.comparingLong(File::length))
                .orElseThrow(() -> new IOException("No LLM candidate after filtering"));

        log.info("Selected LLM:    {} ({} MB)",
                llmGguf.getName(), llmGguf.length() / (1024 * 1024));
        log.info("Selected mmproj: {} ({} MB)",
                mmprojGguf.getName(), mmprojGguf.length() / (1024 * 1024));

        return loadMultimodal(llmGguf, mmprojGguf);
    }

    // ========== Helpers ==========

    /**
     * Return {@code true} if the file's lowercase name matches any mmproj pattern.
     */
    private static boolean isMmprojFile(File file) {
        String lower = file.getName().toLowerCase();
        for (String pattern : MMPROJ_PATTERNS) {
            if (lower.contains(pattern)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Validate that a file exists, is readable, and has non-zero size.
     */
    private static void validateFile(File file, String label) throws IOException {
        if (file == null) {
            throw new IOException(label + " file must not be null");
        }
        if (!file.exists()) {
            throw new IOException(label + " file not found: " + file.getAbsolutePath());
        }
        if (!file.isFile()) {
            throw new IOException(label + " path is not a file: " + file.getAbsolutePath());
        }
        if (!file.canRead()) {
            throw new IOException(label + " file is not readable: " + file.getAbsolutePath());
        }
        if (file.length() == 0) {
            throw new IOException(label + " file is empty: " + file.getAbsolutePath());
        }
    }

    /**
     * Import a GGUF file, wrapping checked exceptions into unchecked ones so the result can
     * be supplied from a {@link CompletableFuture}.
     */
    private static SameDiff importQuiet(File file, String label) {
        try {
            log.info("Loading {} from: {}", label, file.getName());
            long t0 = System.currentTimeMillis();
            SameDiff sd = GGMLModelImport.importModel(file);
            log.info("Loaded {} in {}ms", label, System.currentTimeMillis() - t0);
            return sd;
        } catch (GGMLImportException e) {
            throw new RuntimeException("Failed to import " + label + " from " + file.getName(), e);
        }
    }

    // ========== Result container ==========

    /**
     * Container for a loaded multimodal GGUF model.
     *
     * <p>Holds the language model ({@link #llm}), the vision encoder
     * ({@link #visionEncoder}), and the metadata parsed from each GGUF header.
     *
     * <h3>Qwen3-VL component roles</h3>
     * <ul>
     *   <li>{@code llm} — the autoregressive transformer decoder.  Feed token IDs and
     *       (projected) visual embeddings here.</li>
     *   <li>{@code visionEncoder} — the Vision Transformer plus the multimodal projector
     *       ({@code mm.*} tensors).  Feed raw pixel tensors; receive projected visual
     *       embeddings that match the LLM's hidden dimension.</li>
     * </ul>
     */
    @Data
    @Builder
    public static class MultimodalModel {

        /**
         * The language model (decoder) loaded from the main LLM GGUF file.
         */
        private final SameDiff llm;

        /**
         * The vision encoder (Vision Transformer + multimodal projector) loaded from the
         * mmproj GGUF file.
         */
        private final SameDiff visionEncoder;

        /**
         * GGUF metadata extracted from the LLM file (architecture, layer counts, tokenizer
         * information, etc.).
         */
        private final GGMLMetadata llmMetadata;

        /**
         * GGUF metadata extracted from the vision/mmproj file.
         */
        private final GGMLMetadata visionMetadata;

        /**
         * Return {@code true} if both the LLM and the vision encoder were loaded successfully.
         */
        public boolean isComplete() {
            return llm != null && visionEncoder != null;
        }

        /**
         * Return the architecture string reported by the LLM GGUF header, or {@code "unknown"}
         * if metadata is unavailable.
         */
        public String getLlmArchitecture() {
            return llmMetadata != null ? llmMetadata.getArchitecture() : "unknown";
        }

        /**
         * Return the approximate total parameter count across both models (LLM + vision).
         */
        public long getTotalParameters() {
            long total = 0;
            if (llmMetadata     != null) total += llmMetadata.getTotalParameters();
            if (visionMetadata  != null) total += visionMetadata.getTotalParameters();
            return total;
        }

        /**
         * Convert to a {@code VideoVisionLanguageModel} that wires the LLM and the vision
         * encoder together for end-to-end image/video understanding.
         *
         * <p><b>Note:</b> This method is a stub.  The wiring implementation depends on the
         * {@code samediff-vlm} module and the specific model's forward-pass graph, neither of
         * which is available in the {@code nd4j-ggml} module.  Callers that need full
         * end-to-end inference should depend on the higher-level VLM module and perform the
         * wiring there.
         *
         * @throws UnsupportedOperationException always — the implementation lives in a
         *                                        higher-level module
         */
        public Object toVideoVisionLanguageModel() {
            throw new UnsupportedOperationException(
                "toVideoVisionLanguageModel() is not implemented in nd4j-ggml. " +
                "Depend on the samediff-vlm module and wire the llm and visionEncoder " +
                "components there.");
        }
    }
}
