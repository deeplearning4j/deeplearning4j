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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.llm.config.PreprocessorConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Boundary ablation for the remaining page-10 benchmark failure.
 *
 * Old/new decoder parity is already covered elsewhere. This test isolates whether the
 * maxTiles=5 -> maxTiles=6 collapse boundary survives when the visual-token embedding
 * rows are zeroed while keeping the prompt/grid metadata unchanged.
 */
@Slf4j
public class TestPage10PromptVsVisionBoundary {

    private static final int TARGET_SIZE = 512;
    private static SameDiff visionEncoderSd;
    private static SameDiff decoder;
    private static SameDiff embedTokens;
    private static Tokenizer tokenizer;
    private static boolean loaded;

    @BeforeAll
    public static void setup() {
        System.setProperty("nd4j.optimizer.enabled", "true");
        System.setProperty("nd4j.optimizer.fp16", "true");
    }

    @Test
    @DisplayName("Page-10 maxTiles=5 vs 6: zeroing visual-token rows reveals whether prompt/layout alone preserves the collapse boundary")
    public void testPage10BoundaryWithAllVisualRowsZeroed() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles5 = buildVariant("PAGE10_MAXTILES5_PROMPTVISION", 10, 2048, 5);
        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_PROMPTVISION", 10, 2048, 6);
        VariantInputs zero5 = zeroVisualTokenRows("PAGE10_MAXTILES5_ZERO_VISUAL", maxTiles5);
        VariantInputs zero6 = zeroVisualTokenRows("PAGE10_MAXTILES6_ZERO_VISUAL", maxTiles6);
        try {
            DecodeSnapshot full5 = runOldDecoderDirect(maxTiles5, 40);
            DecodeSnapshot full6 = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot zeroed5 = runOldDecoderDirect(zero5, 40);
            DecodeSnapshot zeroed6 = runOldDecoderDirect(zero6, 40);

            logDecode(maxTiles5, full5);
            logDecode(maxTiles6, full6);
            logDecode(zero5, zeroed5);
            logDecode(zero6, zeroed6);

            assertFalse(isPictureOnlyCollapse(full5.result.getText()),
                    "Baseline maxTiles=5 should remain textual. text='"
                            + safeSnippet(full5.result.getText(), 220) + "'");
            assertTrue(isPictureOnlyCollapse(full6.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse boundary. text='"
                            + safeSnippet(full6.result.getText(), 220) + "'");

            boolean zero5Collapsed = isPictureOnlyCollapse(zeroed5.result.getText());
            boolean zero6Collapsed = isPictureOnlyCollapse(zeroed6.result.getText());
            log.info("Zero-visual boundary result: maxTiles5 collapsed={} text='{}' | maxTiles6 collapsed={} text='{}'",
                    zero5Collapsed, safeSnippet(zeroed5.result.getText(), 180),
                    zero6Collapsed, safeSnippet(zeroed6.result.getText(), 180));

            boolean anyVisualDependence =
                    !Arrays.equals(full5.result.getTokenIds(), zeroed5.result.getTokenIds())
                            || !Arrays.equals(full6.result.getTokenIds(), zeroed6.result.getTokenIds());
            assertTrue(anyVisualDependence,
                    "Zeroing all visual-token rows did not change either side of the page-10 boundary. "
                            + "That would imply the visual embeddings are not influencing generation at all.");

            assertFalse(zero5Collapsed && !zero6Collapsed,
                    "Zeroing visual-token rows inverted the page-10 boundary (maxTiles=5 collapsed while maxTiles=6 did not). "
                            + "zero5='" + safeSnippet(zeroed5.result.getText(), 180) + "' zero6='"
                            + safeSnippet(zeroed6.result.getText(), 180) + "'");
        } finally {
            maxTiles5.close();
            maxTiles6.close();
            zero5.close();
            zero6.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: zeroing one frame block at a time changes the collapse")
    public void testPage10MaxTiles6SingleFrameAblationMatrix() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_FRAME_MATRIX", 10, 2048, 6);
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles6, 40);
            logDecode(maxTiles6, baseline);
            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");

            boolean sawDifferentTokens = false;
            boolean sawNonCollapsed = false;
            List<VariantInputs> ablations = new ArrayList<>();
            try {
                for (int frameIdx = 0; frameIdx < maxTiles6.totalFrames; frameIdx++) {
                    VariantInputs ablated = zeroSingleFrameRows(
                            "PAGE10_MAXTILES6_ZERO_FRAME_" + frameIdx, maxTiles6, frameIdx);
                    ablations.add(ablated);
                    DecodeSnapshot ablatedDecode = runOldDecoderDirect(ablated, 40);
                    logDecode(ablated, ablatedDecode);
                    if (!Arrays.equals(baseline.result.getTokenIds(), ablatedDecode.result.getTokenIds())) {
                        sawDifferentTokens = true;
                    }
                    if (!isPictureOnlyCollapse(ablatedDecode.result.getText())) {
                        sawNonCollapsed = true;
                    }
                }
            } finally {
                for (VariantInputs ablated : ablations) {
                    ablated.close();
                }
            }

            assertTrue(sawDifferentTokens,
                    "Zeroing individual maxTiles=6 frame blocks never changed the collapsed token stream.");
            assertTrue(sawNonCollapsed,
                    "Zeroing each maxTiles=6 frame block one at a time never escaped picture-only collapse.");
        } finally {
            maxTiles6.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: scaling all visual rows by 6/7 tests aggregate-magnitude saturation")
    public void testPage10MaxTiles6UniformVisualScaling() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_SCALE_BASE", 10, 2048, 6);
        VariantInputs scaled = scaleVisualTokenRows("PAGE10_MAXTILES6_SCALE_6_7", maxTiles6, 6.0 / 7.0);
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot scaledDecode = runOldDecoderDirect(scaled, 40);

            logDecode(maxTiles6, baseline);
            logDecode(scaled, scaledDecode);

            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");
            assertArrayEquals(baseline.result.getTokenIds(), scaledDecode.result.getTokenIds(),
                    "Uniform 6/7 scaling of all maxTiles=6 visual rows should leave the collapse unchanged. "
                            + "If this starts differing, the bug is no longer tied to active frame occupancy.");
            assertEquals(baseline.result.getText(), scaledDecode.result.getText(),
                    "Uniform 6/7 scaling of all maxTiles=6 visual rows should preserve the collapsed text.");

            log.info("Uniform-scaling result: collapsed={} text='{}'",
                    isPictureOnlyCollapse(scaledDecode.result.getText()),
                    safeSnippet(scaledDecode.result.getText(), 180));
        } finally {
            maxTiles6.close();
            scaled.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: duplicating frame content without reducing active slots distinguishes occupancy from content")
    public void testPage10MaxTiles6DuplicateFrameOccupancyVsContent() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_DUP_BASE", 10, 2048, 6);
        VariantInputs zeroFrame1 = zeroSingleFrameRows("PAGE10_MAXTILES6_ZERO_FRAME_1_DUP_CHECK", maxTiles6, 1);
        VariantInputs dupFrame0Into1 = copyFrameRows("PAGE10_MAXTILES6_COPY_FRAME0_TO_1", maxTiles6, 0, 1);
        VariantInputs zeroGlobal = zeroSingleFrameRows("PAGE10_MAXTILES6_ZERO_GLOBAL_DUP_CHECK", maxTiles6, 6);
        VariantInputs dupFrame5IntoGlobal = copyFrameRows("PAGE10_MAXTILES6_COPY_FRAME5_TO_GLOBAL", maxTiles6, 5, 6);
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot zero1Decode = runOldDecoderDirect(zeroFrame1, 40);
            DecodeSnapshot dup01Decode = runOldDecoderDirect(dupFrame0Into1, 40);
            DecodeSnapshot zeroGlobalDecode = runOldDecoderDirect(zeroGlobal, 40);
            DecodeSnapshot dup5GlobalDecode = runOldDecoderDirect(dupFrame5IntoGlobal, 40);

            logDecode(maxTiles6, baseline);
            logDecode(zeroFrame1, zero1Decode);
            logDecode(dupFrame0Into1, dup01Decode);
            logDecode(zeroGlobal, zeroGlobalDecode);
            logDecode(dupFrame5IntoGlobal, dup5GlobalDecode);

            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");
            assertFalse(isPictureOnlyCollapse(zero1Decode.result.getText()),
                    "Zeroing tile frame 1 should escape the collapse. text='"
                            + safeSnippet(zero1Decode.result.getText(), 220) + "'");
            assertFalse(isPictureOnlyCollapse(zeroGlobalDecode.result.getText()),
                    "Zeroing the global frame should escape the collapse. text='"
                            + safeSnippet(zeroGlobalDecode.result.getText(), 220) + "'");

            boolean duplicateTileStillCollapsed = isPictureOnlyCollapse(dup01Decode.result.getText());
            boolean duplicateGlobalStillCollapsed = isPictureOnlyCollapse(dup5GlobalDecode.result.getText());
            log.info("Duplicate-frame occupancy check: duplicateTileStillCollapsed={} duplicateGlobalStillCollapsed={} "
                            + "dupTileText='{}' dupGlobalText='{}'",
                    duplicateTileStillCollapsed,
                    duplicateGlobalStillCollapsed,
                    safeSnippet(dup01Decode.result.getText(), 180),
                    safeSnippet(dup5GlobalDecode.result.getText(), 180));

            boolean anyDuplicateChanged =
                    !Arrays.equals(baseline.result.getTokenIds(), dup01Decode.result.getTokenIds())
                            || !Arrays.equals(baseline.result.getTokenIds(), dup5GlobalDecode.result.getTokenIds());
            assertTrue(anyDuplicateChanged,
                    "Replacing frame content while keeping all maxTiles=6 slots active did not change the collapsed token stream at all.");
        } finally {
            maxTiles6.close();
            zeroFrame1.close();
            dupFrame0Into1.close();
            zeroGlobal.close();
            dupFrame5IntoGlobal.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: swapping frame-block order preserves occupancy and content multiset")
    public void testPage10MaxTiles6FrameOrderSensitivity() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_SWAP_BASE", 10, 2048, 6);
        VariantInputs swapTiles01 = swapFrameRows("PAGE10_MAXTILES6_SWAP_TILE0_TILE1", maxTiles6, 0, 1);
        VariantInputs swapTile0Global = swapFrameRows("PAGE10_MAXTILES6_SWAP_TILE0_GLOBAL", maxTiles6, 0, 6);
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot swap01Decode = runOldDecoderDirect(swapTiles01, 40);
            DecodeSnapshot swap0GlobalDecode = runOldDecoderDirect(swapTile0Global, 40);

            logDecode(maxTiles6, baseline);
            logDecode(swapTiles01, swap01Decode);
            logDecode(swapTile0Global, swap0GlobalDecode);

            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");

            boolean swap01Collapsed = isPictureOnlyCollapse(swap01Decode.result.getText());
            boolean swap0GlobalCollapsed = isPictureOnlyCollapse(swap0GlobalDecode.result.getText());
            log.info("Frame-order sensitivity: swap01Collapsed={} swap0GlobalCollapsed={} swap01Text='{}' swap0GlobalText='{}'",
                    swap01Collapsed, swap0GlobalCollapsed,
                    safeSnippet(swap01Decode.result.getText(), 180),
                    safeSnippet(swap0GlobalDecode.result.getText(), 180));

            boolean anySwapChanged =
                    !Arrays.equals(baseline.result.getTokenIds(), swap01Decode.result.getTokenIds())
                            || !Arrays.equals(baseline.result.getTokenIds(), swap0GlobalDecode.result.getTokenIds());
            assertTrue(anySwapChanged,
                    "Swapping maxTiles=6 frame-block order never changed the collapsed token stream.");
        } finally {
            maxTiles6.close();
            swapTiles01.close();
            swapTile0Global.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=5 vs 6: per-frame embedding statistics stay finite across the collapse boundary")
    public void testPage10FrameStatisticsAcrossBoundary() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles5 = buildVariant("PAGE10_MAXTILES5_FRAME_STATS", 10, 2048, 5);
        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_FRAME_STATS", 10, 2048, 6);
        try {
            FrameStatsSummary stats5 = summarizeFrameStats(maxTiles5);
            FrameStatsSummary stats6 = summarizeFrameStats(maxTiles6);

            log.info("Frame stats maxTiles=5: {}", stats5.summary);
            log.info("Frame stats maxTiles=6: {}", stats6.summary);

            assertTrue(stats5.allFinite,
                    "maxTiles=5 frame statistics contain NaN/Inf values: " + stats5.summary);
            assertTrue(stats6.allFinite,
                    "maxTiles=6 frame statistics contain NaN/Inf values: " + stats6.summary);
            assertEquals(5, stats5.frameCount,
                    "Expected 5 total frames (4 tiles + 1 global) at maxTiles=5.");
            assertEquals(7, stats6.frameCount,
                    "Expected 7 total frames (6 tiles + 1 global) at maxTiles=6.");
        } finally {
            maxTiles5.close();
            maxTiles6.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: candidate frame reorders isolate whether global-first or column-major order restores text")
    public void testPage10MaxTiles6CandidateReorders() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_REORDER_BASE", 10, 2048, 6);
        VariantInputs globalFirst = reorderFrameRows(
                "PAGE10_MAXTILES6_REORDER_GLOBAL_FIRST", maxTiles6, new int[]{6, 0, 1, 2, 3, 4, 5});
        VariantInputs columnMajor = reorderFrameRows(
                "PAGE10_MAXTILES6_REORDER_COLUMN_MAJOR", maxTiles6, new int[]{0, 2, 4, 1, 3, 5, 6});
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot globalFirstDecode = runOldDecoderDirect(globalFirst, 40);
            DecodeSnapshot columnMajorDecode = runOldDecoderDirect(columnMajor, 40);

            logDecode(maxTiles6, baseline);
            logDecode(globalFirst, globalFirstDecode);
            logDecode(columnMajor, columnMajorDecode);

            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");

            boolean globalFirstCollapsed = isPictureOnlyCollapse(globalFirstDecode.result.getText());
            boolean columnMajorCollapsed = isPictureOnlyCollapse(columnMajorDecode.result.getText());
            log.info("Candidate reorder results: globalFirstCollapsed={} columnMajorCollapsed={} "
                            + "globalFirstText='{}' columnMajorText='{}'",
                    globalFirstCollapsed,
                    columnMajorCollapsed,
                    safeSnippet(globalFirstDecode.result.getText(), 180),
                    safeSnippet(columnMajorDecode.result.getText(), 180));

            assertTrue(!globalFirstCollapsed || !columnMajorCollapsed,
                    "Neither candidate reorder escaped the maxTiles=6 collapse, so the order signal is weaker than the swap probes suggested.");
        } finally {
            maxTiles6.close();
            globalFirst.close();
            columnMajor.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9 benchmark path: a simple tile-order swap probes whether the order sensitivity survives full tiling")
    public void testPage10MaxTiles9SimpleOrderProbe() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles9 = buildVariant("PAGE10_MAXTILES9_ORDER_BASE", 10, 2048, 9);
        VariantInputs swap01 = swapFrameRows("PAGE10_MAXTILES9_SWAP_TILE0_TILE1", maxTiles9, 0, 1);
        try {
            DecodeSnapshot baseline = runOldDecoderDirect(maxTiles9, 40);
            DecodeSnapshot swapped = runOldDecoderDirect(swap01, 40);

            logDecode(maxTiles9, baseline);
            logDecode(swap01, swapped);

            assertTrue(isPictureOnlyCollapse(baseline.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baseline.result.getText(), 220) + "'");

            boolean swappedCollapsed = isPictureOnlyCollapse(swapped.result.getText());
            log.info("maxTiles=9 simple order probe: swappedCollapsed={} swappedText='{}'",
                    swappedCollapsed, safeSnippet(swapped.result.getText(), 180));

            assertFalse(Arrays.equals(baseline.result.getTokenIds(), swapped.result.getTokenIds()),
                    "Swapping tile0/tile1 at maxTiles=9 should change the token stream if order sensitivity survives the full benchmark tiling.");
        } finally {
            maxTiles9.close();
            swap01.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=6: factor prompt-order vs embedding-order using a global-first candidate")
    public void testPage10MaxTiles6GlobalFirstPromptEmbeddingFactorization() throws Exception {
        ensureModelsLoaded();

        int[] globalFirstOrder = {6, 0, 1, 2, 3, 4, 5};
        VariantInputs baseline = buildVariant("PAGE10_MAXTILES6_FACT_BASE", 10, 2048, 6);
        VariantInputs embedOnly = buildVariant(
                "PAGE10_MAXTILES6_FACT_EMBED_ONLY_GLOBAL_FIRST", 10, 2048, 6, null, globalFirstOrder);
        VariantInputs promptOnly = buildVariant(
                "PAGE10_MAXTILES6_FACT_PROMPT_ONLY_GLOBAL_FIRST", 10, 2048, 6, globalFirstOrder, null);
        VariantInputs both = buildVariant(
                "PAGE10_MAXTILES6_FACT_PROMPT_AND_EMBED_GLOBAL_FIRST", 10, 2048, 6, globalFirstOrder, globalFirstOrder);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot embedOnlyDecode = runOldDecoderDirect(embedOnly, 40);
            DecodeSnapshot promptOnlyDecode = runOldDecoderDirect(promptOnly, 40);
            DecodeSnapshot bothDecode = runOldDecoderDirect(both, 40);

            logDecode(baseline, baselineDecode);
            logDecode(embedOnly, embedOnlyDecode);
            logDecode(promptOnly, promptOnlyDecode);
            logDecode(both, bothDecode);

            boolean baselineCollapsed = isPictureOnlyCollapse(baselineDecode.result.getText());
            boolean embedOnlyCollapsed = isPictureOnlyCollapse(embedOnlyDecode.result.getText());
            boolean promptOnlyCollapsed = isPictureOnlyCollapse(promptOnlyDecode.result.getText());
            boolean bothCollapsed = isPictureOnlyCollapse(bothDecode.result.getText());

            log.info("maxTiles=6 factorization: baselineCollapsed={} embedOnlyCollapsed={} promptOnlyCollapsed={} bothCollapsed={}"
                            + " | embedOnly='{}' promptOnly='{}' both='{}'",
                    baselineCollapsed,
                    embedOnlyCollapsed,
                    promptOnlyCollapsed,
                    bothCollapsed,
                    safeSnippet(embedOnlyDecode.result.getText(), 180),
                    safeSnippet(promptOnlyDecode.result.getText(), 180),
                    safeSnippet(bothDecode.result.getText(), 180));

            assertTrue(baselineCollapsed,
                    "Baseline maxTiles=6 should reproduce the current collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");
            assertTrue(!embedOnlyCollapsed || !promptOnlyCollapsed || !bothCollapsed,
                    "Neither prompt-only, embedding-only, nor consistent global-first reordering changed the maxTiles=6 collapse.");
        } finally {
            baseline.close();
            embedOnly.close();
            promptOnly.close();
            both.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: global-first factorization changes collapse details but does not recover text")
    public void testPage10MaxTiles9GlobalFirstPromptEmbeddingFactorization() throws Exception {
        ensureModelsLoaded();

        int[] globalFirstOrder = {9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_FACT_BASE", 10, 2048, 9);
        VariantInputs embedOnly = buildVariant(
                "PAGE10_MAXTILES9_FACT_EMBED_ONLY_GLOBAL_FIRST", 10, 2048, 9, null, globalFirstOrder);
        VariantInputs promptOnly = buildVariant(
                "PAGE10_MAXTILES9_FACT_PROMPT_ONLY_GLOBAL_FIRST", 10, 2048, 9, globalFirstOrder, null);
        VariantInputs both = buildVariant(
                "PAGE10_MAXTILES9_FACT_PROMPT_AND_EMBED_GLOBAL_FIRST", 10, 2048, 9, globalFirstOrder, globalFirstOrder);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot embedOnlyDecode = runOldDecoderDirect(embedOnly, 40);
            DecodeSnapshot promptOnlyDecode = runOldDecoderDirect(promptOnly, 40);
            DecodeSnapshot bothDecode = runOldDecoderDirect(both, 40);

            logDecode(baseline, baselineDecode);
            logDecode(embedOnly, embedOnlyDecode);
            logDecode(promptOnly, promptOnlyDecode);
            logDecode(both, bothDecode);

            boolean baselineCollapsed = isPictureOnlyCollapse(baselineDecode.result.getText());
            boolean embedOnlyCollapsed = isPictureOnlyCollapse(embedOnlyDecode.result.getText());
            boolean promptOnlyCollapsed = isPictureOnlyCollapse(promptOnlyDecode.result.getText());
            boolean bothCollapsed = isPictureOnlyCollapse(bothDecode.result.getText());

            log.info("maxTiles=9 factorization: baselineCollapsed={} embedOnlyCollapsed={} promptOnlyCollapsed={} bothCollapsed={}"
                            + " | embedOnly='{}' promptOnly='{}' both='{}'",
                    baselineCollapsed,
                    embedOnlyCollapsed,
                    promptOnlyCollapsed,
                    bothCollapsed,
                    safeSnippet(embedOnlyDecode.result.getText(), 180),
                    safeSnippet(promptOnlyDecode.result.getText(), 180),
                    safeSnippet(bothDecode.result.getText(), 180));

            assertTrue(baselineCollapsed,
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");
            assertTrue(embedOnlyCollapsed,
                    "Embedding-only global-first unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(embedOnlyDecode.result.getText(), 220) + "'");
            assertTrue(promptOnlyCollapsed,
                    "Prompt-only global-first unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(promptOnlyDecode.result.getText(), 220) + "'");
            assertTrue(bothCollapsed,
                    "Consistent global-first reorder unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(bothDecode.result.getText(), 220) + "'");

            assertArrayEquals(baselineDecode.result.getTokenIds(), embedOnlyDecode.result.getTokenIds(),
                    "At maxTiles=9, embedding-only global-first should preserve the exact baseline picture-collapse prefix.");
            assertNotEquals(baselineDecode.result.getText(), bothDecode.result.getText(),
                    "Consistent global-first should change the collapsed picture-mode text even though it stays collapsed.");
            assertTrue(bothDecode.result.getText().contains("<picture>"),
                    "Consistent global-first should remain in picture mode at maxTiles=9. text='"
                            + safeSnippet(bothDecode.result.getText(), 220) + "'");
        } finally {
            baseline.close();
            embedOnly.close();
            promptOnly.close();
            both.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: grouped frame ablations isolate whether specific regions lock the model into picture mode")
    public void testPage10MaxTiles9GroupedFrameAblations() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_GROUP_BASE", 10, 2048, 9);
        VariantInputs zeroGlobal = zeroFrameRows("PAGE10_MAXTILES9_ZERO_GLOBAL", baseline, 9);
        VariantInputs zeroRightColumn = zeroFrameRows("PAGE10_MAXTILES9_ZERO_RIGHT_COLUMN", baseline, 2, 5, 8);
        VariantInputs zeroBottomRow = zeroFrameRows("PAGE10_MAXTILES9_ZERO_BOTTOM_ROW", baseline, 6, 7, 8);
        VariantInputs zeroMiddleColumn = zeroFrameRows("PAGE10_MAXTILES9_ZERO_MIDDLE_COLUMN", baseline, 1, 4, 7);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot zeroGlobalDecode = runOldDecoderDirect(zeroGlobal, 40);
            DecodeSnapshot zeroRightColumnDecode = runOldDecoderDirect(zeroRightColumn, 40);
            DecodeSnapshot zeroBottomRowDecode = runOldDecoderDirect(zeroBottomRow, 40);
            DecodeSnapshot zeroMiddleColumnDecode = runOldDecoderDirect(zeroMiddleColumn, 40);

            logDecode(baseline, baselineDecode);
            logDecode(zeroGlobal, zeroGlobalDecode);
            logDecode(zeroRightColumn, zeroRightColumnDecode);
            logDecode(zeroBottomRow, zeroBottomRowDecode);
            logDecode(zeroMiddleColumn, zeroMiddleColumnDecode);

            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            DecodeSnapshot[] ablations = {
                    zeroGlobalDecode, zeroRightColumnDecode, zeroBottomRowDecode, zeroMiddleColumnDecode
            };
            boolean anyTokenChange = false;
            boolean anyNonCollapsed = false;
            for (DecodeSnapshot ablation : ablations) {
                if (!Arrays.equals(baselineDecode.result.getTokenIds(), ablation.result.getTokenIds())) {
                    anyTokenChange = true;
                }
                if (!isPictureOnlyCollapse(ablation.result.getText())) {
                    anyNonCollapsed = true;
                }
            }

            log.info("maxTiles=9 grouped ablations: anyTokenChange={} anyNonCollapsed={} "
                            + "zeroGlobal='{}' zeroRight='{}' zeroBottom='{}' zeroMiddle='{}'",
                    anyTokenChange,
                    anyNonCollapsed,
                    safeSnippet(zeroGlobalDecode.result.getText(), 160),
                    safeSnippet(zeroRightColumnDecode.result.getText(), 160),
                    safeSnippet(zeroBottomRowDecode.result.getText(), 160),
                    safeSnippet(zeroMiddleColumnDecode.result.getText(), 160));

            assertTrue(anyTokenChange,
                    "Grouped maxTiles=9 ablations never changed the picture-collapse token stream.");
        } finally {
            baseline.close();
            zeroGlobal.close();
            zeroRightColumn.close();
            zeroBottomRow.close();
            zeroMiddleColumn.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: no single bottom-row tile explains the distributed footer-row effect")
    public void testPage10MaxTiles9BottomRowSingleTileAblations() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_BOTTOM_BASE", 10, 2048, 9);
        VariantInputs zeroBottomLeft = zeroSingleFrameRows("PAGE10_MAXTILES9_ZERO_BOTTOM_LEFT", baseline, 6);
        VariantInputs zeroBottomMid = zeroSingleFrameRows("PAGE10_MAXTILES9_ZERO_BOTTOM_MID", baseline, 7);
        VariantInputs zeroBottomRight = zeroSingleFrameRows("PAGE10_MAXTILES9_ZERO_BOTTOM_RIGHT", baseline, 8);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot leftDecode = runOldDecoderDirect(zeroBottomLeft, 40);
            DecodeSnapshot midDecode = runOldDecoderDirect(zeroBottomMid, 40);
            DecodeSnapshot rightDecode = runOldDecoderDirect(zeroBottomRight, 40);

            logDecode(baseline, baselineDecode);
            logDecode(zeroBottomLeft, leftDecode);
            logDecode(zeroBottomMid, midDecode);
            logDecode(zeroBottomRight, rightDecode);

            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            boolean leftCollapsed = isPictureOnlyCollapse(leftDecode.result.getText());
            boolean midCollapsed = isPictureOnlyCollapse(midDecode.result.getText());
            boolean rightCollapsed = isPictureOnlyCollapse(rightDecode.result.getText());

            log.info("maxTiles=9 bottom-row single-tile ablations: leftCollapsed={} midCollapsed={} rightCollapsed={} "
                            + "left='{}' mid='{}' right='{}'",
                    leftCollapsed,
                    midCollapsed,
                    rightCollapsed,
                    safeSnippet(leftDecode.result.getText(), 160),
                    safeSnippet(midDecode.result.getText(), 160),
                    safeSnippet(rightDecode.result.getText(), 160));

            assertTrue(leftCollapsed,
                    "Zeroing only the bottom-left tile unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(leftDecode.result.getText(), 220) + "'");
            assertTrue(midCollapsed,
                    "Zeroing only the bottom-middle tile unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(midDecode.result.getText(), 220) + "'");
            assertTrue(rightCollapsed,
                    "Zeroing only the bottom-right tile unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(rightDecode.result.getText(), 220) + "'");

            assertArrayEquals(baselineDecode.result.getTokenIds(), leftDecode.result.getTokenIds(),
                    "Zeroing only the bottom-left tile should preserve the baseline collapse, showing the footer-row effect is not localized.");
            assertArrayEquals(baselineDecode.result.getTokenIds(), midDecode.result.getTokenIds(),
                    "Zeroing only the bottom-middle tile should preserve the baseline collapse, showing the footer-row effect is not localized.");
            assertArrayEquals(baselineDecode.result.getTokenIds(), rightDecode.result.getTokenIds(),
                    "Zeroing only the bottom-right tile should preserve the baseline collapse, showing the footer-row effect is not localized.");
        } finally {
            baseline.close();
            zeroBottomLeft.close();
            zeroBottomMid.close();
            zeroBottomRight.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: moving the bottom-row block earlier tests sequence-position weighting")
    public void testPage10MaxTiles9BottomRowReorderSensitivity() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_REORDER_BASE", 10, 2048, 9);
        VariantInputs bottomRowFirst = buildVariant(
                "PAGE10_MAXTILES9_BOTTOM_ROW_FIRST", 10, 2048, 9, null, new int[]{6, 7, 8, 0, 1, 2, 3, 4, 5, 9});
        VariantInputs bottomRowMiddle = buildVariant(
                "PAGE10_MAXTILES9_BOTTOM_ROW_MIDDLE", 10, 2048, 9, null, new int[]{0, 1, 2, 6, 7, 8, 3, 4, 5, 9});
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot firstDecode = runOldDecoderDirect(bottomRowFirst, 40);
            DecodeSnapshot middleDecode = runOldDecoderDirect(bottomRowMiddle, 40);

            logDecode(baseline, baselineDecode);
            logDecode(bottomRowFirst, firstDecode);
            logDecode(bottomRowMiddle, middleDecode);

            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            boolean firstCollapsed = isPictureOnlyCollapse(firstDecode.result.getText());
            boolean middleCollapsed = isPictureOnlyCollapse(middleDecode.result.getText());
            log.info("maxTiles=9 bottom-row reorder sensitivity: firstCollapsed={} middleCollapsed={} "
                            + "first='{}' middle='{}'",
                    firstCollapsed,
                    middleCollapsed,
                    safeSnippet(firstDecode.result.getText(), 160),
                    safeSnippet(middleDecode.result.getText(), 160));

            assertTrue(firstCollapsed,
                    "Moving the bottom row earlier unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(firstDecode.result.getText(), 220) + "'");
            assertTrue(middleCollapsed,
                    "Moving the bottom row to the middle unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(middleDecode.result.getText(), 220) + "'");

            assertArrayEquals(baselineDecode.result.getTokenIds(), firstDecode.result.getTokenIds(),
                    "Moving the entire bottom-row block earlier should preserve the baseline collapse if the failure is not just sequence position.");
            assertArrayEquals(baselineDecode.result.getTokenIds(), middleDecode.result.getTokenIds(),
                    "Moving the entire bottom-row block to the middle should preserve the baseline collapse if the failure is not just sequence position.");
        } finally {
            baseline.close();
            bottomRowFirst.close();
            bottomRowMiddle.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: scaling only the bottom-row bundle tests whether its aggregate strength drives picture collapse")
    public void testPage10MaxTiles9BottomRowScalingSensitivity() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_BOTTOM_ROW_SCALE_BASE", 10, 2048, 9);
        VariantInputs halfBottomRow = scaleFrameRows("PAGE10_MAXTILES9_SCALE_BOTTOM_ROW_HALF", baseline, 0.5, 6, 7, 8);
        VariantInputs quarterBottomRow = scaleFrameRows("PAGE10_MAXTILES9_SCALE_BOTTOM_ROW_QUARTER", baseline, 0.25, 6, 7, 8);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot halfDecode = runOldDecoderDirect(halfBottomRow, 40);
            DecodeSnapshot quarterDecode = runOldDecoderDirect(quarterBottomRow, 40);

            logDecode(baseline, baselineDecode);
            logDecode(halfBottomRow, halfDecode);
            logDecode(quarterBottomRow, quarterDecode);

            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            boolean halfCollapsed = isPictureOnlyCollapse(halfDecode.result.getText());
            boolean quarterCollapsed = isPictureOnlyCollapse(quarterDecode.result.getText());
            log.info("maxTiles=9 bottom-row scaling: halfCollapsed={} quarterCollapsed={} half='{}' quarter='{}'",
                    halfCollapsed,
                    quarterCollapsed,
                    safeSnippet(halfDecode.result.getText(), 180),
                    safeSnippet(quarterDecode.result.getText(), 180));

            assertTrue(halfCollapsed,
                    "Half-scaling the bottom-row bundle unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(halfDecode.result.getText(), 220) + "'");
            assertTrue(quarterCollapsed,
                    "Quarter-scaling the bottom-row bundle unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(quarterDecode.result.getText(), 220) + "'");

            assertArrayEquals(baselineDecode.result.getTokenIds(), halfDecode.result.getTokenIds(),
                    "Halving the entire bottom-row bundle should preserve the baseline collapse if moderate magnitude is not the trigger.");
            assertArrayEquals(baselineDecode.result.getTokenIds(), quarterDecode.result.getTokenIds(),
                    "Quarter-scaling the entire bottom-row bundle should preserve the baseline collapse if moderate magnitude is not the trigger.");
        } finally {
            baseline.close();
            halfBottomRow.close();
            quarterBottomRow.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: replacing the bottom row with duplicated non-bottom content separates bundle semantics from occupancy")
    public void testPage10MaxTiles9BottomRowReplacementSensitivity() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_BOTTOM_ROW_REPLACE_BASE", 10, 2048, 9);
        VariantInputs middleIntoBottom = copyFrameGroupRows(
                "PAGE10_MAXTILES9_COPY_MIDDLE_ROW_TO_BOTTOM", baseline,
                new int[]{3, 4, 5}, new int[]{6, 7, 8});
        VariantInputs topIntoBottom = copyFrameGroupRows(
                "PAGE10_MAXTILES9_COPY_TOP_ROW_TO_BOTTOM", baseline,
                new int[]{0, 1, 2}, new int[]{6, 7, 8});
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot middleDecode = runOldDecoderDirect(middleIntoBottom, 40);
            DecodeSnapshot topDecode = runOldDecoderDirect(topIntoBottom, 40);

            logDecode(baseline, baselineDecode);
            logDecode(middleIntoBottom, middleDecode);
            logDecode(topIntoBottom, topDecode);

            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            boolean middleCollapsed = isPictureOnlyCollapse(middleDecode.result.getText());
            boolean topCollapsed = isPictureOnlyCollapse(topDecode.result.getText());
            log.info("maxTiles=9 bottom-row replacement: middleCollapsed={} topCollapsed={} middle='{}' top='{}'",
                    middleCollapsed,
                    topCollapsed,
                    safeSnippet(middleDecode.result.getText(), 180),
                    safeSnippet(topDecode.result.getText(), 180));

            assertTrue(middleCollapsed,
                    "Replacing the bottom row with middle-row copies unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(middleDecode.result.getText(), 220) + "'");
            assertTrue(topCollapsed,
                    "Replacing the bottom row with top-row copies unexpectedly escaped the maxTiles=9 collapse. text='"
                            + safeSnippet(topDecode.result.getText(), 220) + "'");

            assertArrayEquals(baselineDecode.result.getTokenIds(), middleDecode.result.getTokenIds(),
                    "Replacing the full bottom-row bundle with middle-row copies should preserve the baseline collapse if semantics do not matter at those slots.");
            assertArrayEquals(baselineDecode.result.getTokenIds(), topDecode.result.getTokenIds(),
                    "Replacing the full bottom-row bundle with top-row copies should preserve the baseline collapse if semantics do not matter at those slots.");
        } finally {
            baseline.close();
            middleIntoBottom.close();
            topIntoBottom.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: near-zero bottom-row embeddings escape like zero, bounding the collapse threshold")
    public void testPage10MaxTiles9BottomRowNearZeroOccupancyThreshold() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_BOTTOM_ROW_EPS_BASE", 10, 2048, 9);
        VariantInputs tinyBottomRow = fillFrameRows("PAGE10_MAXTILES9_FILL_BOTTOM_ROW_TINY", baseline, 1.0e-6, 6, 7, 8);
        VariantInputs zeroBottomRow = fillFrameRows("PAGE10_MAXTILES9_FILL_BOTTOM_ROW_ZERO", baseline, 0.0, 6, 7, 8);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot tinyDecode = runOldDecoderDirect(tinyBottomRow, 40);
            DecodeSnapshot zeroDecode = runOldDecoderDirect(zeroBottomRow, 40);

            logDecode(baseline, baselineDecode);
            logDecode(tinyBottomRow, tinyDecode);
            logDecode(zeroBottomRow, zeroDecode);

            boolean baselineCollapsed = isPictureOnlyCollapse(baselineDecode.result.getText());
            boolean tinyCollapsed = isPictureOnlyCollapse(tinyDecode.result.getText());
            boolean zeroCollapsed = isPictureOnlyCollapse(zeroDecode.result.getText());
            log.info("maxTiles=9 bottom-row near-zero threshold: baselineCollapsed={} tinyCollapsed={} zeroCollapsed={} "
                            + "tiny='{}' zero='{}'",
                    baselineCollapsed,
                    tinyCollapsed,
                    zeroCollapsed,
                    safeSnippet(tinyDecode.result.getText(), 180),
                    safeSnippet(zeroDecode.result.getText(), 180));

            assertTrue(baselineCollapsed,
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");
            assertFalse(zeroCollapsed,
                    "Zero-filling the bottom row should escape the maxTiles=9 collapse. text='"
                            + safeSnippet(zeroDecode.result.getText(), 220) + "'");
            assertFalse(tinyCollapsed,
                    "Near-zero bottom-row embeddings should escape the maxTiles=9 collapse at this threshold. text='"
                            + safeSnippet(tinyDecode.result.getText(), 220) + "'");
            assertArrayEquals(zeroDecode.result.getTokenIds(), tinyDecode.result.getTokenIds(),
                    "Near-zero bottom-row embeddings should match the zero-filled recovery if the threshold is already below representable influence.");
        } finally {
            baseline.close();
            tinyBottomRow.close();
            zeroBottomRow.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: constant bottom-row fill sweep distinguishes occupancy magnitude from semantic content")
    public void testPage10MaxTiles9BottomRowConstantFillSweep() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_BOTTOM_ROW_FILL_SWEEP_BASE", 10, 2048, 9);
        double[] fillValues = {1.0, 0.5, 0.25, 0.1, 0.01, 1.0e-3, 1.0e-4, 1.0e-6, 0.0};
        List<VariantInputs> fillVariants = new ArrayList<>();
        List<DecodeSnapshot> fillDecodes = new ArrayList<>();
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            logDecode(baseline, baselineDecode);
            assertTrue(isPictureOnlyCollapse(baselineDecode.result.getText()),
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");

            int[] zeroTokenIds = null;
            boolean sawNonZeroRecovery = false;
            boolean sawNonZeroExactCollapse = false;
            StringBuilder sweepSummary = new StringBuilder();

            for (double fillValue : fillValues) {
                String label = Double.toString(fillValue)
                        .replace("-", "m")
                        .replace(".", "p");
                VariantInputs variant = fillFrameRows(
                        "PAGE10_MAXTILES9_FILL_BOTTOM_ROW_" + label, baseline, fillValue, 6, 7, 8);
                fillVariants.add(variant);

                DecodeSnapshot decode = runOldDecoderDirect(variant, 40);
                fillDecodes.add(decode);
                logDecode(variant, decode);

                boolean collapsed = isPictureOnlyCollapse(decode.result.getText());
                boolean exactBaseline = Arrays.equals(baselineDecode.result.getTokenIds(), decode.result.getTokenIds());
                if (fillValue == 0.0) {
                    zeroTokenIds = decode.result.getTokenIds();
                    assertFalse(collapsed,
                            "Zero-filled bottom-row slots should escape the maxTiles=9 collapse. text='"
                                    + safeSnippet(decode.result.getText(), 220) + "'");
                }
                if (fillValue != 0.0 && exactBaseline) {
                    sawNonZeroExactCollapse = true;
                }

                if (sweepSummary.length() > 0) {
                    sweepSummary.append(" | ");
                }
                sweepSummary.append("fill=").append(fillValue)
                        .append("{collapsed=").append(collapsed)
                        .append(",exactBaseline=").append(exactBaseline)
                        .append(",text='").append(safeSnippet(decode.result.getText(), 90)).append("'}");
            }

            assertNotNull(zeroTokenIds, "Zero-fill reference was not captured in the sweep.");
            for (int i = 0; i < fillValues.length; i++) {
                double fillValue = fillValues[i];
                if (fillValue == 0.0) {
                    continue;
                }
                if (Arrays.equals(zeroTokenIds, fillDecodes.get(i).result.getTokenIds())) {
                    sawNonZeroRecovery = true;
                    break;
                }
            }

            log.info("maxTiles=9 bottom-row constant fill sweep: {}",
                    sweepSummary);

            assertTrue(sawNonZeroRecovery,
                    "No nonzero constant bottom-row fill matched the zero-filled recovery. "
                            + "That would imply the observed near-zero recovery was not stable.");

            if (!sawNonZeroExactCollapse) {
                log.info("maxTiles=9 constant fill sweep: no nonzero constant fill reproduced the exact baseline collapse. "
                                + "This points away from pure slot occupancy magnitude and toward semantic content in the original bottom-row vectors.");
            }
        } finally {
            baseline.close();
            for (VariantInputs variant : fillVariants) {
                variant.close();
            }
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: aspect-preserving padded 3x3 tiles isolate square-distortion from decoder behavior")
    public void testPage10MaxTiles9AspectPreservingPaddedGrid() throws Exception {
        ensureModelsLoaded();

        VariantInputs baseline = buildVariant("PAGE10_MAXTILES9_ASPECT_BASE", 10, 2048, 9);
        VariantInputs padded = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_3X3", 10, 2048, 3, 3);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot paddedDecode = runOldDecoderDirect(padded, 40);

            logDecode(baseline, baselineDecode);
            logDecode(padded, paddedDecode);

            boolean baselineCollapsed = isPictureOnlyCollapse(baselineDecode.result.getText());
            boolean paddedCollapsed = isPictureOnlyCollapse(paddedDecode.result.getText());
            log.info("maxTiles=9 aspect-preserving padded grid: baselineCollapsed={} paddedCollapsed={} "
                            + "padded='{}'",
                    baselineCollapsed,
                    paddedCollapsed,
                    safeSnippet(paddedDecode.result.getText(), 180));

            assertTrue(baselineCollapsed,
                    "Baseline maxTiles=9 should reproduce the benchmark-style collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");
            assertFalse(Arrays.equals(baselineDecode.result.getTokenIds(), paddedDecode.result.getTokenIds()),
                    "Aspect-preserving padded 3x3 tiles did not change the page-10 maxTiles=9 token stream at all. "
                            + "If this fails, square distortion is probably not the active visual bug.");
        } finally {
            baseline.close();
            padded.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: global-first embedding reorder still matters after aspect-preserving padded tiling")
    public void testPage10MaxTiles9AspectPreservingPaddedGridGlobalFirstEmbeddings() throws Exception {
        ensureModelsLoaded();

        VariantInputs padded = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_3X3_EMBED_BASE", 10, 2048, 3, 3);
        VariantInputs globalFirst = reorderFrameRows(
                "PAGE10_MAXTILES9_ASPECT_PADDED_3X3_EMBED_GLOBAL_FIRST",
                padded,
                new int[]{9, 0, 1, 2, 3, 4, 5, 6, 7, 8});
        try {
            DecodeSnapshot paddedDecode = runOldDecoderDirect(padded, 40);
            DecodeSnapshot globalFirstDecode = runOldDecoderDirect(globalFirst, 40);

            logDecode(padded, paddedDecode);
            logDecode(globalFirst, globalFirstDecode);

            boolean paddedCollapsed = isPictureOnlyCollapse(paddedDecode.result.getText());
            boolean globalFirstCollapsed = isPictureOnlyCollapse(globalFirstDecode.result.getText());
            log.info("maxTiles=9 aspect-preserving + global-first embeddings: paddedCollapsed={} globalFirstCollapsed={} "
                            + "globalFirst='{}'",
                    paddedCollapsed,
                    globalFirstCollapsed,
                    safeSnippet(globalFirstDecode.result.getText(), 180));

            assertFalse(paddedCollapsed,
                    "Aspect-preserving padded 3x3 control should already escape picture collapse. text='"
                            + safeSnippet(paddedDecode.result.getText(), 220) + "'");
            assertFalse(Arrays.equals(paddedDecode.result.getTokenIds(), globalFirstDecode.result.getTokenIds()),
                    "Global-first embedding reorder did not change the aspect-preserving padded 3x3 token stream.");
        } finally {
            padded.close();
            globalFirst.close();
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=9: factor prompt-order vs embedding-order after aspect-preserving padded tiling")
    public void testPage10MaxTiles9AspectPreservingPaddedGlobalFirstFactorization() throws Exception {
        ensureModelsLoaded();

        int[] globalFirstOrder = {9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        VariantInputs baseline = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_FACT_BASE", 10, 2048, 3, 3, null, null);
        VariantInputs embedOnly = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_FACT_EMBED_ONLY", 10, 2048, 3, 3, null, globalFirstOrder);
        VariantInputs promptOnly = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_FACT_PROMPT_ONLY", 10, 2048, 3, 3, globalFirstOrder, null);
        VariantInputs both = buildAspectPreservingPaddedGridVariant(
                "PAGE10_MAXTILES9_ASPECT_PADDED_FACT_BOTH", 10, 2048, 3, 3, globalFirstOrder, globalFirstOrder);
        try {
            DecodeSnapshot baselineDecode = runOldDecoderDirect(baseline, 40);
            DecodeSnapshot embedOnlyDecode = runOldDecoderDirect(embedOnly, 40);
            DecodeSnapshot promptOnlyDecode = runOldDecoderDirect(promptOnly, 40);
            DecodeSnapshot bothDecode = runOldDecoderDirect(both, 40);

            logDecode(baseline, baselineDecode);
            logDecode(embedOnly, embedOnlyDecode);
            logDecode(promptOnly, promptOnlyDecode);
            logDecode(both, bothDecode);

            boolean baselineCollapsed = isPictureOnlyCollapse(baselineDecode.result.getText());
            boolean embedOnlyCollapsed = isPictureOnlyCollapse(embedOnlyDecode.result.getText());
            boolean promptOnlyCollapsed = isPictureOnlyCollapse(promptOnlyDecode.result.getText());
            boolean bothCollapsed = isPictureOnlyCollapse(bothDecode.result.getText());

            log.info("maxTiles=9 aspect-preserving factorization: baselineCollapsed={} embedOnlyCollapsed={} "
                            + "promptOnlyCollapsed={} bothCollapsed={} | embedOnly='{}' promptOnly='{}' both='{}'",
                    baselineCollapsed,
                    embedOnlyCollapsed,
                    promptOnlyCollapsed,
                    bothCollapsed,
                    safeSnippet(embedOnlyDecode.result.getText(), 180),
                    safeSnippet(promptOnlyDecode.result.getText(), 180),
                    safeSnippet(bothDecode.result.getText(), 180));

            assertFalse(baselineCollapsed,
                    "Aspect-preserving padded 3x3 baseline should already escape picture collapse. text='"
                            + safeSnippet(baselineDecode.result.getText(), 220) + "'");
            boolean anyOrderChanged =
                    !Arrays.equals(baselineDecode.result.getTokenIds(), embedOnlyDecode.result.getTokenIds())
                            || !Arrays.equals(baselineDecode.result.getTokenIds(), promptOnlyDecode.result.getTokenIds())
                            || !Arrays.equals(baselineDecode.result.getTokenIds(), bothDecode.result.getTokenIds());
            assertTrue(anyOrderChanged,
                    "Prompt/embedding global-first factorization did not change the aspect-preserving padded token stream.");
        } finally {
            baseline.close();
            embedOnly.close();
            promptOnly.close();
            both.close();
        }
    }

    private static synchronized void ensureModelsLoaded() throws Exception {
        if (loaded) {
            return;
        }

        VLMModelDownloader.DownloadResult visionDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER);
        VLMModelDownloader.DownloadResult decoderDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_DECODER);
        VLMModelDownloader.DownloadResult embedTokensDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_EMBED_TOKENS);
        VLMModelDownloader.DownloadResult tokenizerDl =
                VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER);
        VLMModelDownloader.download(VLMModelDownloader.VLMModel.SMOLDOCLING_TOKENIZER_CONFIG);

        tokenizer = HuggingFaceTokenizer.fromFile(tokenizerDl.getModelFile());
        SameDiff[] models = OnnxModelCache.importAllWithCache(
                visionDl.getModelFile().getAbsolutePath(),
                decoderDl.getModelFile().getAbsolutePath(),
                embedTokensDl.getModelFile().getAbsolutePath()
        );
        visionEncoderSd = models[0];
        decoder = models[1];
        embedTokens = models[2];
        loaded = true;
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, int maxTiles) throws Exception {
        return buildVariant(name, pdfPage, resizeLongestEdge, maxTiles, null, null);
    }

    private VariantInputs buildAspectPreservingPaddedGridVariant(
            String name, int pdfPage, int resizeLongestEdge, int numRows, int numCols) throws Exception {
        return buildAspectPreservingPaddedGridVariant(
                name, pdfPage, resizeLongestEdge, numRows, numCols, null, null);
    }

    private VariantInputs buildAspectPreservingPaddedGridVariant(
            String name, int pdfPage, int resizeLongestEdge, int numRows, int numCols,
            int[] promptSegmentOrder, int[] embeddingFrameOrder) throws Exception {
        BenchmarkConfigApplier.resetModelState(visionEncoderSd);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        BufferedImage rendered = loadPageImage(pdfPage);
        if (resizeLongestEdge > 0) {
            rendered = ImageTiler.resizeLongestEdge(rendered, resizeLongestEdge);
        }

        ImageTiler.SplitImageResult splitResult =
                splitImageAspectPreservingWithPadding(rendered, TARGET_SIZE, numRows, numCols);
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        INDArray visionEmbeddings = encodeManually(imageInput, splitResult);
        imageInput.close();
        INDArray reorderedVisionEmbeddings = null;

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = promptSegmentOrder == null
                ? ImagePromptBuilder.buildImagePromptString(splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame)
                : buildPromptForSegmentOrder(splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame, promptSegmentOrder);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray inputsEmbeds;
        INDArray textEmbeddings = null;
        try {
            textEmbeddings = embedPromptDirect(promptTokenIds);
            INDArray visionForMerge = visionEmbeddings;
            if (embeddingFrameOrder != null) {
                reorderedVisionEmbeddings = reorderVisionEmbeddingFrames(
                        name, visionEmbeddings, splitResult.getTotalFrames(), embeddingFrameOrder);
                visionForMerge = reorderedVisionEmbeddings;
            }
            inputsEmbeds = EmbeddingMerger.mergeEmbeddings(
                    textEmbeddings, visionForMerge, promptTokenIds, imageTokenId);
        } finally {
            closeArray(textEmbeddings);
            closeArray(reorderedVisionEmbeddings);
            visionEmbeddings.close();
        }

        log.info("{}: aspect-preserving padded grid={}x{} frames={} mergedShape={} promptTokens={}",
                name,
                splitResult.numRows,
                splitResult.numCols,
                splitResult.getTotalFrames(),
                Arrays.toString(inputsEmbeds.shape()),
                promptTokenIds.length);

        return new VariantInputs(name, promptTokenIds, inputsEmbeds, splitResult.getTotalFrames());
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, int maxTiles,
                                       int[] promptSegmentOrder, int[] embeddingFrameOrder) throws Exception {
        BenchmarkConfigApplier.resetModelState(visionEncoderSd);
        BenchmarkConfigApplier.resetModelState(embedTokens);

        BufferedImage rendered = loadPageImage(pdfPage);
        if (resizeLongestEdge > 0) {
            rendered = ImageTiler.resizeLongestEdge(rendered, resizeLongestEdge);
        }

        ImageTiler.SplitImageResult splitResult = ImageTiler.splitImageForVLM(rendered, TARGET_SIZE, maxTiles);
        PreprocessorConfig ppConfig = new PreprocessorConfig();
        ppConfig.setSize(new PreprocessorConfig.ImageSize(TARGET_SIZE, TARGET_SIZE));
        ppConfig.setDoRescale(true);
        ppConfig.setRescaleFactor(1.0 / 255.0);
        ppConfig.setDoNormalize(true);
        ppConfig.setImageMean(new double[]{0.5, 0.5, 0.5});
        ppConfig.setImageStd(new double[]{0.5, 0.5, 0.5});
        VLMImagePreprocessor preprocessor = VLMImagePreprocessor.fromConfig(ppConfig);
        INDArray imageInput = VisionEncoderUtils.preprocessFrames(splitResult.frames, preprocessor, TARGET_SIZE);
        preprocessor.shutdown();

        INDArray visionEmbeddings = encodeManually(imageInput, splitResult);
        imageInput.close();
        INDArray reorderedVisionEmbeddings = null;

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = promptSegmentOrder == null
                ? ImagePromptBuilder.buildImagePromptString(splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame)
                : buildPromptForSegmentOrder(splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame, promptSegmentOrder);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray inputsEmbeds;
        INDArray textEmbeddings = null;
        try {
            textEmbeddings = embedPromptDirect(promptTokenIds);
            INDArray visionForMerge = visionEmbeddings;
            if (embeddingFrameOrder != null) {
                reorderedVisionEmbeddings = reorderVisionEmbeddingFrames(
                        name, visionEmbeddings, splitResult.getTotalFrames(), embeddingFrameOrder);
                visionForMerge = reorderedVisionEmbeddings;
            }
            inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionForMerge, promptTokenIds, imageTokenId);
        } finally {
            closeArray(textEmbeddings);
            closeArray(reorderedVisionEmbeddings);
            visionEmbeddings.close();
        }

        log.info("{}: maxTiles={} frames={} grid={}x{} mergedShape={} promptTokens={}",
                name, maxTiles, splitResult.getTotalFrames(), splitResult.numRows, splitResult.numCols,
                Arrays.toString(inputsEmbeds.shape()), promptTokenIds.length);

        return new VariantInputs(name, promptTokenIds, inputsEmbeds, splitResult.getTotalFrames());
    }

    private ImageTiler.SplitImageResult splitImageAspectPreservingWithPadding(
            BufferedImage image, int targetSize, int numRows, int numCols) {
        List<BufferedImage> frames = new ArrayList<>();
        List<ImageTiler.ContentRegion> contentRegions = new ArrayList<>();

        int width = image.getWidth();
        int height = image.getHeight();
        int optimalWidth = (int) Math.ceil((double) width / numCols);
        int optimalHeight = (int) Math.ceil((double) height / numRows);

        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                int startX = c * optimalWidth;
                int startY = r * optimalHeight;
                int endX = Math.min(startX + optimalWidth, width);
                int endY = Math.min(startY + optimalHeight, height);

                BufferedImage tile = image.getSubimage(startX, startY, endX - startX, endY - startY);
                ImageTiler.ResizeResult resized = ImageTiler.resizeToFit(tile, targetSize, targetSize);
                BufferedImage padded = ImageTiler.padToSize(resized.image, targetSize, targetSize);
                frames.add(padded);
                contentRegions.add(new ImageTiler.ContentRegion(resized.width, resized.height));
            }
        }

        ImageTiler.ResizeResult globalResized = ImageTiler.resizeToFit(image, targetSize, targetSize);
        BufferedImage globalPadded = ImageTiler.padToSize(globalResized.image, targetSize, targetSize);
        frames.add(globalPadded);
        contentRegions.add(new ImageTiler.ContentRegion(globalResized.width, globalResized.height));

        log.info("Aspect-preserving padded split: source={}x{} grid={}x{} tileContent={}x{} totalFrames={}",
                width, height, numRows, numCols, optimalWidth, optimalHeight, frames.size());
        return new ImageTiler.SplitImageResult(frames, contentRegions, numRows, numCols);
    }

    private INDArray reorderVisionEmbeddingFrames(String name, INDArray visionEmbeddings, int totalFrames,
                                                  int[] sourceOrderForTargets) {
        assertEquals(totalFrames, sourceOrderForTargets.length,
                "Permutation length must match totalFrames for " + name);

        boolean[] seen = new boolean[totalFrames];
        for (int srcFrame : sourceOrderForTargets) {
            assertTrue(srcFrame >= 0 && srcFrame < totalFrames,
                    "Source frame out of range for " + name + ": " + srcFrame);
            assertFalse(seen[srcFrame], "Permutation repeats frame " + srcFrame + " for " + name);
            seen[srcFrame] = true;
        }

        long seqLen = visionEmbeddings.size(1);
        assertEquals(0, seqLen % totalFrames,
                "Vision embeddings sequence length should divide evenly across frames for " + name);
        int rowsPerFrame = (int) (seqLen / totalFrames);

        INDArray reordered = visionEmbeddings.dup();
        for (int targetFrame = 0; targetFrame < totalFrames; targetFrame++) {
            int sourceFrame = sourceOrderForTargets[targetFrame];
            for (int offset = 0; offset < rowsPerFrame; offset++) {
                int srcRowIdx = sourceFrame * rowsPerFrame + offset;
                int dstRowIdx = targetFrame * rowsPerFrame + offset;
                INDArray srcRow = visionEmbeddings.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(srcRowIdx), NDArrayIndex.all()).dup();
                try {
                    reordered.get(NDArrayIndex.point(0), NDArrayIndex.point(dstRowIdx), NDArrayIndex.all()).assign(srcRow);
                } finally {
                    closeArray(srcRow);
                }
            }
        }

        log.info("{}: reordered vision embeddings with sourceOrderForTargets={}",
                name, Arrays.toString(sourceOrderForTargets));
        return reordered;
    }

    private String buildPromptForSegmentOrder(int numRows, int numCols, int imageSeqLen, int[] sourceOrderForTargets) {
        int tileCount = Math.max(0, numRows * numCols);
        int totalSegments = tileCount + 1;
        assertEquals(totalSegments, sourceOrderForTargets.length,
                "Permutation length must match image segments (tiles + global)");

        boolean[] seen = new boolean[totalSegments];
        for (int srcIdx : sourceOrderForTargets) {
            assertTrue(srcIdx >= 0 && srcIdx < totalSegments,
                    "Source segment out of range: " + srcIdx + " totalSegments=" + totalSegments);
            assertFalse(seen[srcIdx], "Permutation repeats segment " + srcIdx);
            seen[srcIdx] = true;
        }

        String[] descriptors = new String[totalSegments];
        int idx = 0;
        for (int r = 1; r <= numRows; r++) {
            for (int c = 1; c <= numCols; c++) {
                descriptors[idx++] = "<row_" + r + "_col_" + c + ">";
            }
        }
        descriptors[idx] = "<global-img>";

        String fake = "<fake_token_around_image>";
        String image = "<image>";
        StringBuilder sb = new StringBuilder();
        for (int targetIdx = 0; targetIdx < totalSegments; targetIdx++) {
            if (targetIdx > 0 && numCols > 0 && targetIdx % numCols == 0) {
                sb.append("\n");
            }
            sb.append(fake).append(descriptors[sourceOrderForTargets[targetIdx]]);
            for (int i = 0; i < imageSeqLen; i++) {
                sb.append(image);
            }
        }
        sb.append(fake);
        return sb.toString();
    }

    private VariantInputs zeroVisualTokenRows(String name, VariantInputs base) {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        INDArray zeroed = base.inputsEmbeds.dup();
        int zeroedRows = 0;
        for (int i = 0; i < base.promptTokenIds.length; i++) {
            if (base.promptTokenIds[i] == imageTokenId) {
                zeroed.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all()).assign(0.0);
                zeroedRows++;
            }
        }
        assertTrue(zeroedRows > 0, "Expected to zero at least one image-token row for " + base.name);
        log.info("{}: zeroed {} visual-token rows", name, zeroedRows);
        return new VariantInputs(name, base.promptTokenIds, zeroed, base.totalFrames);
    }

    private VariantInputs zeroSingleFrameRows(String name, VariantInputs base, int frameIndex) {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        List<Integer> imagePositions = new ArrayList<>();
        for (int i = 0; i < base.promptTokenIds.length; i++) {
            if (base.promptTokenIds[i] == imageTokenId) {
                imagePositions.add(i);
            }
        }
        assertFalse(imagePositions.isEmpty(), "Expected image token positions for " + base.name);
        assertEquals(0, imagePositions.size() % base.totalFrames,
                "Image token rows should divide evenly across frames for " + base.name);
        assertTrue(frameIndex >= 0 && frameIndex < base.totalFrames,
                "frameIndex out of range: " + frameIndex + " totalFrames=" + base.totalFrames);

        int rowsPerFrame = imagePositions.size() / base.totalFrames;
        int start = frameIndex * rowsPerFrame;
        int end = start + rowsPerFrame;
        INDArray zeroed = base.inputsEmbeds.dup();
        for (int i = start; i < end; i++) {
            int promptIdx = imagePositions.get(i);
            zeroed.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdx), NDArrayIndex.all()).assign(0.0);
        }
        log.info("{}: zeroed frame {} using {} image-token rows", name, frameIndex, rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, zeroed, base.totalFrames);
    }

    private VariantInputs zeroFrameRows(String name, VariantInputs base, int... frameIndices) {
        FrameLayout layout = frameLayout(base);
        INDArray zeroed = base.inputsEmbeds.dup();
        boolean[] seen = new boolean[base.totalFrames];
        for (int frameIndex : frameIndices) {
            assertTrue(frameIndex >= 0 && frameIndex < base.totalFrames,
                    "frameIndex out of range: " + frameIndex + " totalFrames=" + base.totalFrames);
            assertFalse(seen[frameIndex], "Duplicate frame index " + frameIndex + " for " + name);
            seen[frameIndex] = true;
            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int promptIdx = layout.imagePositions.get(frameIndex * layout.rowsPerFrame + offset);
                zeroed.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdx), NDArrayIndex.all()).assign(0.0);
            }
        }
        log.info("{}: zeroed frames {} using {} image-token rows per frame",
                name, Arrays.toString(frameIndices), layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, zeroed, base.totalFrames);
    }

    private VariantInputs scaleVisualTokenRows(String name, VariantInputs base, double scale) {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        INDArray scaled = base.inputsEmbeds.dup();
        int scaledRows = 0;
        for (int i = 0; i < base.promptTokenIds.length; i++) {
            if (base.promptTokenIds[i] == imageTokenId) {
                INDArray row = scaled.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all());
                row.muli(scale);
                scaledRows++;
            }
        }
        assertTrue(scaledRows > 0, "Expected to scale at least one image-token row for " + base.name);
        log.info("{}: scaled {} visual-token rows by {}", name, scaledRows, scale);
        return new VariantInputs(name, base.promptTokenIds, scaled, base.totalFrames);
    }

    private VariantInputs scaleFrameRows(String name, VariantInputs base, double scale, int... frameIndices) {
        FrameLayout layout = frameLayout(base);
        INDArray scaled = base.inputsEmbeds.dup();
        boolean[] seen = new boolean[base.totalFrames];
        for (int frameIndex : frameIndices) {
            assertTrue(frameIndex >= 0 && frameIndex < base.totalFrames,
                    "frameIndex out of range: " + frameIndex + " totalFrames=" + base.totalFrames);
            assertFalse(seen[frameIndex], "Duplicate frame index " + frameIndex + " for " + name);
            seen[frameIndex] = true;
            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int promptIdx = layout.imagePositions.get(frameIndex * layout.rowsPerFrame + offset);
                INDArray row = scaled.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdx), NDArrayIndex.all());
                row.muli(scale);
            }
        }

        log.info("{}: scaled frames {} by {} using {} image-token rows per frame",
                name, Arrays.toString(frameIndices), scale, layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, scaled, base.totalFrames);
    }

    private VariantInputs fillFrameRows(String name, VariantInputs base, double value, int... frameIndices) {
        FrameLayout layout = frameLayout(base);
        INDArray filled = base.inputsEmbeds.dup();
        boolean[] seen = new boolean[base.totalFrames];
        for (int frameIndex : frameIndices) {
            assertTrue(frameIndex >= 0 && frameIndex < base.totalFrames,
                    "frameIndex out of range: " + frameIndex + " totalFrames=" + base.totalFrames);
            assertFalse(seen[frameIndex], "Duplicate frame index " + frameIndex + " for " + name);
            seen[frameIndex] = true;
            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int promptIdx = layout.imagePositions.get(frameIndex * layout.rowsPerFrame + offset);
                filled.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdx), NDArrayIndex.all()).assign(value);
            }
        }

        log.info("{}: filled frames {} with {} using {} image-token rows per frame",
                name, Arrays.toString(frameIndices), value, layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, filled, base.totalFrames);
    }

    private VariantInputs copyFrameRows(String name, VariantInputs base, int sourceFrameIndex, int targetFrameIndex) {
        FrameLayout layout = frameLayout(base);
        assertTrue(sourceFrameIndex >= 0 && sourceFrameIndex < base.totalFrames,
                "sourceFrameIndex out of range: " + sourceFrameIndex + " totalFrames=" + base.totalFrames);
        assertTrue(targetFrameIndex >= 0 && targetFrameIndex < base.totalFrames,
                "targetFrameIndex out of range: " + targetFrameIndex + " totalFrames=" + base.totalFrames);
        assertNotEquals(sourceFrameIndex, targetFrameIndex,
                "Source and target frame must differ for " + name);

        INDArray copied = base.inputsEmbeds.dup();
        for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
            int srcPromptIdx = layout.imagePositions.get(sourceFrameIndex * layout.rowsPerFrame + offset);
            int dstPromptIdx = layout.imagePositions.get(targetFrameIndex * layout.rowsPerFrame + offset);
            INDArray srcRow = copied.get(NDArrayIndex.point(0), NDArrayIndex.point(srcPromptIdx), NDArrayIndex.all()).dup();
            try {
                copied.get(NDArrayIndex.point(0), NDArrayIndex.point(dstPromptIdx), NDArrayIndex.all()).assign(srcRow);
            } finally {
                closeArray(srcRow);
            }
        }

        log.info("{}: copied frame {} into frame {} using {} image-token rows",
                name, sourceFrameIndex, targetFrameIndex, layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, copied, base.totalFrames);
    }

    private VariantInputs copyFrameGroupRows(String name, VariantInputs base, int[] sourceFrames, int[] targetFrames) {
        assertEquals(sourceFrames.length, targetFrames.length,
                "Source and target frame groups must have equal length for " + name);
        FrameLayout layout = frameLayout(base);
        INDArray copied = base.inputsEmbeds.dup();
        boolean[] targetSeen = new boolean[base.totalFrames];
        for (int i = 0; i < sourceFrames.length; i++) {
            int sourceFrameIndex = sourceFrames[i];
            int targetFrameIndex = targetFrames[i];
            assertTrue(sourceFrameIndex >= 0 && sourceFrameIndex < base.totalFrames,
                    "sourceFrameIndex out of range: " + sourceFrameIndex + " totalFrames=" + base.totalFrames);
            assertTrue(targetFrameIndex >= 0 && targetFrameIndex < base.totalFrames,
                    "targetFrameIndex out of range: " + targetFrameIndex + " totalFrames=" + base.totalFrames);
            assertFalse(targetSeen[targetFrameIndex],
                    "Duplicate target frame " + targetFrameIndex + " for " + name);
            targetSeen[targetFrameIndex] = true;
            assertNotEquals(sourceFrameIndex, targetFrameIndex,
                    "Source and target frame must differ for " + name + " at group index " + i);

            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int srcPromptIdx = layout.imagePositions.get(sourceFrameIndex * layout.rowsPerFrame + offset);
                int dstPromptIdx = layout.imagePositions.get(targetFrameIndex * layout.rowsPerFrame + offset);
                INDArray srcRow = base.inputsEmbeds.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(srcPromptIdx), NDArrayIndex.all()).dup();
                try {
                    copied.get(NDArrayIndex.point(0), NDArrayIndex.point(dstPromptIdx), NDArrayIndex.all()).assign(srcRow);
                } finally {
                    closeArray(srcRow);
                }
            }
        }

        log.info("{}: copied frame group {} into {} using {} image-token rows per frame",
                name, Arrays.toString(sourceFrames), Arrays.toString(targetFrames), layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, copied, base.totalFrames);
    }

    private VariantInputs swapFrameRows(String name, VariantInputs base, int frameA, int frameB) {
        FrameLayout layout = frameLayout(base);
        assertTrue(frameA >= 0 && frameA < base.totalFrames,
                "frameA out of range: " + frameA + " totalFrames=" + base.totalFrames);
        assertTrue(frameB >= 0 && frameB < base.totalFrames,
                "frameB out of range: " + frameB + " totalFrames=" + base.totalFrames);
        assertNotEquals(frameA, frameB, "Cannot swap the same frame for " + name);

        INDArray swapped = base.inputsEmbeds.dup();
        for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
            int promptIdxA = layout.imagePositions.get(frameA * layout.rowsPerFrame + offset);
            int promptIdxB = layout.imagePositions.get(frameB * layout.rowsPerFrame + offset);
            INDArray rowA = swapped.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdxA), NDArrayIndex.all()).dup();
            INDArray rowB = swapped.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdxB), NDArrayIndex.all()).dup();
            try {
                swapped.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdxA), NDArrayIndex.all()).assign(rowB);
                swapped.get(NDArrayIndex.point(0), NDArrayIndex.point(promptIdxB), NDArrayIndex.all()).assign(rowA);
            } finally {
                closeArray(rowA);
                closeArray(rowB);
            }
        }

        log.info("{}: swapped frame {} with frame {} using {} image-token rows each",
                name, frameA, frameB, layout.rowsPerFrame);
        return new VariantInputs(name, base.promptTokenIds, swapped, base.totalFrames);
    }

    private VariantInputs reorderFrameRows(String name, VariantInputs base, int[] sourceOrderForTargets) {
        FrameLayout layout = frameLayout(base);
        assertEquals(base.totalFrames, sourceOrderForTargets.length,
                "Permutation length must match totalFrames for " + name);

        boolean[] seen = new boolean[base.totalFrames];
        for (int srcFrame : sourceOrderForTargets) {
            assertTrue(srcFrame >= 0 && srcFrame < base.totalFrames,
                    "Source frame out of range for " + name + ": " + srcFrame);
            assertFalse(seen[srcFrame], "Permutation repeats frame " + srcFrame + " for " + name);
            seen[srcFrame] = true;
        }

        INDArray reordered = base.inputsEmbeds.dup();
        for (int targetFrame = 0; targetFrame < base.totalFrames; targetFrame++) {
            int sourceFrame = sourceOrderForTargets[targetFrame];
            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int srcPromptIdx = layout.imagePositions.get(sourceFrame * layout.rowsPerFrame + offset);
                int dstPromptIdx = layout.imagePositions.get(targetFrame * layout.rowsPerFrame + offset);
                INDArray srcRow = base.inputsEmbeds.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(srcPromptIdx), NDArrayIndex.all()).dup();
                try {
                    reordered.get(NDArrayIndex.point(0), NDArrayIndex.point(dstPromptIdx), NDArrayIndex.all()).assign(srcRow);
                } finally {
                    closeArray(srcRow);
                }
            }
        }

        log.info("{}: reordered frames with sourceOrderForTargets={}", name, Arrays.toString(sourceOrderForTargets));
        return new VariantInputs(name, base.promptTokenIds, reordered, base.totalFrames);
    }

    private FrameStatsSummary summarizeFrameStats(VariantInputs base) {
        FrameLayout layout = frameLayout(base);
        StringBuilder sb = new StringBuilder();
        boolean allFinite = true;
        for (int frameIdx = 0; frameIdx < base.totalFrames; frameIdx++) {
            double l2Sum = 0.0;
            double meanAbsSum = 0.0;
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            long valueCount = 0L;

            for (int offset = 0; offset < layout.rowsPerFrame; offset++) {
                int promptIdx = layout.imagePositions.get(frameIdx * layout.rowsPerFrame + offset);
                INDArray rowDup = base.inputsEmbeds.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(promptIdx), NDArrayIndex.all()).dup();
                try {
                    float[] row = rowDup.data().asFloat();
                    for (float v : row) {
                        double dv = v;
                        if (!Double.isFinite(dv)) {
                            allFinite = false;
                        }
                        l2Sum += dv * dv;
                        meanAbsSum += Math.abs(dv);
                        min = Math.min(min, dv);
                        max = Math.max(max, dv);
                        valueCount++;
                    }
                } finally {
                    closeArray(rowDup);
                }
            }

            double l2 = Math.sqrt(l2Sum);
            double meanAbs = valueCount == 0 ? 0.0 : meanAbsSum / valueCount;
            if (frameIdx > 0) {
                sb.append(" | ");
            }
            sb.append("f").append(frameIdx)
                    .append("{l2=").append(String.format("%.4f", l2))
                    .append(",meanAbs=").append(String.format("%.6f", meanAbs))
                    .append(",min=").append(String.format("%.6f", min))
                    .append(",max=").append(String.format("%.6f", max))
                    .append("}");
        }

        return new FrameStatsSummary(base.totalFrames, allFinite, sb.toString());
    }

    private FrameLayout frameLayout(VariantInputs base) {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        List<Integer> imagePositions = new ArrayList<>();
        for (int i = 0; i < base.promptTokenIds.length; i++) {
            if (base.promptTokenIds[i] == imageTokenId) {
                imagePositions.add(i);
            }
        }
        assertFalse(imagePositions.isEmpty(), "Expected image token positions for " + base.name);
        assertEquals(0, imagePositions.size() % base.totalFrames,
                "Image token rows should divide evenly across frames for " + base.name);
        return new FrameLayout(imagePositions, imagePositions.size() / base.totalFrames);
    }

    private INDArray encodeManually(INDArray imageInput, ImageTiler.SplitImageResult splitResult) {
        List<String> visionInputNames = visionEncoderSd.inputs();
        String[] visionOutputNames = visionEncoderSd.outputs().toArray(new String[0]);
        List<INDArray> frameEmbeddings = new ArrayList<>();
        try {
            for (int frameIdx = 0; frameIdx < splitResult.getTotalFrames(); frameIdx++) {
                INDArray frameSlice = imageInput.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();

                Map<String, INDArray> visionInputMap = new HashMap<>();
                for (String inputName : visionInputNames) {
                    if ("pixel_values".equals(inputName)) {
                        visionInputMap.put(inputName, singleFrame);
                    } else if ("pixel_attention_mask".equals(inputName)) {
                        ImageTiler.ContentRegion region = splitResult.contentRegions.get(frameIdx);
                        visionInputMap.put(inputName,
                                ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE));
                    }
                }

                Map<String, INDArray> visionOutputs = visionEncoderSd.output(visionInputMap, visionOutputNames);
                try {
                    VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
                    frameEmbeddings.add(selected.tensor.dup());
                } finally {
                    for (INDArray arr : visionOutputs.values()) {
                        closeArray(arr);
                    }
                    for (INDArray arr : visionInputMap.values()) {
                        closeArray(arr);
                    }
                    closeArray(singleFrame);
                }
            }

            return frameEmbeddings.size() == 1
                    ? frameEmbeddings.get(0).dup()
                    : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
        } finally {
            for (INDArray arr : frameEmbeddings) {
                closeArray(arr);
            }
            resetSameDiff(visionEncoderSd);
        }
    }

    private static INDArray embedPromptDirect(int[] tokenIds) {
        INDArray inputIds = Nd4j.createFromArray(new int[][]{tokenIds}).castTo(DataType.INT64);
        Map<String, INDArray> embedInputs = new HashMap<>();
        for (String inputName : embedTokens.inputs()) {
            embedInputs.put(inputName, inputIds);
        }
        Map<String, INDArray> embedOutputs = embedTokens.output(embedInputs,
                embedTokens.outputs().toArray(new String[0]));
        try {
            for (INDArray arr : embedOutputs.values()) {
                return arr.dup();
            }
            throw new IllegalStateException("embed_tokens model produced no output");
        } finally {
            closeArray(inputIds);
            for (INDArray arr : embedOutputs.values()) {
                closeArray(arr);
            }
            resetSameDiff(embedTokens);
        }
    }

    private DecodeSnapshot runOldDecoderDirect(VariantInputs variant, int maxTokens) throws Exception {
        resetDirectModelState(decoder);
        resetDirectModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        StaticKvCacheDecodeLoop loop = StaticKvCacheDecodeLoop.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(variant.inputsEmbeds.size(2))
                .build();

        long startNs = System.nanoTime();
        GenerationResult result = loop.decode(variant.inputsEmbeds.dup(), variant.promptTokenIds);
        long elapsedNs = System.nanoTime() - startNs;
        return new DecodeSnapshot(result, elapsedNs);
    }

    private static BufferedImage loadPageImage(int pdfPage) throws IOException {
        String configuredPath = System.getProperty("vlm.test.pdf.path");
        File pdfFile;
        if (configuredPath != null && !configuredPath.isBlank()) {
            pdfFile = new File(configuredPath);
        } else {
            pdfFile = new File(System.getProperty("user.dir"), "pathfinder-mythic.pdf");
        }
        assumeTrue(pdfFile.exists(), "PDF not found at " + pdfFile.getAbsolutePath()
                + ". Place pathfinder-mythic.pdf in platform-tests/ or set -Dvlm.test.pdf.path");
        int renderDpi = Integer.getInteger("vlm.test.pdf.dpi", 150);

        try (PDDocument document = PDDocument.load(pdfFile)) {
            PDFRenderer renderer = new PDFRenderer(document);
            BufferedImage image = renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
            log.info("Loaded benchmark PDF page {} at {} DPI: {}x{} from {}",
                    pdfPage, renderDpi, image.getWidth(), image.getHeight(), pdfFile.getAbsolutePath());
            return image;
        }
    }

    private static void resetSameDiff(SameDiff sd) {
        BenchmarkConfigApplier.resetModelState(sd);
        Nd4j.getExecutioner().commit();
    }

    private static void resetDirectModelState(SameDiff sd) {
        sd.clearPlaceholderOverrides();
        sd.clearPlaceholders(true);
        sd.clearOpInputs();
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        Nd4j.getExecutioner().commit();
    }

    private static void closeArray(INDArray arr) {
        if (arr != null && arr.closeable() && !arr.wasClosed()) {
            arr.close();
        }
    }

    private static void logDecode(VariantInputs variant, DecodeSnapshot snapshot) {
        log.info("{} decode: tokens={} elapsedMs={} text='{}'",
                variant.name,
                snapshot.result.getGeneratedTokenCount(),
                snapshot.elapsedNs / 1_000_000,
                safeSnippet(snapshot.result.getText(), 220));
        log.info("{} tokens={}", variant.name, Arrays.toString(snapshot.result.getTokenIds()));
    }

    private static boolean isPictureOnlyCollapse(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim().toLowerCase();
        return normalized.startsWith("<doctag><picture>")
                || (normalized.contains("<picture>") && normalized.contains("<end_of_utterance>"));
    }

    private static String safeSnippet(String text, int maxChars) {
        if (text == null) {
            return "null";
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim();
        if (normalized.length() <= maxChars) {
            return normalized;
        }
        return normalized.substring(0, maxChars) + "...";
    }

    private static final class VariantInputs implements AutoCloseable {
        private final String name;
        private final int[] promptTokenIds;
        private final INDArray inputsEmbeds;
        private final int totalFrames;

        private VariantInputs(String name, int[] promptTokenIds, INDArray inputsEmbeds, int totalFrames) {
            this.name = name;
            this.promptTokenIds = promptTokenIds;
            this.inputsEmbeds = inputsEmbeds;
            this.totalFrames = totalFrames;
        }

        @Override
        public void close() {
            closeArray(inputsEmbeds);
        }
    }

    private static final class FrameLayout {
        private final List<Integer> imagePositions;
        private final int rowsPerFrame;

        private FrameLayout(List<Integer> imagePositions, int rowsPerFrame) {
            this.imagePositions = imagePositions;
            this.rowsPerFrame = rowsPerFrame;
        }
    }

    private static final class FrameStatsSummary {
        private final int frameCount;
        private final boolean allFinite;
        private final String summary;

        private FrameStatsSummary(int frameCount, boolean allFinite, String summary) {
            this.frameCount = frameCount;
            this.allFinite = allFinite;
            this.summary = summary;
        }
    }

    private static final class DecodeSnapshot {
        private final GenerationResult result;
        private final long elapsedNs;

        private DecodeSnapshot(GenerationResult result, long elapsedNs) {
            this.result = result;
            this.elapsedNs = elapsedNs;
        }
    }
}
