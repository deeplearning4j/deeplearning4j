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
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.encoder.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.loading.OnnxModelCache;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoder;
import org.eclipse.deeplearning4j.vlm.model.encoder.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspDebugger;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.autodiff.samediff.execution.SlotState;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Isolates the benchmark-equivalent page-10 input pipeline:
 * resize strategy, VisionEncoder wrapper vs manual per-frame execution, and
 * how those choices affect early decode output before Triton/replay tuning.
 */
@Slf4j
public class TestGenerationPipelineBenchmarkInputPipeline {

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
    @DisplayName("Page-10 vision embeddings: VisionEncoder wrapper matches manual loop without resize")
    public void testPage10VisionWrapperParityWithoutResize() throws Exception {
        ensureModelsLoaded();

        VariantInputs manual = buildVariant("PAGE10_NO_RESIZE_MANUAL", 0, false);
        VariantInputs wrapped = buildVariant("PAGE10_NO_RESIZE_WRAPPER", 0, true);
        try {
            assertVariantInputParity(manual, wrapped, 1e-3, 1e-3);
        } finally {
            manual.close();
            wrapped.close();
        }
    }

    @Test
    @DisplayName("Page-10 vision embeddings: VisionEncoder wrapper matches manual loop with benchmark resize")
    public void testPage10VisionWrapperParityWithBenchmarkResize() throws Exception {
        ensureModelsLoaded();

        VariantInputs manual = buildVariant("PAGE10_RESIZE2048_MANUAL", 2048, false);
        VariantInputs wrapped = buildVariant("PAGE10_RESIZE2048_WRAPPER", 2048, true);
        try {
            assertVariantInputParity(manual, wrapped, 1e-3, 1e-3);
        } finally {
            manual.close();
            wrapped.close();
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: AUTO and SLOT_BY_SLOT produce the same benchmark-resized embeddings and decode prefix")
    public void testPage10VisionEncoderExecutionModeParityWithBenchmarkResize() throws Exception {
        ensureModelsLoaded();

        VariantInputs slotBySlot = buildVariant(
                "PAGE10_RESIZE2048_MANUAL_SLOT", 10, 2048, 9, false, GraphExecutionMode.SLOT_BY_SLOT);
        VariantInputs auto = buildVariant(
                "PAGE10_RESIZE2048_MANUAL_AUTO", 10, 2048, 9, false, GraphExecutionMode.AUTO);
        try {
            double visionDiff = maxAbsDiff(slotBySlot.visionEmbeddings, auto.visionEmbeddings);
            double mergedDiff = maxAbsDiff(slotBySlot.inputsEmbeds, auto.inputsEmbeds);
            log.info("Vision execution-mode parity maxTiles=9: overallVisionDiff={} overallMergedDiff={} perFrame={}",
                    visionDiff, mergedDiff, summarizeVisionPerFrameDiff(slotBySlot, auto));

            DecodeSnapshot slotDecode = runOldDecoderDirect(slotBySlot, 40);
            DecodeSnapshot autoDecode = runOldDecoderDirect(auto, 40);

            logDecode(slotBySlot, slotDecode);
            logDecode(auto, autoDecode);

            int firstDivergent = findFirstDivergentToken(
                    slotDecode.result.getTokenIds(), autoDecode.result.getTokenIds());
            log.info("Vision execution-mode decode divergence maxTiles=9: firstDivergent={} slotText='{}' autoText='{}'",
                    firstDivergent,
                    safeSnippet(slotDecode.result.getText(), 180),
                    safeSnippet(autoDecode.result.getText(), 180));

            assertTrue(visionDiff <= 1e-3,
                    "Vision embeddings differ between PAGE10_RESIZE2048_MANUAL_SLOT and PAGE10_RESIZE2048_MANUAL_AUTO "
                            + "maxAbsDiff=" + visionDiff);
            assertTrue(mergedDiff <= 1e-3,
                    "Merged inputs differ between PAGE10_RESIZE2048_MANUAL_SLOT and PAGE10_RESIZE2048_MANUAL_AUTO "
                            + "maxAbsDiff=" + mergedDiff);
            assertEquals(-1, firstDivergent,
                    "Vision encoder execution mode changed the page-10 decode prefix. "
                            + "If this fails, the remaining regression is in vision-model execution rather than decode.");
            assertEquals(slotDecode.result.getText(), autoDecode.result.getText(),
                    "Vision encoder execution mode changed the page-10 decoded text.");
        } finally {
            slotBySlot.close();
            auto.close();
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: AUTO and SLOT_BY_SLOT parity at maxTiles=5")
    public void testPage10VisionEncoderExecutionModeParityMaxTiles5() throws Exception {
        ensureModelsLoaded();

        VariantInputs slotBySlot = buildVariant(
                "PAGE10_MAXTILES5_MANUAL_SLOT", 10, 2048, 5, false, GraphExecutionMode.SLOT_BY_SLOT);
        VariantInputs auto = buildVariant(
                "PAGE10_MAXTILES5_MANUAL_AUTO", 10, 2048, 5, false, GraphExecutionMode.AUTO);
        try {
            double visionDiff = maxAbsDiff(slotBySlot.visionEmbeddings, auto.visionEmbeddings);
            double mergedDiff = maxAbsDiff(slotBySlot.inputsEmbeds, auto.inputsEmbeds);
            log.info("Vision execution-mode parity maxTiles=5: overallVisionDiff={} overallMergedDiff={} perFrame={}",
                    visionDiff, mergedDiff, summarizeVisionPerFrameDiff(slotBySlot, auto));

            assertTrue(visionDiff <= 1e-3,
                    "Vision embeddings differ between PAGE10_MAXTILES5_MANUAL_SLOT and PAGE10_MAXTILES5_MANUAL_AUTO "
                            + "maxAbsDiff=" + visionDiff);
            assertTrue(mergedDiff <= 1e-3,
                    "Merged inputs differ between PAGE10_MAXTILES5_MANUAL_SLOT and PAGE10_MAXTILES5_MANUAL_AUTO "
                            + "maxAbsDiff=" + mergedDiff);
        } finally {
            slotBySlot.close();
            auto.close();
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: single-frame AUTO matches single-frame SLOT_BY_SLOT when session is reset between frames")
    public void testPage10VisionEncoderSingleFrameExecutionModeParityMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5)) {
            List<FrameEncodingObservation> slotFrames = null;
            List<FrameEncodingObservation> autoFrames = null;
            try {
                slotFrames = encodeVisionFrames(prepared, GraphExecutionMode.SLOT_BY_SLOT, true, false);
                autoFrames = encodeVisionFrames(prepared, GraphExecutionMode.AUTO, true, true);

                String summary = summarizeFrameDiffs(slotFrames, autoFrames);
                log.info("Vision single-frame execution-mode parity maxTiles=5: {}", summary);

                for (int i = 0; i < slotFrames.size(); i++) {
                    double frameDiff = maxAbsDiff(slotFrames.get(i).embedding, autoFrames.get(i).embedding);
                    assertTrue(frameDiff <= 1e-3,
                            "Single-frame AUTO diverged from single-frame SLOT_BY_SLOT at frame " + i
                                    + " maxAbsDiff=" + frameDiff
                                    + " autoState=" + autoFrames.get(i).describeReplayState());
                }
            } finally {
                closeFrameObservations(slotFrames);
                closeFrameObservations(autoFrames);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: AUTO only drifts when the same session is reused across frames")
    public void testPage10VisionEncoderAutoSharedSessionDriftMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5)) {
            List<FrameEncodingObservation> slotFrames = null;
            List<FrameEncodingObservation> autoResetFrames = null;
            List<FrameEncodingObservation> autoSharedFrames = null;
            try {
                slotFrames = encodeVisionFrames(prepared, GraphExecutionMode.SLOT_BY_SLOT, true, false);
                autoResetFrames = encodeVisionFrames(prepared, GraphExecutionMode.AUTO, true, true);
                autoSharedFrames = encodeVisionFrames(prepared, GraphExecutionMode.AUTO, false, true);

                int firstResetDivergent = findFirstDivergentFrame(slotFrames, autoResetFrames, 1e-3);
                int firstSharedDivergent = findFirstDivergentFrame(slotFrames, autoSharedFrames, 1e-3);

                log.info("Vision AUTO reset-vs-shared maxTiles=5: resetSummary={} sharedSummary={}",
                        summarizeFrameDiffs(slotFrames, autoResetFrames),
                        summarizeFrameDiffs(slotFrames, autoSharedFrames));

                assertEquals(-1, firstResetDivergent,
                        "AUTO with a fresh session per frame should match SLOT_BY_SLOT frame-for-frame. "
                                + "If this fails, AUTO is wrong even for isolated single-frame execution.");
                assertTrue(firstSharedDivergent >= 0,
                        "Expected shared-session AUTO to diverge from SLOT_BY_SLOT when frames are reused in one session.");

                FrameEncodingObservation divergent = autoSharedFrames.get(firstSharedDivergent);
                log.info("First shared-session AUTO divergence at frame {}: {}",
                        firstSharedDivergent, divergent.describeReplayState());
            } finally {
                closeFrameObservations(slotFrames);
                closeFrameObservations(autoResetFrames);
                closeFrameObservations(autoSharedFrames);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: AUTO shared session honors same-identity host mutations after compile")
    public void testPage10VisionEncoderAutoSharedSessionSameIdentityMutationMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput slotFrame0 = prepareVisionFrameInput(prepared, 0, "slot-f0");
             VisionFrameInput slotFrame1 = prepareVisionFrameInput(prepared, 1, "slot-f1");
             VisionFrameInput autoFrame0 = prepareVisionFrameInput(prepared, 0, "auto-f0");
             VisionFrameInput autoFrame1 = prepareVisionFrameInput(prepared, 1, "auto-f1")) {

            List<FrameEncodingObservation> slotObservations = new ArrayList<>();
            List<FrameEncodingObservation> autoObservations = new ArrayList<>();
            try {
                beginVisionSession(GraphExecutionMode.SLOT_BY_SLOT);
                try {
                    slotObservations.add(executeVisionFrameInput(slotFrame0, false, 0));
                    slotObservations.add(executeVisionFrameInput(slotFrame1, false, 1));
                    slotFrame1.singleFrame.assign(0.0);
                    slotObservations.add(executeVisionFrameInput(slotFrame1, false, 2));
                } finally {
                    endVisionSession();
                }

                beginVisionSession(GraphExecutionMode.AUTO);
                try {
                    autoObservations.add(executeVisionFrameInput(autoFrame0, true, 0));
                    autoObservations.add(executeVisionFrameInput(autoFrame1, true, 1));
                    autoFrame1.singleFrame.assign(0.0);
                    autoObservations.add(executeVisionFrameInput(autoFrame1, true, 2));
                } finally {
                    endVisionSession();
                }

                double compileStepDiff = maxAbsDiff(slotObservations.get(1).embedding, autoObservations.get(1).embedding);
                double mutatedStepDiff = maxAbsDiff(slotObservations.get(2).embedding, autoObservations.get(2).embedding);
                double autoMutationDelta = maxAbsDiff(autoObservations.get(1).embedding, autoObservations.get(2).embedding);

                log.info("Vision AUTO same-identity mutation maxTiles=5: compileStepDiff={} mutatedStepDiff={} "
                                + "autoMutationDelta={} autoReplay={} parity={}",
                        compileStepDiff,
                        mutatedStepDiff,
                        autoMutationDelta,
                        summarizeReplayStates(autoObservations),
                        summarizeFrameDiffs(slotObservations, autoObservations));

                assertTrue(compileStepDiff <= 1e-3,
                        "AUTO diverged from SLOT_BY_SLOT before the mutation step. compileStepDiff=" + compileStepDiff);
                assertTrue(mutatedStepDiff <= 1e-3,
                        "AUTO ignored or miscomputed a same-identity host mutation after compile. "
                                + "If this fails, the frozen shared-session path is not syncing host writes.");
                assertTrue(autoMutationDelta > 1e-3,
                        "AUTO produced the same embedding after zeroing the frame in-place. "
                                + "If this fails, the compiled path never observed the host mutation.");
            } finally {
                closeFrameObservations(slotObservations);
                closeFrameObservations(autoObservations);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: AUTO shared session honors fresh-identity replacements after compile")
    public void testPage10VisionEncoderAutoSharedSessionFreshIdentityMutationMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput slotFrame0 = prepareVisionFrameInput(prepared, 0, "slot-f0");
             VisionFrameInput slotFrame1 = prepareVisionFrameInput(prepared, 1, "slot-f1");
             VisionFrameInput slotFrame1Zero = prepareVisionFrameInput(prepared, 1, "slot-f1-zero");
             VisionFrameInput autoFrame0 = prepareVisionFrameInput(prepared, 0, "auto-f0");
             VisionFrameInput autoFrame1 = prepareVisionFrameInput(prepared, 1, "auto-f1");
             VisionFrameInput autoFrame1Zero = prepareVisionFrameInput(prepared, 1, "auto-f1-zero")) {

            slotFrame1Zero.singleFrame.assign(0.0);
            autoFrame1Zero.singleFrame.assign(0.0);

            List<FrameEncodingObservation> slotObservations = null;
            List<FrameEncodingObservation> autoObservations = null;
            try {
                slotObservations = executeVisionFrameInputs(GraphExecutionMode.SLOT_BY_SLOT, false,
                        slotFrame0, slotFrame1, slotFrame1Zero);
                autoObservations = executeVisionFrameInputs(GraphExecutionMode.AUTO, true,
                        autoFrame0, autoFrame1, autoFrame1Zero);

                double compileStepDiff = maxAbsDiff(slotObservations.get(1).embedding, autoObservations.get(1).embedding);
                double replacedStepDiff = maxAbsDiff(slotObservations.get(2).embedding, autoObservations.get(2).embedding);
                double autoReplacementDelta = maxAbsDiff(autoObservations.get(1).embedding, autoObservations.get(2).embedding);

                log.info("Vision AUTO fresh-identity replacement maxTiles=5: compileStepDiff={} replacedStepDiff={} "
                                + "autoReplacementDelta={} autoReplay={} parity={}",
                        compileStepDiff,
                        replacedStepDiff,
                        autoReplacementDelta,
                        summarizeReplayStates(autoObservations),
                        summarizeFrameDiffs(slotObservations, autoObservations));

                assertTrue(compileStepDiff <= 1e-3,
                        "AUTO diverged from SLOT_BY_SLOT before the replacement step. compileStepDiff=" + compileStepDiff);
                assertTrue(replacedStepDiff <= 1e-3,
                        "AUTO ignored or miscomputed a fresh-identity replacement after compile. "
                                + "If this fails while same-identity mutation passes, the bug is stale pointer/arg-table refresh.");
                assertTrue(autoReplacementDelta > 1e-3,
                        "AUTO produced the same embedding after swapping in a zeroed fresh array. "
                                + "If this fails, the compiled path never observed the replacement input.");
            } finally {
                closeFrameObservations(slotObservations);
                closeFrameObservations(autoObservations);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: SLOT_BY_SLOT shared session can repeat the same frame identity without mutation")
    public void testPage10VisionEncoderSlotSharedSessionSameIdentityRepeatMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput frame0 = prepareVisionFrameInput(prepared, 0, "slot-repeat-f0");
             VisionFrameInput frame1 = prepareVisionFrameInput(prepared, 1, "slot-repeat-f1")) {

            List<FrameEncodingObservation> observations = null;
            try {
                observations = executeVisionFrameInputs(GraphExecutionMode.SLOT_BY_SLOT, false,
                        frame0, frame1, frame1);

                double repeatedDelta = maxAbsDiff(observations.get(1).embedding, observations.get(2).embedding);
                log.info("Vision SLOT_BY_SLOT same-identity repeat maxTiles=5: repeatedDelta={}",
                        repeatedDelta);
                assertTrue(repeatedDelta <= 1e-3,
                        "Repeating the same frame identity in one shared SLOT_BY_SLOT session should "
                                + "reproduce the same embedding.");
            } finally {
                closeFrameObservations(observations);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: SLOT_BY_SLOT shared session can repeat a fresh frame identity without mutation")
    public void testPage10VisionEncoderSlotSharedSessionFreshIdentityRepeatMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput frame0 = prepareVisionFrameInput(prepared, 0, "slot-fresh-repeat-f0");
             VisionFrameInput frame1 = prepareVisionFrameInput(prepared, 1, "slot-fresh-repeat-f1");
             VisionFrameInput frame1Copy = prepareVisionFrameInput(prepared, 1, "slot-fresh-repeat-f1-copy")) {

            List<FrameEncodingObservation> observations = null;
            try {
                observations = executeVisionFrameInputs(GraphExecutionMode.SLOT_BY_SLOT, false,
                        frame0, frame1, frame1Copy);

                double repeatedDelta = maxAbsDiff(observations.get(1).embedding, observations.get(2).embedding);
                log.info("Vision SLOT_BY_SLOT fresh-identity repeat maxTiles=5: repeatedDelta={}",
                        repeatedDelta);
                assertTrue(repeatedDelta <= 1e-3,
                        "Repeating a fresh copy of the same frame in one shared SLOT_BY_SLOT session should "
                                + "reproduce the same embedding.");
            } finally {
                closeFrameObservations(observations);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: explicit sync after zeroing still hits the same conv2d rank failure in shared SLOT_BY_SLOT")
    public void testPage10VisionEncoderSlotSharedSessionZeroMutationWithExplicitSyncMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput frame0 = prepareVisionFrameInput(prepared, 0, "slot-sync-f0");
             VisionFrameInput frame1 = prepareVisionFrameInput(prepared, 1, "slot-sync-f1")) {

            List<FrameEncodingObservation> observations = new ArrayList<>();
            try {
                beginVisionSession(GraphExecutionMode.SLOT_BY_SLOT);
                try {
                    observations.add(executeVisionFrameInput(frame0, false, 0));
                    observations.add(executeVisionFrameInput(frame1, false, 1));
                    frame1.singleFrame.assign(0.0);
                    frame1.singleFrame.syncToDevice();
                    RuntimeException failure = assertThrows(RuntimeException.class,
                            () -> executeVisionFrameInput(frame1, false, 2));
                    String failureSummary = summarizeThrowableMessages(failure);
                    log.info("Vision SLOT_BY_SLOT zero mutation with explicit sync failure summary: {}",
                            failureSummary);
                    assertTrue(failureSummary.contains("slot 281 (conv2d)")
                                    || failureSummary.contains("CUSTOM CONV2D OP")
                                    || failureSummary.contains("rank of input array must be equal to 4"),
                            "Expected the shared-session zeroed frame to fail with the same conv2d rank issue, but got: "
                                    + failureSummary);
                } finally {
                    endVisionSession();
                }
            } finally {
                closeFrameObservations(observations);
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: zero-mutation failure snapshots permute/conv2d slot neighborhood")
    public void testPage10VisionEncoderSlotSharedSessionZeroMutationSlotNeighborhoodMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput frame0 = prepareVisionFrameInput(prepared, 0, "slot-neighborhood-f0");
             VisionFrameInput frame1 = prepareVisionFrameInput(prepared, 1, "slot-neighborhood-f1")) {

            beginVisionSession(GraphExecutionMode.SLOT_BY_SLOT);
            try {
                executeVisionFrameInput(frame0, false, 0);
                executeVisionFrameInput(frame1, false, 1);

                String before = describeVisionRuntimeSlotNeighborhood(276, 277, 278, 279, 280, 281, 282);
                log.info("Vision zero-mutation slot neighborhood before failure:\n{}", before);

                frame1.singleFrame.assign(0.0);
                frame1.singleFrame.syncToDevice();

                RuntimeException failure = assertThrows(RuntimeException.class,
                        () -> executeVisionFrameInput(frame1, false, 2));

                String after = describeVisionRuntimeSlotNeighborhood(276, 277, 278, 279, 280, 281, 282);
                log.info("Vision zero-mutation slot neighborhood after failure:\n{}", after);

                String failureSummary = summarizeThrowableMessages(failure);
                assertTrue(failureSummary.contains("slot 281 (conv2d)")
                                || failureSummary.contains("CUSTOM CONV2D OP"),
                        "Expected the shared-session zero-mutation repro to fail at conv2d slot 281, but got: "
                                + failureSummary);
            } finally {
                endVisionSession();
            }
        }
    }

    @Test
    @DisplayName("Page-10 vision encoder: a zeroed frame also fails in a fresh SLOT_BY_SLOT session")
    public void testPage10VisionEncoderSlotFreshSessionZeroedFrameMaxTiles5() throws Exception {
        ensureModelsLoaded();

        try (PreparedVisionInput prepared = prepareVisionInput(10, 2048, 5);
             VisionFrameInput frame1 = prepareVisionFrameInput(prepared, 1, "slot-fresh-zero-f1")) {

            frame1.singleFrame.assign(0.0);
            frame1.singleFrame.syncToDevice();

            RuntimeException failure = assertThrows(RuntimeException.class,
                    () -> executeVisionFrameInputs(GraphExecutionMode.SLOT_BY_SLOT, false, frame1));

            String failureSummary = summarizeThrowableMessages(failure);
            log.info("Vision fresh-session zeroed-frame failure summary: {}", failureSummary);

            assertTrue(failureSummary.contains("slot 281 (conv2d)")
                            || failureSummary.contains("CUSTOM CONV2D OP")
                            || failureSummary.contains("rank of input array must be equal to 4"),
                    "Expected a fresh-session zeroed frame to expose the same conv2d rank failure, but got: "
                            + failureSummary);
        }
    }

    @Test
    @DisplayName("Page-10 explicit resize is redundant with splitImageForVLM and preserves old-decoder SLOT_BY_SLOT prefix")
    public void testPage10ResizeRedundancyOldDecoderSlotBySlot() throws Exception {
        ensureModelsLoaded();

        VariantInputs noResize = buildVariant("PAGE10_NO_RESIZE_MANUAL", 10, 0, false);
        VariantInputs resized = buildVariant("PAGE10_RESIZE2048_MANUAL", 10, 2048, false);
        try {
            boolean geometryDiffers = noResize.totalFrames != resized.totalFrames
                    || noResize.numRows != resized.numRows
                    || noResize.numCols != resized.numCols
                    || noResize.promptTokenIds.length != resized.promptTokenIds.length
                    || !Arrays.equals(noResize.promptTokenIds, resized.promptTokenIds)
                    || !Arrays.equals(noResize.inputsEmbeds.shape(), resized.inputsEmbeds.shape());
            assertFalse(geometryDiffers,
                    "Explicit resize to longest-edge 2048 should be redundant once splitImageForVLM normalizes page 10. "
                            + "If this changed, the benchmark path and non-benchmark path are not actually equivalent.");

            DecodeSnapshot noResizeDecode = runOldDecoderSlotBySlot(noResize, 40);
            DecodeSnapshot resizedDecode = runOldDecoderSlotBySlot(resized, 40);

            logDecode(noResize, noResizeDecode);
            logDecode(resized, resizedDecode);

            int firstDivergent = findFirstDivergentToken(noResizeDecode.result.getTokenIds(),
                    resizedDecode.result.getTokenIds());
            assertEquals(-1, firstDivergent,
                    "Explicit resize changed the old-decoder token prefix even though the post-tiling geometry is identical.");
            assertEquals(noResizeDecode.result.getText(), resizedDecode.result.getText(),
                    "Explicit resize changed the old-decoder decoded text even though the post-tiling geometry is identical.");
        } finally {
            noResize.close();
            resized.close();
        }
    }

    @Test
    @DisplayName("Benchmark-style page contrast: page 10 has a larger visual prompt than page 0")
    public void testBenchmarkPageContrastInputGeometry() throws Exception {
        ensureModelsLoaded();

        VariantInputs page0 = buildVariant("PAGE0_RESIZE2048_WRAPPER", 0, 2048, true);
        VariantInputs page10 = buildVariant("PAGE10_RESIZE2048_WRAPPER", 10, 2048, true);
        try {
            log.info("Page contrast geometry: page0 frames={} grid={}x{} promptTokens={} mergedShape={} | page10 frames={} grid={}x{} promptTokens={} mergedShape={}",
                    page0.totalFrames, page0.numRows, page0.numCols, page0.promptTokenIds.length,
                    Arrays.toString(page0.inputsEmbeds.shape()),
                    page10.totalFrames, page10.numRows, page10.numCols, page10.promptTokenIds.length,
                    Arrays.toString(page10.inputsEmbeds.shape()));

            assertTrue(page10.totalFrames >= page0.totalFrames,
                    "Page 10 should not have fewer benchmark tiles than page 0");
            assertTrue(page10.promptTokenIds.length >= page0.promptTokenIds.length,
                    "Page 10 should not have a shorter merged prompt than page 0 under benchmark tiling");
            assertTrue(page10.inputsEmbeds.size(1) >= page0.inputsEmbeds.size(1),
                    "Page 10 should not have a shorter merged embedding sequence than page 0");
        } finally {
            page0.close();
            page10.close();
        }
    }

    @Test
    @DisplayName("Page-0 resized benchmark path: direct, apply-only, and apply+compile all collapse identically")
    public void testPage0OldDecoderBenchmarkHarnessBisection() throws Exception {
        ensureModelsLoaded();

        VariantInputs page0 = buildVariant("PAGE0_RESIZE2048_WRAPPER", 0, 2048, true);
        try {
            int maxTokens = 40;
            DecodeSnapshot direct = runOldDecoderDirect(page0, maxTokens);
            DecodeSnapshot applyOnly = runOldDecoderWithConfig(page0, maxTokens, true, false);
            DecodeSnapshot applyAndCompile = runOldDecoderWithConfig(page0, maxTokens, true, true);

            logDecode(page0.withName("PAGE0_DIRECT"), direct);
            logDecode(page0.withName("PAGE0_APPLY_ONLY"), applyOnly);
            logDecode(page0.withName("PAGE0_APPLY_COMPILE"), applyAndCompile);

            String directText = direct.result.getText();
            String applyOnlyText = applyOnly.result.getText();
            String compiledText = applyAndCompile.result.getText();

            boolean applyOnlyCollapsed = isPictureOnlyCollapse(applyOnlyText);
            boolean compileCollapsed = isPictureOnlyCollapse(compiledText);
            boolean directCollapsed = isPictureOnlyCollapse(directText);
            log.info("Page-0 harness bisection: directCollapsed={} applyOnlyCollapsed={} compileCollapsed={}",
                    directCollapsed, applyOnlyCollapsed, compileCollapsed);

            assertTrue(directCollapsed,
                    "Resized page-0 direct run should reproduce the same picture-only collapse seen in the benchmark-style path. Text: "
                            + safeSnippet(directText, 220));
            assertTrue(applyOnlyCollapsed,
                    "Apply-only resized page-0 run should reproduce the same picture-only collapse. Text: "
                            + safeSnippet(applyOnlyText, 220));
            assertTrue(compileCollapsed,
                    "Apply+compile resized page-0 run should reproduce the same picture-only collapse. Text: "
                            + safeSnippet(compiledText, 220));
            assertArrayEquals(direct.result.getTokenIds(), applyOnly.result.getTokenIds(),
                    "Direct and apply-only resized page-0 runs should match token-for-token.");
            assertArrayEquals(direct.result.getTokenIds(), applyAndCompile.result.getTokenIds(),
                    "Direct and apply+compile resized page-0 runs should match token-for-token.");
            assertEquals(directText, applyOnlyText,
                    "Direct and apply-only resized page-0 decoded text should match.");
            assertEquals(directText, compiledText,
                    "Direct and apply+compile resized page-0 decoded text should match.");
        } finally {
            page0.close();
        }
    }

    @Test
    @DisplayName("Page-0 direct old decoder: no-resize and resize-2048 converge to the same splitImageForVLM geometry")
    public void testPage0ResizeSensitivityOldDecoderDirect() throws Exception {
        ensureModelsLoaded();

        VariantInputs noResize = buildVariant("PAGE0_NO_RESIZE_MANUAL", 0, 0, false);
        VariantInputs resized = buildVariant("PAGE0_RESIZE2048_MANUAL", 0, 2048, false);
        try {
            log.info("Page-0 resize sensitivity geometry: noResize frames={} grid={}x{} promptTokens={} mergedShape={} | resized frames={} grid={}x{} promptTokens={} mergedShape={}",
                    noResize.totalFrames, noResize.numRows, noResize.numCols, noResize.promptTokenIds.length,
                    Arrays.toString(noResize.inputsEmbeds.shape()),
                    resized.totalFrames, resized.numRows, resized.numCols, resized.promptTokenIds.length,
                    Arrays.toString(resized.inputsEmbeds.shape()));

            assertEquals(noResize.totalFrames, resized.totalFrames,
                    "Page-0 no-resize and resize-2048 should converge to the same frame count after splitImageForVLM normalization.");
            assertEquals(noResize.promptTokenIds.length, resized.promptTokenIds.length,
                    "Page-0 no-resize and resize-2048 should produce the same prompt length after splitImageForVLM normalization.");
            assertArrayEquals(noResize.inputsEmbeds.shape(), resized.inputsEmbeds.shape(),
                    "Page-0 no-resize and resize-2048 should produce the same merged embedding shape.");

            DecodeSnapshot noResizeDecode = runOldDecoderDirect(noResize, 20);
            DecodeSnapshot resizedDecode = runOldDecoderDirect(resized, 20);

            logDecode(noResize, noResizeDecode);
            logDecode(resized, resizedDecode);

            assertArrayEquals(noResizeDecode.result.getTokenIds(), resizedDecode.result.getTokenIds(),
                    "Page-0 no-resize and resize-2048 should produce identical token output after geometry convergence.");
            assertEquals(noResizeDecode.result.getText(), resizedDecode.result.getText(),
                    "Page-0 no-resize and resize-2048 should produce identical decoded text.");
        } finally {
            noResize.close();
            resized.close();
        }
    }

    @Test
    @DisplayName("Page-10 old decoder reacts to tile-count changes instead of emitting the same collapse for every visual prompt size")
    public void testPage10TileCountSensitivityOldDecoderDirect() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles1 = buildVariant("PAGE10_MAXTILES1_MANUAL", 10, 2048, 1, false);
        VariantInputs maxTiles4 = buildVariant("PAGE10_MAXTILES4_MANUAL", 10, 2048, 4, false);
        VariantInputs maxTiles9 = buildVariant("PAGE10_MAXTILES9_MANUAL", 10, 2048, 9, false);
        try {
            assertTrue(maxTiles1.promptTokenIds.length < maxTiles9.promptTokenIds.length,
                    "Reducing page-10 maxTiles should shorten the prompt. "
                            + "maxTiles=1 promptTokens=" + maxTiles1.promptTokenIds.length
                            + " maxTiles=9 promptTokens=" + maxTiles9.promptTokenIds.length);
            assertTrue(maxTiles1.inputsEmbeds.size(1) < maxTiles9.inputsEmbeds.size(1),
                    "Reducing page-10 maxTiles should shorten merged embeddings. "
                            + "maxTiles=1 mergedSeq=" + maxTiles1.inputsEmbeds.size(1)
                            + " maxTiles=9 mergedSeq=" + maxTiles9.inputsEmbeds.size(1));

            DecodeSnapshot decode1 = runOldDecoderDirect(maxTiles1, 40);
            DecodeSnapshot decode4 = runOldDecoderDirect(maxTiles4, 40);
            DecodeSnapshot decode9 = runOldDecoderDirect(maxTiles9, 40);

            logDecode(maxTiles1, decode1);
            logDecode(maxTiles4, decode4);
            logDecode(maxTiles9, decode9);

            boolean oneVsNineSameTokens = Arrays.equals(decode1.result.getTokenIds(), decode9.result.getTokenIds());
            boolean oneVsNineSameText = decode1.result.getText().equals(decode9.result.getText());
            assertFalse(oneVsNineSameTokens && oneVsNineSameText,
                    "Changing page-10 tile count from 1 to 9 did not change the old-decoder output at all. "
                            + "That would mean the benchmark collapse is insensitive to large changes in visual prompt size. "
                            + "maxTiles=1 text='" + safeSnippet(decode1.result.getText(), 220)
                            + "' maxTiles=9 text='" + safeSnippet(decode9.result.getText(), 220) + "'");
        } finally {
            maxTiles1.close();
            maxTiles4.close();
            maxTiles9.close();
        }
    }

    @Test
    @DisplayName("Page-10 old decoder changes when visual-token embeddings are zeroed")
    public void testPage10VisualTokenAblationOldDecoderDirect() throws Exception {
        ensureModelsLoaded();

        VariantInputs full = buildVariant("PAGE10_FULL_VISUAL_MANUAL", 10, 2048, 9, false);
        VariantInputs zeroed = zeroVisualTokenEmbeddings("PAGE10_ZERO_VISUAL_MANUAL", full);
        try {
            DecodeSnapshot fullDecode = runOldDecoderDirect(full, 40);
            DecodeSnapshot zeroDecode = runOldDecoderDirect(zeroed, 40);

            logDecode(full, fullDecode);
            logDecode(zeroed, zeroDecode);

            boolean sameTokens = Arrays.equals(fullDecode.result.getTokenIds(), zeroDecode.result.getTokenIds());
            boolean sameText = fullDecode.result.getText().equals(zeroDecode.result.getText());
            assertFalse(sameTokens && sameText,
                    "Zeroing the page-10 visual-token rows did not change the old-decoder output. "
                            + "That implies the merged image embeddings are not influencing generation. "
                            + "full='" + safeSnippet(fullDecode.result.getText(), 220)
                            + "' zeroVisual='" + safeSnippet(zeroDecode.result.getText(), 220) + "'");
        } finally {
            full.close();
            zeroed.close();
        }
    }

    @Test
    @DisplayName("Page-10 tile-count boundary is shared by StaticKvCacheDecodeLoop and GenerationPipeline")
    public void testPage10TileCountSensitivityGenerationPipelineParity() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles4 = buildVariant("PAGE10_MAXTILES4_MANUAL", 10, 2048, 4, false);
        VariantInputs maxTiles9 = buildVariant("PAGE10_MAXTILES9_MANUAL", 10, 2048, 9, false);
        try {
            DecodeSnapshot old4 = runOldDecoderDirect(maxTiles4, 40);
            DecodeSnapshot new4 = runGenerationPipelineDirect(maxTiles4, 40);
            DecodeSnapshot old9 = runOldDecoderDirect(maxTiles9, 40);
            DecodeSnapshot new9 = runGenerationPipelineDirect(maxTiles9, 40);

            logDecode(maxTiles4.withName("PAGE10_MAXTILES4_OLD"), old4);
            logDecode(maxTiles4.withName("PAGE10_MAXTILES4_NEW"), new4);
            logDecode(maxTiles9.withName("PAGE10_MAXTILES9_OLD"), old9);
            logDecode(maxTiles9.withName("PAGE10_MAXTILES9_NEW"), new9);

            assertArrayEquals(old4.result.getTokenIds(), new4.result.getTokenIds(),
                    "GenerationPipeline diverged from the old decoder on page 10 with maxTiles=4.");
            assertEquals(old4.result.getText(), new4.result.getText(),
                    "GenerationPipeline text diverged from the old decoder on page 10 with maxTiles=4.");

            int[] old9Tokens = old9.result.getTokenIds();
            int[] new9Tokens = new9.result.getTokenIds();
            assertTrue(new9Tokens.length >= old9Tokens.length,
                    "GenerationPipeline should not terminate earlier than the old decoder at maxTiles=9.");
            assertArrayEquals(old9Tokens, Arrays.copyOf(new9Tokens, old9Tokens.length),
                    "GenerationPipeline should share the same picture-collapse prefix as the old decoder at maxTiles=9.");
            assertTrue(new9.result.getText().startsWith(old9.result.getText()),
                    "GenerationPipeline maxTiles=9 text should contain the same collapse prefix. old='"
                            + safeSnippet(old9.result.getText(), 220) + "' new='"
                            + safeSnippet(new9.result.getText(), 220) + "'");
        } finally {
            maxTiles4.close();
            maxTiles9.close();
        }
    }

    @Test
    @DisplayName("Page-10 GenerationPipeline has a tile-count transition between textual output and picture collapse")
    public void testPage10GenerationPipelineTileCountTransitionMatrix() throws Exception {
        ensureModelsLoaded();

        int[] tileCounts = {4, 5, 6, 7, 8, 9};
        List<VariantInputs> variants = new ArrayList<>();
        List<TileDecodeObservation> observations = new ArrayList<>();
        try {
            int previousPromptLen = -1;
            for (int maxTiles : tileCounts) {
                VariantInputs variant = buildVariant("PAGE10_MAXTILES" + maxTiles + "_NEW", 10, 2048, maxTiles, false);
                variants.add(variant);
                assertTrue(variant.promptTokenIds.length >= previousPromptLen,
                        "Prompt length should be non-decreasing as page-10 maxTiles increases. previous=" + previousPromptLen
                                + " current=" + variant.promptTokenIds.length + " at maxTiles=" + maxTiles);
                previousPromptLen = variant.promptTokenIds.length;

                DecodeSnapshot decode = runGenerationPipelineDirect(variant, 40);
                logDecode(variant, decode);
                observations.add(new TileDecodeObservation(
                        maxTiles,
                        variant.promptTokenIds.length,
                        decode.result.getText(),
                        decode.result.getTokenIds()));
            }

            Integer firstCollapseTile = null;
            Integer lastTextualTile = null;
            boolean sawTextual = false;
            for (TileDecodeObservation observation : observations) {
                boolean collapsed = isPictureOnlyCollapse(observation.text);
                boolean textual = hasTextualStructure(observation.text) && !collapsed;
                log.info("Tile transition: maxTiles={} promptTokens={} collapsed={} textual={} text='{}'",
                        observation.maxTiles,
                        observation.promptTokens,
                        collapsed,
                        textual,
                        safeSnippet(observation.text, 220));
                if (textual) {
                    sawTextual = true;
                    lastTextualTile = observation.maxTiles;
                }
                if (collapsed && firstCollapseTile == null) {
                    firstCollapseTile = observation.maxTiles;
                }
            }

            TileDecodeObservation maxTiles9 = observations.get(observations.size() - 1);
            assertTrue(isPictureOnlyCollapse(maxTiles9.text),
                    "Page-10 GenerationPipeline should still reproduce the benchmark picture-only collapse at maxTiles=9. "
                            + "text='" + safeSnippet(maxTiles9.text, 220) + "'");
            assertTrue(sawTextual,
                    "GenerationPipeline never produced textual docling structure for any tested page-10 tile count. "
                            + "Observed outputs: " + observationsSummary(observations));
            assertNotNull(firstCollapseTile,
                    "GenerationPipeline never entered picture-only collapse for any tested page-10 tile count. "
                            + "Observed outputs: " + observationsSummary(observations));
            assertNotNull(lastTextualTile,
                    "GenerationPipeline produced no textual page-10 output before collapsing. "
                            + "Observed outputs: " + observationsSummary(observations));
            assertTrue(lastTextualTile < firstCollapseTile,
                    "Expected a real page-10 transition from textual output to picture collapse as tile count grows. "
                            + "lastTextual=" + lastTextualTile + " firstCollapse=" + firstCollapseTile
                            + " observations=" + observationsSummary(observations));
        } finally {
            for (VariantInputs variant : variants) {
                variant.close();
            }
        }
    }

    @Test
    @DisplayName("Page-10 maxTiles=4 vs maxTiles=9 diverges after prefill and warmup")
    public void testPage10TileBoundaryDivergesAfterWarmup() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles4 = buildVariant("PAGE10_MAXTILES4_MANUAL", 10, 2048, 4, false);
        VariantInputs maxTiles9 = buildVariant("PAGE10_MAXTILES9_MANUAL", 10, 2048, 9, false);
        try {
            WarmupSignals signals4 = capturePrefillAndWarmupTokensPadded(maxTiles4, 4);
            WarmupSignals signals9 = capturePrefillAndWarmupTokensPadded(maxTiles9, 4);

            DecodeSnapshot decode4 = runOldDecoderDirect(maxTiles4, 40);
            DecodeSnapshot decode9 = runOldDecoderDirect(maxTiles9, 40);
            logDecode(maxTiles4.withName("PAGE10_MAXTILES4_BOUNDARY"), decode4);
            logDecode(maxTiles9.withName("PAGE10_MAXTILES9_BOUNDARY"), decode9);

            int firstDivergent = findFirstDivergentToken(decode4.result.getTokenIds(), decode9.result.getTokenIds());
            assertEquals(signals4.prefillTokenId, signals9.prefillTokenId,
                    "Page-10 tile boundary should not diverge at prefill token.");
            assertEquals(signals4.warmupTokenId, signals9.warmupTokenId,
                    "Page-10 tile boundary should not diverge at warmup token.");
            assertEquals(2, firstDivergent,
                    "Expected page-10 tile boundary to diverge immediately after warmup (token index 2). "
                            + "signals4=" + signals4.describe(tokenizer)
                            + " signals9=" + signals9.describe(tokenizer)
                            + " tokens4=" + Arrays.toString(Arrays.copyOf(decode4.result.getTokenIds(),
                            Math.min(8, decode4.result.getTokenIds().length)))
                            + " tokens9=" + Arrays.toString(Arrays.copyOf(decode9.result.getTokenIds(),
                            Math.min(8, decode9.result.getTokenIds().length))));

            log.info("Boundary divergence confirmed: prefill/warmup shared, firstDivergent={} token4={} ('{}') token9={} ('{}')",
                    firstDivergent,
                    decode4.result.getTokenIds()[firstDivergent],
                    tokenizer.decode(new int[]{decode4.result.getTokenIds()[firstDivergent]}, false),
                    decode9.result.getTokenIds()[firstDivergent],
                    tokenizer.decode(new int[]{decode9.result.getTokenIds()[firstDivergent]}, false));
        } finally {
            maxTiles4.close();
            maxTiles9.close();
        }
    }

    @Test
    @DisplayName("Page-10 handoff-state comparison: maxTiles=5 vs maxTiles=6 warmup logits already diverge")
    public void testPage10HandoffStateComparisonPrefillVsCollapse() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles5 = buildVariant("PAGE10_MAXTILES5_HANDOFF", 10, 2048, 5, false);
        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_HANDOFF", 10, 2048, 6, false);
        try {
            HandoffState state5 = captureHandoffState(maxTiles5, 4);
            HandoffState state6 = captureHandoffState(maxTiles6, 4);

            log.info("Handoff state maxTiles=5: prefillToken={} ('{}') warmupToken={} ('{}') kvFingerprint={} firstDecodeLogitsTop3={}",
                    state5.prefillTokenId,
                    tokenizer.decode(new int[]{state5.prefillTokenId}, false),
                    state5.warmupTokenId,
                    tokenizer.decode(new int[]{state5.warmupTokenId}, false),
                    state5.kvCacheFingerprint,
                    Arrays.toString(state5.firstDecodeLogitsTop3));

            log.info("Handoff state maxTiles=6: prefillToken={} ('{}') warmupToken={} ('{}') kvFingerprint={} firstDecodeLogitsTop3={}",
                    state6.prefillTokenId,
                    tokenizer.decode(new int[]{state6.prefillTokenId}, false),
                    state6.warmupTokenId,
                    tokenizer.decode(new int[]{state6.warmupTokenId}, false),
                    state6.kvCacheFingerprint,
                    Arrays.toString(state6.firstDecodeLogitsTop3));

            assertEquals(state5.prefillTokenId, state6.prefillTokenId,
                    "Prefill token should be identical between maxTiles=5 and maxTiles=6.");
            assertEquals(state5.warmupTokenId, state6.warmupTokenId,
                    "Warmup token should be identical between maxTiles=5 and maxTiles=6.");

            assertNotEquals(state5.kvCacheFingerprint, state6.kvCacheFingerprint,
                    "KV cache fingerprints should differ between maxTiles=5 and maxTiles=6 since the visual prompt sizes differ.");

            assertFalse(Arrays.equals(state5.firstDecodeLogitsTop3, state6.firstDecodeLogitsTop3),
                    "First-decode-step top-3 logits should differ between maxTiles=5 (textual) and maxTiles=6 (collapse). "
                            + "This proves the long visual prompt already perturbs warmup logits differently. "
                            + "top3_5=" + Arrays.toString(state5.firstDecodeLogitsTop3)
                            + " top3_6=" + Arrays.toString(state6.firstDecodeLogitsTop3));

            log.info("Handoff comparison confirmed: prefill/warmup tokens shared, KV fingerprints differ, first-decode logits differ. "
                    + "The long-prompt collapse originates at the warmup step, not later in decode.");
        } finally {
            maxTiles5.close();
            maxTiles6.close();
        }
    }

    @Test
    @DisplayName("Page-10 transition boundary at maxTiles=5/6: old decoder and GenerationPipeline produce identical output")
    public void testPage10TileTransitionBoundaryParityMaxTiles5And6() throws Exception {
        ensureModelsLoaded();

        VariantInputs maxTiles5 = buildVariant("PAGE10_MAXTILES5_MANUAL", 10, 2048, 5, false);
        VariantInputs maxTiles6 = buildVariant("PAGE10_MAXTILES6_MANUAL", 10, 2048, 6, false);
        try {
            DecodeSnapshot old5 = runOldDecoderDirect(maxTiles5, 40);
            DecodeSnapshot new5 = runGenerationPipelineDirect(maxTiles5, 40);
            DecodeSnapshot old6 = runOldDecoderDirect(maxTiles6, 40);
            DecodeSnapshot new6 = runGenerationPipelineDirect(maxTiles6, 40);

            logDecode(maxTiles5.withName("PAGE10_MAXTILES5_OLD"), old5);
            logDecode(maxTiles5.withName("PAGE10_MAXTILES5_NEW"), new5);
            logDecode(maxTiles6.withName("PAGE10_MAXTILES6_OLD"), old6);
            logDecode(maxTiles6.withName("PAGE10_MAXTILES6_NEW"), new6);

            assertFalse(isPictureOnlyCollapse(old5.result.getText()),
                    "Old decoder should remain textual at page-10 maxTiles=5. text='"
                            + safeSnippet(old5.result.getText(), 220) + "'");
            assertTrue(isPictureOnlyCollapse(old6.result.getText()),
                    "Old decoder should collapse at page-10 maxTiles=6. text='"
                            + safeSnippet(old6.result.getText(), 220) + "'");

            assertArrayEquals(old5.result.getTokenIds(), new5.result.getTokenIds(),
                    "GenerationPipeline diverged from old decoder at the last non-collapsed boundary (maxTiles=5).");
            assertEquals(old5.result.getText(), new5.result.getText(),
                    "GenerationPipeline text diverged from old decoder at maxTiles=5.");

            int[] old6Tokens = old6.result.getTokenIds();
            int[] new6Tokens = new6.result.getTokenIds();
            assertArrayEquals(old6Tokens, new6Tokens,
                    "GenerationPipeline should match the old decoder token-for-token at maxTiles=6 after the stop-token fix. "
                            + "oldLen=" + old6Tokens.length + " newLen=" + new6Tokens.length
                            + " oldText='" + safeSnippet(old6.result.getText(), 220) + "'"
                            + " newText='" + safeSnippet(new6.result.getText(), 220) + "'");
            assertEquals(old6.result.getText(), new6.result.getText(),
                    "GenerationPipeline text should match the old decoder at maxTiles=6.");
        } finally {
            maxTiles5.close();
            maxTiles6.close();
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

    private VariantInputs buildVariant(String name, int resizeLongestEdge, boolean useVisionEncoderWrapper) throws Exception {
        return buildVariant(name, 10, resizeLongestEdge, 9, useVisionEncoderWrapper, null);
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, boolean useVisionEncoderWrapper)
            throws Exception {
        return buildVariant(name, pdfPage, resizeLongestEdge, 9, useVisionEncoderWrapper, null);
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, int maxTiles,
                                       boolean useVisionEncoderWrapper)
            throws Exception {
        return buildVariant(name, pdfPage, resizeLongestEdge, maxTiles, useVisionEncoderWrapper, null);
    }

    private VariantInputs buildVariant(String name, int pdfPage, int resizeLongestEdge, int maxTiles,
                                       boolean useVisionEncoderWrapper, GraphExecutionMode visionMode)
            throws Exception {
        BenchmarkConfigApplier.resetModelState(visionEncoderSd);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        if (visionMode != null) {
            visionEncoderSd.setGraphExecutionMode(visionMode);
        }

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

        INDArray visionEmbeddings = useVisionEncoderWrapper
                ? encodeWithVisionEncoderWrapper(imageInput, splitResult)
                : encodeManually(imageInput, splitResult);
        imageInput.close();

        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        int imageSeqLenPerFrame = (int) visionEmbeddings.size(1) / splitResult.getTotalFrames();
        String imagePrompt = ImagePromptBuilder.buildImagePromptString(
                splitResult.numRows, splitResult.numCols, imageSeqLenPerFrame);
        String chatPrompt = "<|im_start|>User:" + imagePrompt
                + "Convert this page to docling.<end_of_utterance>\nAssistant:";
        int[] promptTokenIds = tokenizer.encode(chatPrompt, false).getIds();

        INDArray textEmbeddings = embedPromptDirect(promptTokenIds);
        INDArray inputsEmbeds = EmbeddingMerger.mergeEmbeddings(textEmbeddings, visionEmbeddings, promptTokenIds, imageTokenId);
        textEmbeddings.close();

        log.info("{}: resize={} maxTiles={} wrapper={} image={}x{} frames={} grid={}x{} visionShape={} mergedShape={} promptTokens={}",
                name,
                resizeLongestEdge,
                maxTiles,
                useVisionEncoderWrapper,
                rendered.getWidth(),
                rendered.getHeight(),
                splitResult.getTotalFrames(),
                splitResult.numRows,
                splitResult.numCols,
                Arrays.toString(visionEmbeddings.shape()),
                Arrays.toString(inputsEmbeds.shape()),
                promptTokenIds.length);

        return new VariantInputs(name, resizeLongestEdge, useVisionEncoderWrapper, splitResult.getTotalFrames(),
                splitResult.numRows, splitResult.numCols, promptTokenIds, visionEmbeddings, inputsEmbeds);
    }

    private PreparedVisionInput prepareVisionInput(int pdfPage, int resizeLongestEdge, int maxTiles) throws Exception {
        resetDirectModelState(visionEncoderSd);

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
        return new PreparedVisionInput(pdfPage, resizeLongestEdge, maxTiles, imageInput, splitResult);
    }

    private List<FrameEncodingObservation> encodeVisionFrames(PreparedVisionInput prepared,
                                                              GraphExecutionMode executionMode,
                                                              boolean resetBetweenFrames,
                                                              boolean captureReplayState) {
        List<String> visionInputNames = visionEncoderSd.inputs();
        String[] visionOutputNames = visionEncoderSd.outputs().toArray(new String[0]);
        List<FrameEncodingObservation> observations = new ArrayList<>();

        try {
            visionEncoderSd.setGraphExecutionMode(executionMode);
            if (resetBetweenFrames) {
                resetDirectModelState(visionEncoderSd);
            }

            for (int frameIdx = 0; frameIdx < prepared.splitResult.getTotalFrames(); frameIdx++) {
                if (resetBetweenFrames) {
                    resetDirectModelState(visionEncoderSd);
                    visionEncoderSd.setGraphExecutionMode(executionMode);
                }

                INDArray frameSlice = prepared.imageInput.get(
                        NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                        NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();
                INDArray pixelMask = null;
                Map<String, INDArray> visionOutputs = null;
                try {
                    Map<String, INDArray> visionInputMap = new HashMap<>();
                    for (String inputName : visionInputNames) {
                        if ("pixel_values".equals(inputName)) {
                            visionInputMap.put(inputName, singleFrame);
                        } else if ("pixel_attention_mask".equals(inputName)) {
                            ImageTiler.ContentRegion region = prepared.splitResult.contentRegions.get(frameIdx);
                            pixelMask = ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE);
                            visionInputMap.put(inputName, pixelMask);
                        }
                    }

                    visionOutputs = visionEncoderSd.output(visionInputMap, visionOutputNames);
                    VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
                    INDArray embedding = selected.tensor.dup();
                    FrameReplayState replayState = captureReplayState ? captureVisionReplayState(frameIdx) : null;
                    observations.add(new FrameEncodingObservation(frameIdx, embedding, replayState));
                } finally {
                    closeArrayMap(visionOutputs);
                    closeArray(pixelMask);
                    closeArray(singleFrame);
                }
            }

            return observations;
        } finally {
            resetSameDiff(visionEncoderSd);
        }
    }

    private List<FrameEncodingObservation> executeVisionFrameInputs(GraphExecutionMode executionMode,
                                                                    boolean captureReplayState,
                                                                    VisionFrameInput... inputs) {
        List<FrameEncodingObservation> observations = new ArrayList<>();

        try {
            beginVisionSession(executionMode);
            for (int stepIdx = 0; stepIdx < inputs.length; stepIdx++) {
                observations.add(executeVisionFrameInput(inputs[stepIdx], captureReplayState, stepIdx));
            }
            return observations;
        } finally {
            endVisionSession();
        }
    }

    private void beginVisionSession(GraphExecutionMode executionMode) {
        resetDirectModelState(visionEncoderSd);
        visionEncoderSd.setGraphExecutionMode(executionMode);
    }

    private void endVisionSession() {
        resetSameDiff(visionEncoderSd);
    }

    private FrameEncodingObservation executeVisionFrameInput(VisionFrameInput input,
                                                             boolean captureReplayState,
                                                             int stepIdx) {
        List<String> visionInputNames = visionEncoderSd.inputs();
        String[] visionOutputNames = visionEncoderSd.outputs().toArray(new String[0]);
        Map<String, INDArray> visionInputMap = new HashMap<>();
        Map<String, INDArray> visionOutputs = null;
        String singleFrameBefore = describeVisionInputArrayState(input.singleFrame);
        String pixelMaskBefore = describeVisionInputArrayState(input.pixelMask);
        try {
            for (String inputName : visionInputNames) {
                if ("pixel_values".equals(inputName)) {
                    visionInputMap.put(inputName, input.singleFrame);
                } else if ("pixel_attention_mask".equals(inputName)) {
                    visionInputMap.put(inputName, input.pixelMask);
                }
            }

            visionOutputs = visionEncoderSd.output(visionInputMap, visionOutputNames);
            VisionEncoderUtils.VisionOutput selected = VisionEncoderUtils.selectVisionOutput(visionOutputs);
            INDArray embedding = selected.tensor.dup();
            FrameReplayState replayState = captureReplayState ? captureVisionReplayState(stepIdx) : null;
            return new FrameEncodingObservation(input.frameIdx, embedding, replayState);
        } catch (RuntimeException e) {
            String singleFrameAfter = describeVisionInputArrayState(input.singleFrame);
            String pixelMaskAfter = describeVisionInputArrayState(input.pixelMask);
            throw new RuntimeException(
                    "Vision frame execution failed for label=" + input.label
                            + " frameIdx=" + input.frameIdx
                            + " stepIdx=" + stepIdx
                            + " singleFrameBefore={" + singleFrameBefore + "}"
                            + " singleFrameAfter={" + singleFrameAfter + "}"
                            + " pixelMaskBefore={" + pixelMaskBefore + "}"
                            + " pixelMaskAfter={" + pixelMaskAfter + "}",
                    e);
        } finally {
            closeArrayMap(visionOutputs);
        }
    }

    private VisionFrameInput prepareVisionFrameInput(PreparedVisionInput prepared, int frameIdx, String label) {
        INDArray frameSlice = prepared.imageInput.get(
                NDArrayIndex.point(0), NDArrayIndex.point(frameIdx),
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray singleFrame = frameSlice.reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE).dup();
        ImageTiler.ContentRegion region = prepared.splitResult.contentRegions.get(frameIdx);
        INDArray pixelMask = ImageTiler.createPixelAttentionMask(region.width, region.height, TARGET_SIZE);
        return new VisionFrameInput(frameIdx, label, singleFrame, pixelMask);
    }

    private String describeVisionRuntimeSlotNeighborhood(int... slotIndices) {
        DynamicShapePlanExecutor executor = visionEncoderSd.getOrCreateSession().getDynamicShapePlanExecutor();
        if (executor == null) {
            return "Vision executor unavailable";
        }

        DynamicShapePlan plan = executor.getCurrentPlan();
        if (plan == null || plan.getSlots() == null) {
            return "Vision plan unavailable";
        }

        INDArray[] outputSlots = getExecutorOutputSlots(executor);
        StringBuilder sb = new StringBuilder();
        for (int slotIdx : slotIndices) {
            if (sb.length() > 0) {
                sb.append('\n');
            }
            sb.append(describeVisionRuntimeSlot(plan, executor, outputSlots, slotIdx));
        }
        return sb.toString();
    }

    private String describeVisionRuntimeSlot(DynamicShapePlan plan,
                                             DynamicShapePlanExecutor executor,
                                             INDArray[] outputSlots,
                                             int slotIdx) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slotIdx < 0 || slotIdx >= slots.length) {
            return "slot[" + slotIdx + "] out-of-range";
        }

        DynamicShapeSlot slot = slots[slotIdx];
        SlotState state = executor.getSlotState(slotIdx);
        INDArray array = outputSlots != null && slotIdx < outputSlots.length ? outputSlots[slotIdx] : null;
        String arrayDesc = array == null ? "null" : describeVisionInputArrayState(array);

        return "slot[" + slotIdx + "] "
                + slot
                + " nativeState=" + state
                + " cachedOut=" + arrayDesc;
    }

    private INDArray[] getExecutorOutputSlots(DynamicShapePlanExecutor executor) {
        try {
            Field field = DynamicShapePlanExecutor.class.getDeclaredField("outputSlots");
            field.setAccessible(true);
            return (INDArray[]) field.get(executor);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Unable to access DynamicShapePlanExecutor.outputSlots", e);
        }
    }

    private String summarizeThrowableMessages(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable current = t;
        while (current != null) {
            if (current.getMessage() != null && !current.getMessage().isBlank()) {
                if (sb.length() > 0) {
                    sb.append(" | caused by: ");
                }
                sb.append(current.getMessage());
            }
            current = current.getCause();
        }
        return sb.toString();
    }

    private FrameReplayState captureVisionReplayState(int frameIdx) {
        try {
            DspDebugger debugger = DspDebugger.attach(visionEncoderSd);
            DspDebugger.PlanReport plan = debugger.analyzePlan();
            DspDebugger.GraphReplayReport replay = debugger.analyzeGraphReplay();
            int totalReplayCount = 0;
            int replayingSegments = 0;
            int captureFailures = 0;
            for (DspDebugger.SegmentReplayInfo segment : replay.segments) {
                totalReplayCount += segment.replayCount;
                if (segment.isReplaying()) {
                    replayingSegments++;
                }
                if (segment.hasCaptureFailure()) {
                    captureFailures++;
                }
            }
            return new FrameReplayState(
                    frameIdx,
                    plan.planPhase,
                    replay.pointersStable,
                    replay.frozenExecutionCount,
                    totalReplayCount,
                    replayingSegments,
                    captureFailures
            );
        } catch (Throwable t) {
            log.info("Vision replay state unavailable at frame {}: {}", frameIdx, t.getMessage());
            return new FrameReplayState(frameIdx, null, false, -1, -1, -1, -1);
        }
    }

    private VariantInputs zeroVisualTokenEmbeddings(String name, VariantInputs base) {
        int imageTokenId = ImagePromptBuilder.resolveImageTokenId(tokenizer);
        INDArray zeroedInputs = base.inputsEmbeds.dup();
        int visualTokenCount = 0;
        for (int i = 0; i < base.promptTokenIds.length; i++) {
            if (base.promptTokenIds[i] == imageTokenId) {
                zeroedInputs.get(NDArrayIndex.point(0), NDArrayIndex.point(i), NDArrayIndex.all()).assign(0.0);
                visualTokenCount++;
            }
        }

        assertTrue(visualTokenCount > 0,
                "Expected at least one image token in page-10 prompt before zeroing visual-token rows.");
        log.info("{}: zeroed {} visual-token rows out of {} prompt positions", name, visualTokenCount,
                base.promptTokenIds.length);

        return new VariantInputs(name, base.resizeLongestEdge, base.useVisionEncoderWrapper,
                base.totalFrames, base.numRows, base.numCols, base.promptTokenIds, null, zeroedInputs);
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
            return renderer.renderImageWithDPI(pdfPage, renderDpi, ImageType.RGB);
        }
    }

    private INDArray encodeWithVisionEncoderWrapper(INDArray imageInput, ImageTiler.SplitImageResult splitResult) {
        VisionEncoder visionEncoder = VisionEncoder.builder()
                .model(visionEncoderSd)
                .targetSize(TARGET_SIZE)
                .maxTiles(9)
                .build();
        try {
            VisionEncoder.Result result = visionEncoder.encode(imageInput, splitResult.getTotalFrames(), splitResult);
            return result.getEmbeddings().dup();
        } finally {
            resetSameDiff(visionEncoderSd);
        }
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
                        if (arr != null && arr.closeable() && !arr.wasClosed()) {
                            arr.close();
                        }
                    }
                    singleFrame.close();
                    for (INDArray arr : visionInputMap.values()) {
                        if (arr != null && arr.closeable() && !arr.wasClosed()) {
                            arr.close();
                        }
                    }
                }
            }

            return frameEmbeddings.size() == 1
                    ? frameEmbeddings.get(0).dup()
                    : Nd4j.concat(1, frameEmbeddings.toArray(new INDArray[0]));
        } finally {
            for (INDArray arr : frameEmbeddings) {
                if (arr != null && arr.closeable() && !arr.wasClosed()) {
                    arr.close();
                }
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
            inputIds.close();
            for (INDArray arr : embedOutputs.values()) {
                if (arr != null && arr.closeable() && !arr.wasClosed()) {
                    arr.close();
                }
            }
            resetSameDiff(embedTokens);
        }
    }

    private DecodeSnapshot runOldDecoderSlotBySlot(VariantInputs variant, int maxTokens) throws Exception {
        return runOldDecoderWithConfig(variant, maxTokens, true, true);
    }

    private DecodeSnapshot runGenerationPipelineDirect(VariantInputs variant, int maxTokens) throws Exception {
        resetDirectModelState(decoder);
        resetDirectModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        GenerationPipeline pipeline = GenerationPipeline.create(GenerationPipelineConfig.builder()
                .decoder(decoder)
                .embedTokens(embedTokens)
                .tokenizer(tokenizer)
                .ioConfig(ioConfig)
                .samplingConfig(SamplingConfig.greedy())
                .maxNewTokens(maxTokens)
                .hiddenSize(variant.inputsEmbeds.size(2))
                .build());
        try {
            long startNs = System.nanoTime();
            GenerationResult result = pipeline.generate(variant.inputsEmbeds.dup(), variant.promptTokenIds, maxTokens);
            long elapsedNs = System.nanoTime() - startNs;
            return new DecodeSnapshot(result, elapsedNs);
        } finally {
            pipeline.close();
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

    private DecodeSnapshot runOldDecoderWithConfig(VariantInputs variant, int maxTokens,
                                                   boolean applyConfig, boolean compileModels) throws Exception {
        BenchmarkConfig config = BenchmarkConfig.create("SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT)
                .maxTokens(maxTokens)
                .minDiversityPct(0);
        BenchmarkConfigApplier.resetModelState(decoder);
        BenchmarkConfigApplier.resetModelState(embedTokens);
        if (applyConfig) {
            BenchmarkConfigApplier.apply(config);
        }
        if (compileModels) {
            BenchmarkConfigApplier.compileModels(decoder, "decoder", embedTokens, "embed_tokens", config);
        }

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

    private WarmupSignals capturePrefillAndWarmupTokensPadded(VariantInputs variant, int maxNewTokens) throws Exception {
        resetDirectModelState(decoder);
        resetDirectModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames() != null
                ? ioConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = variant.promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        long hiddenSize = variant.inputsEmbeds.size(2);
        List<String> decoderInputNames = decoder.inputs();
        boolean dspActive = true;
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> staticKvBuffers = new HashMap<>();

        INDArray prefillEmbeddings = variant.inputsEmbeds.dup();
        INDArray prefillInputIds = Nd4j.createFromArray(variant.promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        INDArray warmupEmbeddings = null;
        INDArray warmupInputIds = null;
        Map<String, INDArray> prefillOutputs = null;
        Map<String, INDArray> warmupOutputs = null;
        try {
            Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, decoderInputNames, decoder,
                    prefillEmbeddings, prefillInputIds,
                    0, prefillSeqLen,
                    null, maxKvLen, 0,
                    false, hiddenSize,
                    reusableInputs, dspActive,
                    null, null);
            prefillOutputs = decoder.output(prefillInputMap, allOutputNames.toArray(new String[0]));

            int prefillTokenId = sampleLastToken(prefillOutputs.get(ioConfig.getLogitsOutputName()));
            for (String keyName : kvNames.keyNames) {
                INDArray presentKv = prefillOutputs.get(keyName);
                staticKvBuffers.put(ioConfig.presentToInputName(keyName), padKvToStaticSize(presentKv, maxKvLen));
            }
            for (String valName : kvNames.valueNames) {
                INDArray presentKv = prefillOutputs.get(valName);
                staticKvBuffers.put(ioConfig.presentToInputName(valName), padKvToStaticSize(presentKv, maxKvLen));
            }

            warmupEmbeddings = embedPromptDirect(new int[]{prefillTokenId});
            warmupInputIds = Nd4j.createFromArray(new int[]{prefillTokenId})
                    .reshape(1, 1).castTo(DataType.INT64);
            Map<String, INDArray> warmupInputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, decoderInputNames, decoder,
                    warmupEmbeddings, warmupInputIds,
                    prefillSeqLen, 1,
                    staticKvBuffers, maxKvLen, prefillSeqLen,
                    true, hiddenSize,
                    reusableInputs, dspActive,
                    null, null);
            warmupOutputs = decoder.output(warmupInputMap, allOutputNames.toArray(new String[0]));
            int warmupTokenId = sampleWarmupToken(warmupOutputs.get(ioConfig.getLogitsOutputName()));

            WarmupSignals signals = new WarmupSignals(variant.name, variant.promptTokenIds.length,
                    prefillTokenId, warmupTokenId);
            log.info("{} warmup signals: {}", variant.name, signals.describe(tokenizer));
            return signals;
        } finally {
            closeArrayMap(warmupOutputs);
            closeArrayMap(prefillOutputs);
            for (INDArray kv : staticKvBuffers.values()) {
                closeArray(kv);
            }
            closeArray(warmupInputIds);
            closeArray(warmupEmbeddings);
            closeArray(prefillInputIds);
            closeArray(prefillEmbeddings);
        }
    }

    private static void assertVariantInputParity(VariantInputs expected, VariantInputs actual,
                                                 double maxVisionAbsDiff, double maxMergedAbsDiff) {
        assertEquals(expected.totalFrames, actual.totalFrames,
                "Frame count differs between " + expected.name + " and " + actual.name);
        assertEquals(expected.numRows, actual.numRows,
                "Row count differs between " + expected.name + " and " + actual.name);
        assertEquals(expected.numCols, actual.numCols,
                "Column count differs between " + expected.name + " and " + actual.name);
        assertArrayEquals(expected.promptTokenIds, actual.promptTokenIds,
                "Prompt token IDs differ between " + expected.name + " and " + actual.name);
        assertArrayEquals(expected.visionEmbeddings.shape(), actual.visionEmbeddings.shape(),
                "Vision embedding shapes differ between " + expected.name + " and " + actual.name);
        assertArrayEquals(expected.inputsEmbeds.shape(), actual.inputsEmbeds.shape(),
                "Merged input shapes differ between " + expected.name + " and " + actual.name);

        double visionDiff = maxAbsDiff(expected.visionEmbeddings, actual.visionEmbeddings);
        double mergedDiff = maxAbsDiff(expected.inputsEmbeds, actual.inputsEmbeds);
        log.info("Input parity {} vs {}: visionMaxAbsDiff={} mergedMaxAbsDiff={}",
                expected.name, actual.name, visionDiff, mergedDiff);

        assertTrue(visionDiff <= maxVisionAbsDiff,
                "Vision embeddings differ between " + expected.name + " and " + actual.name
                        + " maxAbsDiff=" + visionDiff);
        assertTrue(mergedDiff <= maxMergedAbsDiff,
                "Merged inputs differ between " + expected.name + " and " + actual.name
                        + " maxAbsDiff=" + mergedDiff);
    }

    private static double maxAbsDiff(INDArray a, INDArray b) {
        INDArray diff = Transforms.abs(a.castTo(DataType.FLOAT).sub(b.castTo(DataType.FLOAT)), false);
        try {
            return diff.maxNumber().doubleValue();
        } finally {
            diff.close();
        }
    }

    private static String summarizeVisionPerFrameDiff(VariantInputs left, VariantInputs right) {
        if (left.visionEmbeddings == null || right.visionEmbeddings == null) {
            return "<visionEmbeddings unavailable>";
        }
        if (left.totalFrames != right.totalFrames) {
            return "frameCountMismatch left=" + left.totalFrames + " right=" + right.totalFrames;
        }

        int rowsPerFrame = (int) (left.visionEmbeddings.size(1) / left.totalFrames);
        StringBuilder sb = new StringBuilder();
        for (int frameIdx = 0; frameIdx < left.totalFrames; frameIdx++) {
            int start = frameIdx * rowsPerFrame;
            int end = start + rowsPerFrame;
            INDArray leftFrame = left.visionEmbeddings.get(
                    NDArrayIndex.point(0), NDArrayIndex.interval(start, end), NDArrayIndex.all()).dup();
            INDArray rightFrame = right.visionEmbeddings.get(
                    NDArrayIndex.point(0), NDArrayIndex.interval(start, end), NDArrayIndex.all()).dup();
            try {
                double diff = maxAbsDiff(leftFrame, rightFrame);
                if (frameIdx > 0) {
                    sb.append(" | ");
                }
                sb.append("f").append(frameIdx).append("=").append(diff);
            } finally {
                leftFrame.close();
                rightFrame.close();
            }
        }
        return sb.toString();
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

    private static void logDecode(VariantInputs variant, DecodeSnapshot snapshot) {
        log.info("{} decode: tokens={} elapsedMs={} text='{}'",
                variant.name,
                snapshot.result.getGeneratedTokenCount(),
                snapshot.elapsedNs / 1_000_000,
                safeSnippet(snapshot.result.getText(), 220));
        log.info("{} tokens={}", variant.name, Arrays.toString(snapshot.result.getTokenIds()));
    }

    private static int findFirstDivergentToken(int[] left, int[] right) {
        int minLen = Math.min(left.length, right.length);
        for (int i = 0; i < minLen; i++) {
            if (left[i] != right[i]) {
                return i;
            }
        }
        return left.length == right.length ? -1 : minLen;
    }

    private static int sampleLastToken(INDArray logits) {
        return Nd4j.argMax(
                logits.get(NDArrayIndex.point(0), NDArrayIndex.point(logits.shape()[1] - 1), NDArrayIndex.all()).dup(),
                -1).getInt(0);
    }

    private static int sampleWarmupToken(INDArray logits) {
        return Nd4j.argMax(
                logits.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup(),
                -1).getInt(0);
    }

    private static INDArray padKvToStaticSize(INDArray presentKv, long maxKvLen) {
        long[] shape = presentKv.shape();
        long batch = shape[0];
        long heads = shape[1];
        long seqLen = shape[2];
        long dim = shape[3];
        if (seqLen >= maxKvLen) {
            return presentKv.dup();
        }
        INDArray padding = Nd4j.zeros(presentKv.dataType(), batch, heads, maxKvLen - seqLen, dim);
        INDArray result = Nd4j.concat(2, presentKv, padding);
        padding.close();
        return result;
    }

    private static void closeArrayMap(Map<String, INDArray> arrays) {
        if (arrays == null) {
            return;
        }
        for (INDArray arr : arrays.values()) {
            closeArray(arr);
        }
    }

    private static void closeFrameObservations(List<FrameEncodingObservation> observations) {
        if (observations == null) {
            return;
        }
        for (FrameEncodingObservation observation : observations) {
            if (observation != null) {
                observation.close();
            }
        }
    }

    private static void closeArray(INDArray arr) {
        if (arr != null && arr.closeable() && !arr.wasClosed()) {
            arr.close();
        }
    }

    private static String describeVisionInputArrayState(INDArray arr) {
        if (arr == null) {
            return "null";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("closed=").append(arr.wasClosed());
        try {
            sb.append(",dtype=").append(arr.dataType());
            sb.append(",len=").append(arr.length());
            sb.append(",rank=").append(arr.rank());
            sb.append(",shape=").append(Arrays.toString(arr.shape()));
            sb.append(",stride=").append(Arrays.toString(arr.stride()));
            sb.append(",order=").append(arr.ordering());
            sb.append(",view=").append(arr.isView());
            sb.append(",attached=").append(arr.isAttached());
        } catch (Throwable t) {
            sb.append(",metaErr=").append(t.getClass().getSimpleName()).append(':').append(t.getMessage());
        }

        try {
            sb.append(",dataPtr=0x").append(Long.toHexString(arr.data().pointer().address()));
            sb.append(",shapePtr=0x").append(Long.toHexString(arr.shapeInfoDataBuffer().pointer().address()));
        } catch (Throwable t) {
            sb.append(",ptrErr=").append(t.getClass().getSimpleName()).append(':').append(t.getMessage());
        }

        try {
            if (arr.length() > 0) {
                sb.append(",sample0=").append(arr.getDouble(0));
            }
        } catch (Throwable t) {
            sb.append(",sampleErr=").append(t.getClass().getSimpleName()).append(':').append(t.getMessage());
        }

        return sb.toString();
    }

    private static String observationsSummary(List<TileDecodeObservation> observations) {
        StringBuilder sb = new StringBuilder();
        for (TileDecodeObservation observation : observations) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append("tiles=").append(observation.maxTiles)
                    .append(" prompt=").append(observation.promptTokens)
                    .append(" collapse=").append(isPictureOnlyCollapse(observation.text))
                    .append(" textual=").append(hasTextualStructure(observation.text))
                    .append(" text='").append(safeSnippet(observation.text, 80)).append("'");
        }
        return sb.toString();
    }

    private static int findFirstDivergentFrame(List<FrameEncodingObservation> left,
                                               List<FrameEncodingObservation> right,
                                               double tolerance) {
        int limit = Math.min(left.size(), right.size());
        for (int i = 0; i < limit; i++) {
            if (maxAbsDiff(left.get(i).embedding, right.get(i).embedding) > tolerance) {
                return i;
            }
        }
        return left.size() == right.size() ? -1 : limit;
    }

    private static String summarizeFrameDiffs(List<FrameEncodingObservation> left,
                                              List<FrameEncodingObservation> right) {
        StringBuilder sb = new StringBuilder();
        int limit = Math.min(left.size(), right.size());
        for (int i = 0; i < limit; i++) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            double diff = maxAbsDiff(left.get(i).embedding, right.get(i).embedding);
            sb.append("f").append(i).append('=').append(diff);
            if (right.get(i).replayState != null) {
                sb.append(" {").append(right.get(i).replayState.shortSummary()).append('}');
            }
        }
        if (left.size() != right.size()) {
            sb.append(" sizeMismatch=").append(left.size()).append(" vs ").append(right.size());
        }
        return sb.toString();
    }

    private static String summarizeReplayStates(List<FrameEncodingObservation> observations) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < observations.size(); i++) {
            if (sb.length() > 0) {
                sb.append(" | ");
            }
            sb.append("s").append(i).append(":f").append(observations.get(i).frameIdx).append('=')
                    .append(observations.get(i).describeReplayState());
        }
        return sb.toString();
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

    private static boolean isPictureOnlyCollapse(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.replace('\n', ' ').replace('\r', ' ').trim().toLowerCase();
        return normalized.startsWith("<doctag><picture>")
                || (normalized.contains("<picture>") && normalized.contains("<end_of_utterance>"));
    }

    private static boolean hasTextualStructure(String text) {
        if (text == null) {
            return false;
        }
        String normalized = text.toLowerCase();
        return normalized.contains("<text>")
                || normalized.contains("<section_header")
                || normalized.contains("<page>");
    }

    private static final class VariantInputs implements AutoCloseable {
        private final String name;
        private final int resizeLongestEdge;
        private final boolean useVisionEncoderWrapper;
        private final int totalFrames;
        private final int numRows;
        private final int numCols;
        private final int[] promptTokenIds;
        private final INDArray visionEmbeddings;
        private final INDArray inputsEmbeds;

        private VariantInputs(String name, int resizeLongestEdge, boolean useVisionEncoderWrapper,
                              int totalFrames, int numRows, int numCols, int[] promptTokenIds,
                              INDArray visionEmbeddings, INDArray inputsEmbeds) {
            this.name = name;
            this.resizeLongestEdge = resizeLongestEdge;
            this.useVisionEncoderWrapper = useVisionEncoderWrapper;
            this.totalFrames = totalFrames;
            this.numRows = numRows;
            this.numCols = numCols;
            this.promptTokenIds = promptTokenIds;
            this.visionEmbeddings = visionEmbeddings;
            this.inputsEmbeds = inputsEmbeds;
        }

        private VariantInputs withName(String newName) {
            return new VariantInputs(newName, resizeLongestEdge, useVisionEncoderWrapper,
                    totalFrames, numRows, numCols, promptTokenIds, visionEmbeddings, inputsEmbeds);
        }

        @Override
        public void close() {
            if (inputsEmbeds != null && inputsEmbeds.closeable() && !inputsEmbeds.wasClosed()) {
                inputsEmbeds.close();
            }
            if (visionEmbeddings != null && visionEmbeddings.closeable() && !visionEmbeddings.wasClosed()) {
                visionEmbeddings.close();
            }
        }
    }

    private static final class PreparedVisionInput implements AutoCloseable {
        private final int pdfPage;
        private final int resizeLongestEdge;
        private final int maxTiles;
        private final INDArray imageInput;
        private final ImageTiler.SplitImageResult splitResult;

        private PreparedVisionInput(int pdfPage, int resizeLongestEdge, int maxTiles,
                                    INDArray imageInput, ImageTiler.SplitImageResult splitResult) {
            this.pdfPage = pdfPage;
            this.resizeLongestEdge = resizeLongestEdge;
            this.maxTiles = maxTiles;
            this.imageInput = imageInput;
            this.splitResult = splitResult;
        }

        @Override
        public void close() {
            closeArray(imageInput);
        }
    }

    private static final class FrameEncodingObservation implements AutoCloseable {
        private final int frameIdx;
        private final INDArray embedding;
        private final FrameReplayState replayState;

        private FrameEncodingObservation(int frameIdx, INDArray embedding, FrameReplayState replayState) {
            this.frameIdx = frameIdx;
            this.embedding = embedding;
            this.replayState = replayState;
        }

        private String describeReplayState() {
            return replayState == null ? "no-replay-state" : replayState.shortSummary();
        }

        @Override
        public void close() {
            closeArray(embedding);
        }
    }

    private static final class VisionFrameInput implements AutoCloseable {
        private final int frameIdx;
        private final String label;
        private final INDArray singleFrame;
        private final INDArray pixelMask;

        private VisionFrameInput(int frameIdx, String label, INDArray singleFrame, INDArray pixelMask) {
            this.frameIdx = frameIdx;
            this.label = label;
            this.singleFrame = singleFrame;
            this.pixelMask = pixelMask;
        }

        @Override
        public void close() {
            closeArray(singleFrame);
            closeArray(pixelMask);
        }
    }

    private static final class FrameReplayState {
        private final int frameIdx;
        private final PlanPhase planPhase;
        private final boolean pointersStable;
        private final int frozenExecutionCount;
        private final int totalReplayCount;
        private final int replayingSegments;
        private final int captureFailures;

        private FrameReplayState(int frameIdx,
                                 PlanPhase planPhase,
                                 boolean pointersStable,
                                 int frozenExecutionCount,
                                 int totalReplayCount,
                                 int replayingSegments,
                                 int captureFailures) {
            this.frameIdx = frameIdx;
            this.planPhase = planPhase;
            this.pointersStable = pointersStable;
            this.frozenExecutionCount = frozenExecutionCount;
            this.totalReplayCount = totalReplayCount;
            this.replayingSegments = replayingSegments;
            this.captureFailures = captureFailures;
        }

        private String shortSummary() {
            return "phase=" + planPhase
                    + ",stable=" + pointersStable
                    + ",frozenExec=" + frozenExecutionCount
                    + ",replays=" + totalReplayCount
                    + ",replayingSegs=" + replayingSegments
                    + ",captureFailures=" + captureFailures;
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

    private static final class TileDecodeObservation {
        private final int maxTiles;
        private final int promptTokens;
        private final String text;
        private final int[] tokenIds;

        private TileDecodeObservation(int maxTiles, int promptTokens, String text, int[] tokenIds) {
            this.maxTiles = maxTiles;
            this.promptTokens = promptTokens;
            this.text = text;
            this.tokenIds = tokenIds;
        }
    }

    private static final class WarmupSignals {
        private final String name;
        private final int promptTokens;
        private final int prefillTokenId;
        private final int warmupTokenId;

        private WarmupSignals(String name, int promptTokens, int prefillTokenId, int warmupTokenId) {
            this.name = name;
            this.promptTokens = promptTokens;
            this.prefillTokenId = prefillTokenId;
            this.warmupTokenId = warmupTokenId;
        }

        private String describe(Tokenizer tokenizer) {
            return "name=" + name
                    + " promptTokens=" + promptTokens
                    + " prefill=" + prefillTokenId + " ('" + tokenizer.decode(new int[]{prefillTokenId}, false) + "')"
                    + " warmup=" + warmupTokenId + " ('" + tokenizer.decode(new int[]{warmupTokenId}, false) + "')";
        }
    }

    private static final class HandoffState {
        private final int prefillTokenId;
        private final int warmupTokenId;
        private final long kvCacheFingerprint;
        private final int[] firstDecodeLogitsTop3;

        private HandoffState(int prefillTokenId, int warmupTokenId, long kvCacheFingerprint, int[] firstDecodeLogitsTop3) {
            this.prefillTokenId = prefillTokenId;
            this.warmupTokenId = warmupTokenId;
            this.kvCacheFingerprint = kvCacheFingerprint;
            this.firstDecodeLogitsTop3 = firstDecodeLogitsTop3;
        }
    }

    private HandoffState captureHandoffState(VariantInputs variant, int maxNewTokens) throws Exception {
        resetDirectModelState(decoder);
        resetDirectModelState(embedTokens);

        ModelIOConfig ioConfig = ModelIOConfig.discover(decoder);
        ModelIOConfig.KVCacheNames kvNames = ioConfig.getKvCacheNames() != null
                ? ioConfig.getKvCacheNames()
                : ModelIOConfig.findKVCacheOutputNames(decoder);

        List<String> allOutputNames = new ArrayList<>();
        allOutputNames.add(ioConfig.getLogitsOutputName());
        allOutputNames.addAll(kvNames.keyNames);
        allOutputNames.addAll(kvNames.valueNames);

        int prefillSeqLen = variant.promptTokenIds.length;
        long maxKvLen = prefillSeqLen + maxNewTokens;
        long hiddenSize = variant.inputsEmbeds.size(2);
        List<String> decoderInputNames = decoder.inputs();
        boolean dspActive = true;
        Map<String, INDArray> reusableInputs = new HashMap<>();
        Map<String, INDArray> staticKvBuffers = new HashMap<>();

        INDArray prefillEmbeddings = variant.inputsEmbeds.dup();
        INDArray prefillInputIds = Nd4j.createFromArray(variant.promptTokenIds)
                .reshape(1, prefillSeqLen).castTo(DataType.INT64);
        INDArray warmupEmbeddings = null;
        INDArray warmupInputIds = null;
        Map<String, INDArray> prefillOutputs = null;
        Map<String, INDArray> warmupOutputs = null;
        try {
            Map<String, INDArray> prefillInputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, decoderInputNames, decoder,
                    prefillEmbeddings, prefillInputIds,
                    0, prefillSeqLen,
                    null, maxKvLen, 0,
                    false, hiddenSize,
                    reusableInputs, dspActive,
                    null, null);
            prefillOutputs = decoder.output(prefillInputMap, allOutputNames.toArray(new String[0]));

            int prefillTokenId = sampleLastToken(prefillOutputs.get(ioConfig.getLogitsOutputName()));
            for (String keyName : kvNames.keyNames) {
                INDArray presentKv = prefillOutputs.get(keyName);
                staticKvBuffers.put(ioConfig.presentToInputName(keyName), padKvToStaticSize(presentKv, maxKvLen));
            }
            for (String valName : kvNames.valueNames) {
                INDArray presentKv = prefillOutputs.get(valName);
                staticKvBuffers.put(ioConfig.presentToInputName(valName), padKvToStaticSize(presentKv, maxKvLen));
            }

            warmupEmbeddings = embedPromptDirect(new int[]{prefillTokenId});
            warmupInputIds = Nd4j.createFromArray(new int[]{prefillTokenId})
                    .reshape(1, 1).castTo(DataType.INT64);
            Map<String, INDArray> warmupInputMap = DecoderInputBuilder.buildDecoderInputMap(
                    ioConfig, decoderInputNames, decoder,
                    warmupEmbeddings, warmupInputIds,
                    prefillSeqLen, 1,
                    staticKvBuffers, maxKvLen, prefillSeqLen,
                    true, hiddenSize,
                    reusableInputs, dspActive,
                    null, null);
            warmupOutputs = decoder.output(warmupInputMap, allOutputNames.toArray(new String[0]));
            int warmupTokenId = sampleWarmupToken(warmupOutputs.get(ioConfig.getLogitsOutputName()));

            long kvFingerprint = computeKvCacheFingerprint(staticKvBuffers);

            INDArray firstDecodeLogits = warmupOutputs.get(ioConfig.getLogitsOutputName());
            int[] top3 = topKTokenIds(firstDecodeLogits, 3);

            return new HandoffState(prefillTokenId, warmupTokenId, kvFingerprint, top3);
        } finally {
            closeArrayMap(warmupOutputs);
            closeArrayMap(prefillOutputs);
            for (INDArray kv : staticKvBuffers.values()) {
                closeArray(kv);
            }
            closeArray(warmupInputIds);
            closeArray(warmupEmbeddings);
            closeArray(prefillInputIds);
            closeArray(prefillEmbeddings);
        }
    }

    private static long computeKvCacheFingerprint(Map<String, INDArray> kvBuffers) {
        long hash = 0;
        for (Map.Entry<String, INDArray> entry : kvBuffers.entrySet()) {
            INDArray arr = entry.getValue();
            if (arr == null) continue;
            INDArray summary = arr.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.interval(0, Math.min(4, arr.size(2))), NDArrayIndex.interval(0, Math.min(8, arr.size(3))));
            try {
                double[] flat = summary.data().asDouble();
                for (double v : flat) {
                    long bits = Double.doubleToLongBits(v);
                    hash = hash * 31 + (bits ^ (bits >>> 32));
                }
            } finally {
                summary.close();
            }
        }
        return hash;
    }

    private static int[] topKTokenIds(INDArray logits, int k) {
        INDArray lastLogits = logits.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all()).dup();
        try {
            INDArray[] sortedWithIndices = Nd4j.sortWithIndices(lastLogits.dup(), 0, false);
            INDArray indices = sortedWithIndices[1];
            int[] result = new int[k];
            for (int i = 0; i < k; i++) {
                result[i] = indices.getInt(i);
            }
            sortedWithIndices[0].close();
            indices.close();
            return result;
        } finally {
            lastLogits.close();
        }
    }
}
