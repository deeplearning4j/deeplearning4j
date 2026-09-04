/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.ListenerVariables;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce.same.AMax;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * Produces finalized Tensor G3 Q4_K activation-boundary calibration from real
 * inference executions of the exact canonical SameDiff graph.
 *
 * <p>The producer is deliberately backend-neutral. The caller owns prompt
 * rendering and generation; this class owns exact Q4 op discovery, listener
 * lifecycle, deterministic dataset identity, per-sample completeness, and
 * conservative INT8 scale calculation. Only scalar maxima are retained.</p>
 */
public final class SdxTensorG3Q4Calibration {
    public static final String CALIBRATION_ABI =
            "sdx.tensor-g3.q4.minmax-power2-headroom-v1";
    public static final int REQUIRED_SAMPLE_COUNT = 32;

    private static final double ENVELOPE_HEADROOM = 2.0;
    private static final String OBSERVER_PREFIX = "__sdx_tensor_g3_q4_calibration_";
    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
    private static final List<String> CALIBRATION_PROMPTS = List.of(
            "Hello. Give one concise helpful response.",
            "Summarize why deterministic builds matter.",
            "Explain a graph edge and a graph node in plain language.",
            "List three safe steps for diagnosing a failed deployment.",
            "Compare CPU and accelerator execution without inventing measurements.",
            "Return the integers 3, 5, 8, 13, and 21 in ascending order.",
            "What is 17 multiplied by 23? Show the result only.",
            "Rewrite this sentence clearly: systems should fail closed on invalid artifacts.",
            "Describe a cache key using source, target, compiler, and configuration.",
            "Give a two-line JSON example with a name and a numeric value.",
            "Explain why a checksum is not the same as runtime qualification.",
            "State one benefit and one risk of static quantization.",
            "Translate 'good morning' into Japanese and French.",
            "Read these Unicode samples: café, 日本語, 안녕하세요, مرحبا.",
            "Name four common tensor data types.",
            "Explain the difference between shape and stride.",
            "What should happen when required model metadata is missing?",
            "Provide a short SQL query selecting active rows ordered by id.",
            "Provide a short Python function that adds two numbers.",
            "Provide a short Java method that returns the larger integer.",
            "Summarize the sequence import, calibrate, compile, verify.",
            "Explain why calibration must use the exact canonical model.",
            "Describe a cold-cache run and a warm-cache run.",
            "Give a concise definition of reproducibility.",
            "Which is larger: 0.125 or 0.0625?",
            "Write a sentence containing a date, a percentage, and a currency value.",
            "Explain why finite numerical output can still be incorrect.",
            "Describe how to compare an accelerator result with a desktop baseline.",
            "Name three failure signals that should stop artifact promotion.",
            "Explain why application code should not parse provider-private artifacts.",
            "Summarize a safe compiler-owned calibration lifecycle.",
            "Conclude with one short sentence about verified execution."
    );

    private SdxTensorG3Q4Calibration() {
    }

    /** Ordered raw records used by the internal calibration runner. */
    public static List<String> calibrationPrompts() {
        return CALIBRATION_PROMPTS;
    }

    /** Return whether the graph contains at least one well-formed Q4_K qmatmul. */
    public static boolean requiresCalibration(SameDiff graph) throws IOException {
        return !q4Operations(graph).isEmpty();
    }

    /**
     * Execute the ordered raw prompt dataset and collect one complete
     * observation for every Q4 operation per prompt.
     */
    public static Result calibrate(
            SameDiff graph,
            String tokenizerAssetSha256,
            List<String> prompts,
            SampleExecutor executor) throws IOException {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(prompts, "prompts");
        Objects.requireNonNull(executor, "executor");
        if (prompts.size() != REQUIRED_SAMPLE_COUNT) {
            throw new IOException("Tensor G3 Q4 calibration requires exactly "
                    + REQUIRED_SAMPLE_COUNT + " ordered prompt samples");
        }
        String datasetSha256 = datasetSha256(tokenizerAssetSha256, prompts);
        List<OperationSpec> operations = q4Operations(graph);
        if (operations.isEmpty()) {
            return new Result(0, datasetSha256, Collections.emptyMap());
        }

        CalibrationCollector collector = new CalibrationCollector(graph, operations);
        List<Listener> previousListeners = new ArrayList<>(graph.getListeners());
        List<Listener> calibrationListeners = new ArrayList<>(previousListeners);
        calibrationListeners.add(collector);
        graph.setListeners(calibrationListeners);
        try {
            for (int sampleIndex = 0; sampleIndex < prompts.size(); sampleIndex++) {
                collector.beginSample(sampleIndex);
                try {
                    executor.execute(prompts.get(sampleIndex));
                } catch (Exception failure) {
                    throw new IOException("Tensor G3 Q4 calibration sample " + sampleIndex
                            + " failed", failure);
                }
                collector.endSample();
            }
            return collector.result(datasetSha256);
        } catch (IllegalStateException invalidObservation) {
            throw new IOException(invalidObservation.getMessage(), invalidObservation);
        } finally {
            graph.setListeners(previousListeners);
        }
    }

    /** Deterministic identity of the exact raw prompt dataset and tokenizer assets. */
    public static String datasetSha256(
            String tokenizerAssetSha256, List<String> prompts) throws IOException {
        if (tokenizerAssetSha256 == null || !SHA256.matcher(tokenizerAssetSha256).matches()) {
            throw new IOException("tokenizerAssetSha256 must be lowercase SHA-256");
        }
        if (prompts == null || prompts.size() != REQUIRED_SAMPLE_COUNT) {
            throw new IOException("Tensor G3 Q4 calibration dataset must contain exactly "
                    + REQUIRED_SAMPLE_COUNT + " prompt records");
        }
        MessageDigest digest = sha256Digest();
        digest.update((CALIBRATION_ABI + "\0").getBytes(StandardCharsets.UTF_8));
        updateInt(digest, prompts.size());
        digest.update(tokenizerAssetSha256.getBytes(StandardCharsets.US_ASCII));
        for (int index = 0; index < prompts.size(); index++) {
            String prompt = prompts.get(index);
            if (prompt == null || prompt.isEmpty()) {
                throw new IOException("calibration prompt " + index + " is empty");
            }
            byte[] bytes = prompt.getBytes(StandardCharsets.UTF_8);
            updateInt(digest, index);
            updateInt(digest, bytes.length);
            digest.update(bytes);
        }
        return hex(digest.digest());
    }

    private static List<OperationSpec> q4Operations(SameDiff graph) throws IOException {
        Objects.requireNonNull(graph, "graph");
        List<OperationSpec> result = new ArrayList<>();
        for (SameDiffOp candidate : graph.getOps().values()) {
            if (candidate.getOp() == null
                    || !"ggml_qmatmul".equals(candidate.getOp().opName())) {
                continue;
            }
            if (!(candidate.getOp() instanceof DynamicCustomOp)) {
                throw new IOException("ggml_qmatmul " + candidate.getName()
                        + " does not expose custom-op arguments");
            }
            long[] integerArgs = ((DynamicCustomOp) candidate.getOp()).iArgs();
            if (integerArgs == null || integerArgs.length == 0) {
                throw new IOException("ggml_qmatmul " + candidate.getName()
                        + " has no quantization type argument");
            }
            if (integerArgs[0] != 8L) {
                continue;
            }
            if (integerArgs.length != 4 || integerArgs[1] <= 0L
                    || integerArgs[2] <= 0L || integerArgs[2] % 256L != 0L
                    || (integerArgs[3] != 0L && integerArgs[3] != 1L)) {
                throw new IOException("ggml_qmatmul " + candidate.getName()
                        + " is not a valid Q4_K contract");
            }
            List<String> inputs = candidate.getInputsToOp();
            List<String> outputs = candidate.getOutputsOfOp();
            if (inputs == null || inputs.size() != 2
                    || outputs == null || outputs.size() != 1) {
                throw new IOException("Q4_K operation " + candidate.getName()
                        + " must have exactly two inputs and one output");
            }
            result.add(new OperationSpec(
                    candidate.getName(), inputs.get(0), outputs.get(0)));
        }
        result.sort((left, right) -> left.opName.compareTo(right.opName));
        return result;
    }

    private static float scaleFor(double observedAbsoluteMaximum, int quantizationMaximum) {
        if (!Double.isFinite(observedAbsoluteMaximum) || observedAbsoluteMaximum <= 0.0) {
            throw new IllegalStateException(
                    "calibration absolute maximum must be finite and positive");
        }
        double guarded = observedAbsoluteMaximum * ENVELOPE_HEADROOM;
        if (!Double.isFinite(guarded) || guarded <= 0.0) {
            throw new IllegalStateException("calibration headroom overflowed");
        }
        int exponent = Math.getExponent(guarded);
        double bucket = Math.scalb(1.0, exponent);
        if (bucket < guarded) {
            bucket = Math.scalb(bucket, 1);
        }
        double exactScale = bucket / quantizationMaximum;
        float scale = (float) exactScale;
        while (Float.isFinite(scale)
                && (double) scale * quantizationMaximum < bucket) {
            scale = Math.nextUp(scale);
        }
        if (!Float.isFinite(scale) || scale <= 0.0f) {
            throw new IllegalStateException("calibration scale is not finite and positive");
        }
        return scale;
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static void updateInt(MessageDigest digest, int value) {
        digest.update(ByteBuffer.allocate(Integer.BYTES).putInt(value).array());
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return result.toString();
    }

    @FunctionalInterface
    public interface SampleExecutor {
        void execute(String prompt) throws Exception;
    }

    /** Final immutable scalar calibration for one Q4 operation. */
    public static final class OperatorCalibration {
        private final float activationScale;
        private final float outputScale;
        private final double observedActivationAbsoluteMaximum;
        private final double observedOutputAbsoluteMaximum;

        private OperatorCalibration(
                float activationScale,
                float outputScale,
                double observedActivationAbsoluteMaximum,
                double observedOutputAbsoluteMaximum) {
            this.activationScale = activationScale;
            this.outputScale = outputScale;
            this.observedActivationAbsoluteMaximum = observedActivationAbsoluteMaximum;
            this.observedOutputAbsoluteMaximum = observedOutputAbsoluteMaximum;
        }

        public float activationScale() {
            return activationScale;
        }

        public float outputScale() {
            return outputScale;
        }

        public double observedActivationAbsoluteMaximum() {
            return observedActivationAbsoluteMaximum;
        }

        public double observedOutputAbsoluteMaximum() {
            return observedOutputAbsoluteMaximum;
        }
    }

    /** Final source-independent calibration result; the contract writer adds source identity. */
    public static final class Result {
        private final int sampleCount;
        private final String datasetSha256;
        private final Map<String, OperatorCalibration> operatorCalibrations;

        private Result(
                int sampleCount,
                String datasetSha256,
                Map<String, OperatorCalibration> operatorCalibrations) {
            this.sampleCount = sampleCount;
            this.datasetSha256 = datasetSha256;
            this.operatorCalibrations = Collections.unmodifiableMap(
                    new LinkedHashMap<>(new TreeMap<>(operatorCalibrations)));
        }

        public int sampleCount() {
            return sampleCount;
        }

        public String datasetSha256() {
            return datasetSha256;
        }

        public Map<String, OperatorCalibration> operatorCalibrations() {
            return operatorCalibrations;
        }

        public boolean hasQ4Operations() {
            return !operatorCalibrations.isEmpty();
        }
    }

    private static final class OperationSpec {
        private final String opName;
        private final String activationVariable;
        private final String outputVariable;

        private OperationSpec(
                String opName, String activationVariable, String outputVariable) {
            this.opName = opName;
            this.activationVariable = activationVariable;
            this.outputVariable = outputVariable;
        }
    }

    /** Package-visible for focused platform tests; production uses {@link #calibrate}. */
    static final class CalibrationCollector extends BaseListener {
        private final Map<String, MutableCalibration> calibrations = new TreeMap<>();
        private final Map<String, List<ObservationTarget>> targetsByVariable =
                new LinkedHashMap<>();
        private final Set<String> requiredVariables = new LinkedHashSet<>();
        private int activeSample = -1;
        private int completedSamples;

        private CalibrationCollector(SameDiff graph, List<OperationSpec> operations)
                throws IOException {
            Map<String, String> observerByBoundary = new LinkedHashMap<>();
            for (OperationSpec operation : operations) {
                MutableCalibration calibration = new MutableCalibration(operation.opName);
                calibrations.put(operation.opName, calibration);
                addBoundaryTarget(
                        graph,
                        observerByBoundary,
                        operation.activationVariable,
                        new ObservationTarget(calibration, true));
                addBoundaryTarget(
                        graph,
                        observerByBoundary,
                        operation.outputVariable,
                        new ObservationTarget(calibration, false));
            }
        }

        static CalibrationCollector forGraph(SameDiff graph) throws IOException {
            return new CalibrationCollector(graph, q4Operations(graph));
        }

        private void addBoundaryTarget(
                SameDiff graph,
                Map<String, String> observerByBoundary,
                String boundaryVariable,
                ObservationTarget target) throws IOException {
            // Keep the direct boundary as an opportunistic callback target for standard
            // execution, but do not require it. Requiring every full activation/output made
            // DSP retain hundreds of model-sized tensors until generation returned.
            targetsByVariable.computeIfAbsent(boundaryVariable, ignored -> new ArrayList<>())
                    .add(target);
            String observerVariable = observerByBoundary.get(boundaryVariable);
            if (observerVariable == null) {
                observerVariable = installAbsoluteMaximumObserver(
                        graph, boundaryVariable, observerByBoundary.size());
                observerByBoundary.put(boundaryVariable, observerVariable);
            }
            requiredVariables.add(observerVariable);
            targetsByVariable.computeIfAbsent(observerVariable, ignored -> new ArrayList<>())
                    .add(target);
        }

        private static String installAbsoluteMaximumObserver(
                SameDiff graph, String boundaryVariable, int observerIndex) throws IOException {
            SDVariable boundary = graph.getVariable(boundaryVariable);
            if (boundary == null) {
                throw new IOException("Tensor G3 calibration boundary variable is missing: "
                        + boundaryVariable);
            }
            String baseName = OBSERVER_PREFIX + observerIndex;
            String maximumName = baseName + "_amax";
            if (graph.hasVariable(maximumName)) {
                return maximumName;
            }
            SDVariable maximum = new AMax(graph, boundary).outputVariable();
            return graph.updateVariableNameAndReference(maximum, maximumName, true).name();
        }

        void beginSample(int sampleIndex) {
            if (activeSample >= 0) {
                throw new IllegalStateException("calibration sample " + activeSample
                        + " was not completed");
            }
            activeSample = sampleIndex;
            for (MutableCalibration calibration : calibrations.values()) {
                calibration.activationObservedInSample = false;
                calibration.outputObservedInSample = false;
            }
        }

        void endSample() {
            if (activeSample < 0) {
                throw new IllegalStateException("no calibration sample is active");
            }
            for (MutableCalibration calibration : calibrations.values()) {
                if (!calibration.activationObservedInSample
                        || !calibration.outputObservedInSample) {
                    throw new IllegalStateException("calibration sample " + activeSample
                            + " did not execute both boundaries for Q4 operation "
                            + calibration.opName);
                }
            }
            completedSamples++;
            activeSample = -1;
        }

        Result result(String datasetSha256) {
            if (activeSample >= 0) {
                throw new IllegalStateException("calibration sample " + activeSample
                        + " is still active");
            }
            if (completedSamples < REQUIRED_SAMPLE_COUNT) {
                throw new IllegalStateException("Tensor G3 Q4 calibration completed only "
                        + completedSamples + " samples; at least "
                        + REQUIRED_SAMPLE_COUNT + " are required");
            }
            Map<String, OperatorCalibration> finalized = new TreeMap<>();
            for (MutableCalibration calibration : calibrations.values()) {
                finalized.put(calibration.opName, new OperatorCalibration(
                        scaleFor(calibration.activationAbsoluteMaximum, 127),
                        scaleFor(calibration.outputAbsoluteMaximum, 126),
                        calibration.activationAbsoluteMaximum,
                        calibration.outputAbsoluteMaximum));
            }
            return new Result(completedSamples, datasetSha256, finalized);
        }

        @Override
        public ListenerVariables requiredVariables(SameDiff sameDiff) {
            return ListenerVariables.builder()
                    .inferenceVariables(requiredVariables.toArray(new String[0]))
                    .build();
        }

        @Override
        public boolean isActive(Operation operation) {
            return operation == Operation.INFERENCE;
        }

        @Override
        public boolean requiresAllActivations() {
            return false;
        }

        @Override
        public void activationAvailable(
                SameDiff sameDiff,
                At at,
                MultiDataSet batch,
                SameDiffOp op,
                String variableName,
                INDArray activation) {
            if (activeSample < 0) {
                return;
            }
            List<ObservationTarget> targets = targetsByVariable.get(variableName);
            if (targets == null || targets.isEmpty()) {
                return;
            }
            // Required callbacks are scalar max(abs(boundary)) graph outputs. Direct
            // boundary callbacks remain supported, so amaxNumber keeps both paths exact.
            double absoluteMaximum = activation.amaxNumber().doubleValue();
            if (!Double.isFinite(absoluteMaximum) || absoluteMaximum < 0.0) {
                throw new IllegalStateException("calibration variable " + variableName
                        + " produced a non-finite absolute maximum");
            }
            for (ObservationTarget target : targets) {
                if (target.activation) {
                    target.calibration.activationAbsoluteMaximum = Math.max(
                            target.calibration.activationAbsoluteMaximum, absoluteMaximum);
                    target.calibration.activationObservedInSample = true;
                } else {
                    target.calibration.outputAbsoluteMaximum = Math.max(
                            target.calibration.outputAbsoluteMaximum, absoluteMaximum);
                    target.calibration.outputObservedInSample = true;
                }
            }
        }

        @Override
        public void opExecution(
                SameDiff sameDiff,
                At at,
                MultiDataSet batch,
                SameDiffOp op,
                OpContext opContext,
                INDArray[] outputs) {
            // activationAvailable owns scalar collection so shared producer outputs
            // are measured once even when several Q4 operations consume them.
        }
    }

    private static final class MutableCalibration {
        private final String opName;
        private double activationAbsoluteMaximum;
        private double outputAbsoluteMaximum;
        private boolean activationObservedInSample;
        private boolean outputObservedInSample;

        private MutableCalibration(String opName) {
            this.opName = opName;
        }
    }

    private static final class ObservationTarget {
        private final MutableCalibration calibration;
        private final boolean activation;

        private ObservationTarget(MutableCalibration calibration, boolean activation) {
            this.calibration = calibration;
            this.activation = activation;
        }
    }
}
