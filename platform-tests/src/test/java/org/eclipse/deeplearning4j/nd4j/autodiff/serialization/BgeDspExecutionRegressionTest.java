/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.serialization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapeSlot;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanIntrospection;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * One-off reproducer for the May 29 BGE SameDiff/DSP regression where native
 * execution returns status=0 but last_hidden_state contains non-finite values.
 */
@Slf4j
public class BgeDspExecutionRegressionTest extends BaseND4JTest {

    private static final String MODEL_PATH = System.getProperty("bge.model.path",
            System.getProperty("user.home") + "/.kompile/models/bge-base-en-v1.5/bge-base-en-v1.5.sdz");
    private static final String OUTPUT = "last_hidden_state";
    private static final int SEQ_LEN = 512;
    private static final int BATCH = Integer.getInteger("bge.repro.batch", 1);

    @Override
    public long getTimeoutMilliseconds() {
        return 20 * 60 * 1000L;
    }

    // Prod runs BGE in fp16/float through the GraphOptimizer, NOT BaseND4JTest's DOUBLE
    // default. Double at [32,512] needs ~32 GB and OOMs — that is a test artifact, not the
    // prod circumstance. Use FLOAT for batch>1 (or -Dbge.repro.float=true).
    @Override
    public org.nd4j.linalg.api.buffer.DataType getDataType() {
        return (BATCH > 1 || Boolean.getBoolean("bge.repro.float"))
                ? org.nd4j.linalg.api.buffer.DataType.FLOAT
                : org.nd4j.linalg.api.buffer.DataType.DOUBLE;
    }

    // -Dbge.repro.fp16=true runs the model in half precision (prod uses fp16 → ~16 GB at
    // [32,512], leaving device headroom for CUDA-graph capture instead of filling the card).
    @Override
    public org.nd4j.linalg.api.buffer.DataType getDefaultFPDataType() {
        return Boolean.getBoolean("bge.repro.fp16")
                ? org.nd4j.linalg.api.buffer.DataType.HALF
                : getDataType();
    }

    @Test
    public void testBgeLastHiddenStateRepeatedExecutionMatrix() throws Throwable {
        File modelFile = new File(MODEL_PATH);
        assumeTrue(modelFile.exists(), "BGE SDZ not found at " + modelFile.getAbsolutePath());

        int iterations = Integer.getInteger("bge.repro.iterations", 5);
        String only = System.getProperty("bge.repro.config", "").trim();
        String sequence = System.getProperty("bge.repro.sequence", "").trim();

        List<RunConfig> configs = List.of(
                new RunConfig("dsp-auto-disk-cache", true, GraphExecutionMode.AUTO, true, false, true, false),
                new RunConfig("dsp-auto-force-recompile", true, GraphExecutionMode.AUTO, true, true, true, false),
                new RunConfig("dsp-auto-disk-disabled", true, GraphExecutionMode.AUTO, false, false, true, false),
                new RunConfig("dsp-triton", true, GraphExecutionMode.TRITON, true, false, true, false),
                new RunConfig("dsp-triton-skipped", true, GraphExecutionMode.AUTO, true, false, true, true),
                new RunConfig("dsp-cuda-graphs", true, GraphExecutionMode.CUDA_GRAPHS, true, false, true, false),
                new RunConfig("dsp-cuda-graphs-disabled", true, GraphExecutionMode.AUTO, true, false, false, false),
                new RunConfig("dsp-slot-by-slot", true, GraphExecutionMode.SLOT_BY_SLOT, true, false, false, true),
                new RunConfig("standard-samediff-no-dsp", false, GraphExecutionMode.SLOT_BY_SLOT, false, false, false, true));

        for (RunConfig config : selectConfigs(configs, only, sequence)) {
            try {
                runConfig(modelFile, config, iterations);
            } catch (Throwable t) {
                String message = config.name + ": " + t.getMessage();
                log.error("BGE regression config failed: {}", message, t);
                throw t;
            }
        }
    }

    private static List<RunConfig> selectConfigs(List<RunConfig> configs, String only, String sequence) {
        if (!sequence.isEmpty()) {
            Map<String, RunConfig> byName = configs.stream()
                    .collect(Collectors.toMap(c -> c.name, c -> c, (a, b) -> a, LinkedHashMap::new));
            List<RunConfig> selected = new ArrayList<>();
            for (String name : sequence.split(",")) {
                String trimmed = name.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                RunConfig config = byName.get(trimmed);
                assertNotNull(config, "Unknown BGE repro config in bge.repro.sequence: " + trimmed);
                selected.add(config);
            }
            assertFalse(selected.isEmpty(), "bge.repro.sequence did not select any configs");
            return selected;
        }
        if (!only.isEmpty()) {
            return configs.stream().filter(c -> c.name.equals(only)).collect(Collectors.toList());
        }
        return configs;
    }

    private void runConfig(File modelFile, RunConfig config, int iterations) throws Exception {
        log.info("=== BGE DSP regression config: {} ===", config);

        Map<String, String> previous = captureProperties();
        boolean previousDspEnabled = InferenceSession.isDynamicShapePlanEnabled();
        InferenceSession.setDynamicShapePlanEnabled(config.dspEnabled);
        applyConfig(config);
        applyTempDirOverride();

        SameDiff model = null;
        INDArray inputIds = null;
        List<INDArray> auxInputs = new ArrayList<>();
        try {
            model = SDZSerializer.load(modelFile, true);
            assertNotNull(model, "SDZSerializer.load returned null");
            // Prod runs the encoder through GraphOptimizer (fusions + optional fp16 weight
            // quantization via -Dnd4j.optimizer.fp16=true). Match that so we reproduce the
            // real graph structure and ~16 GB footprint, not a raw unoptimized load.
            if (Boolean.getBoolean("bge.repro.optimize")) {
                int opsBefore = model.ops().length;
                model = GraphOptimizer.optimize(model, OUTPUT);
                log.info("GraphOptimizer applied: {} -> {} ops (fp16={})",
                        opsBefore, model.ops().length,
                        System.getProperty(ND4JSystemProperties.OPTIMIZER_FP16, "false"));
            }
            log.info("Loaded BGE model inputs={} outputs={} vars={} ops={}",
                    model.inputs(), model.outputs(), model.variables().size(), model.ops().length);

            inputIds = validationInputIds();
            Map<String, INDArray> placeholders = new LinkedHashMap<>();
            // Provide every input the (possibly pre-optimized) model declares: input_ids =
            // token ids, *mask* = all-1 (real tokens), everything else (token_type_ids) = all-0.
            for (String inName : model.inputs()) {
                if (inName.equals("input_ids")) {
                    placeholders.put(inName, inputIds);
                } else if (inName.toLowerCase().contains("mask")) {
                    INDArray mask = Nd4j.ones(DataType.INT64, BATCH, SEQ_LEN);
                    placeholders.put(inName, mask);
                    auxInputs.add(mask);
                } else {
                    INDArray aux = Nd4j.zeros(DataType.INT64, BATCH, SEQ_LEN);
                    placeholders.put(inName, aux);
                    auxInputs.add(aux);
                }
            }

            // Repro knob: bge.repro.freshInput=true allocates a NEW input_ids array
            // (fresh DataBuffer) each iteration — mirrors kompile's crawl, which feeds a
            // fresh batch every model.output() call. This stresses the frozen fast-path
            // arg-refresh gate (view-over-external-input DataBuffer changes → needsArgRefresh).
            boolean freshInput = Boolean.getBoolean("bge.repro.freshInput");

            for (int i = 0; i < iterations; i++) {
                if (freshInput && i > 0) {
                    inputIds.close();
                    inputIds = validationInputIds();
                    placeholders.put("input_ids", inputIds);
                }
                Map<String, INDArray> result = model.output(placeholders, OUTPUT);
                INDArray output = result.get(OUTPUT);
                assertNotNull(output, config.name + " iteration " + i + ": missing " + OUTPUT);

                try {
                    OutputStats stats = validateOutput(config.name, i, output);
                    log.info("{} iteration {}: {}", config.name, i, stats);
                } catch (AssertionError e) {
                    if (config.dspEnabled) {
                        reportDspSlotIssues(model, config.name, i);
                    }
                    throw e;
                }
            }
        } finally {
            if (model != null) {
                try {
                    model.close();
                } catch (Throwable t) {
                    log.warn("{}: failed to close BGE SameDiff model: {}", config.name, t.getMessage());
                }
            }
            if (inputIds != null) {
                inputIds.close();
            }
            for (INDArray aux : auxInputs) {
                if (aux != null) aux.close();
            }
            restoreProperties(previous);
            InferenceSession.setDynamicShapePlanEnabled(previousDspEnabled);
        }
    }

    private static INDArray validationInputIds() {
        long[] tokenIds = new long[BATCH * SEQ_LEN];
        long[] prefix = {
                101, 2023, 2003, 1037, 27354, 3231, 2005, 1996,
                7861, 8270, 4667, 2944, 1012, 102
        };
        for (int b = 0; b < BATCH; b++) {
            System.arraycopy(prefix, 0, tokenIds, b * SEQ_LEN, prefix.length);
        }
        return Nd4j.createFromArray(tokenIds).reshape(BATCH, SEQ_LEN).castTo(DataType.INT64);
    }

    private static OutputStats validateOutput(String configName, int iteration, INDArray output) {
        assertArrayEquals(new long[]{BATCH, SEQ_LEN, 768}, output.shape(),
                configName + " iteration " + iteration + ": unexpected " + OUTPUT + " shape");
        boolean hasNaN = output.isNaN().any();
        boolean hasInf = output.isInfinite().any();
        if (hasNaN || hasInf) {
            long firstBad = firstNonFiniteIndex(output);
            fail(configName + " iteration " + iteration + ": output contains non-finite values" +
                    " (hasNaN=" + hasNaN + ", hasInf=" + hasInf + ", firstBadIndex=" + firstBad + ")");
        }

        INDArray abs = TransientArrays.abs(output);
        double min = output.minNumber().doubleValue();
        double max = output.maxNumber().doubleValue();
        double maxAbs = abs.maxNumber().doubleValue();
        double meanAbs = abs.meanNumber().doubleValue();
        abs.close();

        assertTrue(Double.isFinite(min) && Double.isFinite(max) &&
                        Double.isFinite(maxAbs) && Double.isFinite(meanAbs),
                configName + " iteration " + iteration + ": non-finite reduction stats");
        assertTrue(maxAbs > 1e-8, configName + " iteration " + iteration + ": output magnitude is unusably small");
        assertTrue(maxAbs < 1e6, configName + " iteration " + iteration + ": output magnitude is implausibly large");
        return new OutputStats(Arrays.toString(output.shape()), output.dataType(), min, max, maxAbs, meanAbs);
    }

    private static long firstNonFiniteIndex(INDArray output) {
        long len = output.length();
        for (long i = 0; i < len; i++) {
            double value = output.getDouble(i);
            if (!Double.isFinite(value)) {
                return i;
            }
        }
        return -1;
    }

    private static void reportDspSlotIssues(SameDiff model, String configName, int iteration) {
        try {
            DspHandle handle = model.dsp();
            int firstNan = handle.firstNaNSlot();
            int[] outputFlags = handle.validateOutputs();
            if (firstNan >= 0) {
                log.error("{} iteration {}: DSP non-finite slot detected firstNaNSlot={} outputFlags={}",
                        configName, iteration, firstNan, Arrays.toString(outputFlags));
                reportPlanSlot(model, configName, iteration, firstNan);
                handle.snapshotAllSlots().entrySet().stream()
                        .filter(e -> e.getValue().contains("***"))
                        .limit(20)
                        .forEach(e -> log.error("{} iteration {}: {}", configName, iteration, e.getValue()));
            } else if (hasAnyFlag(outputFlags)) {
                log.info("{} iteration {}: DSP slot scan finite; outputFlags contain non-zero metadata flags",
                        configName, iteration);
            } else {
                log.info("{} iteration {}: DSP slot scan clean; outputFlags={}",
                        configName, iteration, Arrays.toString(outputFlags));
            }
        } catch (Throwable t) {
            log.warn("{} iteration {}: DSP slot introspection unavailable: {}",
                    configName, iteration, t.getMessage());
        }
    }

    private static void reportPlanSlot(SameDiff model, String configName, int iteration, int outputSlotIndex) {
        if (outputSlotIndex < 0) {
            return;
        }

        DynamicShapePlanExecutor executor = model.getOrCreateSession().getDynamicShapePlanExecutor();
        DynamicShapePlan plan = executor != null ? executor.getCurrentPlan() : null;
        if (plan == null || plan.getSlots() == null) {
            log.warn("{} iteration {}: no Java DSP plan available for output slot {}",
                    configName, iteration, outputSlotIndex);
            return;
        }

        int producerStep = findProducerStep(plan, outputSlotIndex);
        if (producerStep < 0) {
            log.warn("{} iteration {}: no producer step found for output slot {}",
                    configName, iteration, outputSlotIndex);
            return;
        }

            log.error("{} iteration {}: first non-finite output slot {} is produced by step {}:\n{}",
                configName, iteration, outputSlotIndex, producerStep,
                PlanIntrospection.formatSlot(plan, producerStep));
        reportSlotValue(model, configName, iteration, outputSlotIndex - 1, "upstream-output");
        reportConstantValue(model, configName, iteration, "onnx::MatMul_1511");
        for (Integer upstreamStep : PlanIntrospection.getProducersOf(plan, producerStep)) {
            log.error("{} iteration {}: upstream producer for first non-finite slot:\n{}",
                    configName, iteration, PlanIntrospection.formatSlot(plan, upstreamStep));
        }
    }

    private static void reportSlotValue(SameDiff model, String configName, int iteration, int slotIndex, String label) {
        try {
            INDArray arr = model.dsp().getSlotOutput(slotIndex);
            logArrayStats(configName, iteration, label + " slot " + slotIndex, arr);
        } catch (Throwable t) {
            log.warn("{} iteration {}: unable to inspect {} slot {}: {}",
                    configName, iteration, label, slotIndex, t.getMessage());
        }
    }

    private static void reportConstantValue(SameDiff model, String configName, int iteration, String name) {
        try {
            INDArray arr = model.getArrForVarName(name);
            logArrayStats(configName, iteration, "constant " + name, arr);
        } catch (Throwable t) {
            log.warn("{} iteration {}: unable to inspect constant {}: {}",
                    configName, iteration, name, t.getMessage());
        }
    }

    private static void logArrayStats(String configName, int iteration, String label, INDArray arr) {
        if (arr == null) {
            log.warn("{} iteration {}: {} is null", configName, iteration, label);
            return;
        }
        boolean hasNaN = arr.isNaN().any();
        boolean hasInf = arr.isInfinite().any();
        double min = Double.NaN;
        double max = Double.NaN;
        double sum = Double.NaN;
        if (arr.dataType().isNumerical()) {
            min = arr.minNumber().doubleValue();
            max = arr.maxNumber().doubleValue();
            sum = arr.sumNumber().doubleValue();
        }
        log.error("{} iteration {}: {} shape={} dtype={} hasNaN={} hasInf={} min={} max={} sum={}",
                configName, iteration, label, Arrays.toString(arr.shape()), arr.dataType(),
                hasNaN, hasInf, min, max, sum);
    }

    private static int findProducerStep(DynamicShapePlan plan, int outputSlotIndex) {
        DynamicShapeSlot[] slots = plan.getSlots();
        for (int step = 0; step < slots.length; step++) {
            int[] outputSlots = slots[step].getOutputSlotIndices();
            if (outputSlots == null) {
                continue;
            }
            for (int outputSlot : outputSlots) {
                if (outputSlot == outputSlotIndex) {
                    return step;
                }
            }
        }
        return -1;
    }

    private static boolean hasAnyFlag(int[] flags) {
        for (int flag : flags) {
            if (flag != 0) {
                return true;
            }
        }
        return false;
    }

    private static void applyConfig(RunConfig config) {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, Boolean.toString(config.dspEnabled));
        System.setProperty(ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE, config.mode.name());
        System.setProperty(ND4JSystemProperties.DSP_PLAN_CACHE_DISK_ENABLED, Boolean.toString(config.diskCacheEnabled));
        System.setProperty(ND4JSystemProperties.DSP_PLAN_CACHE_FORCE_RECOMPILE, Boolean.toString(config.forceRecompile));
        System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, Boolean.toString(config.cudaGraphsEnabled));
        System.setProperty(ND4JSystemProperties.TRITON_SKIP_KERNELS, Boolean.toString(config.tritonSkipped));
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "ALL"));
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL,
                System.getProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "full"));
    }

    private static Map<String, String> captureProperties() {
        Map<String, String> props = new LinkedHashMap<>();
        for (String key : propertyKeys()) {
            props.put(key, System.getProperty(key));
        }
        return props;
    }

    private static void restoreProperties(Map<String, String> props) {
        for (Map.Entry<String, String> e : props.entrySet()) {
            if (e.getValue() == null) {
                System.clearProperty(e.getKey());
            } else {
                System.setProperty(e.getKey(), e.getValue());
            }
        }
    }

    private static List<String> propertyKeys() {
        return List.of(
                ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED,
                ND4JSystemProperties.DSP_GRAPH_EXECUTION_MODE,
                ND4JSystemProperties.DSP_PLAN_CACHE_DISK_ENABLED,
                ND4JSystemProperties.DSP_PLAN_CACHE_FORCE_RECOMPILE,
                ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED,
                ND4JSystemProperties.TRITON_SKIP_KERNELS,
                ND4JSystemProperties.DSP_DIAGNOSTICS,
                ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL,
                "java.io.tmpdir");
    }

    private static void applyTempDirOverride() {
        String tmpDir = System.getProperty("bge.repro.tmpdir");
        if (tmpDir == null || tmpDir.trim().isEmpty()) {
            return;
        }
        File dir = new File(tmpDir);
        assertTrue(dir.exists() || dir.mkdirs(), "Could not create BGE repro temp dir: " + dir.getAbsolutePath());
        System.setProperty("java.io.tmpdir", dir.getAbsolutePath());
        log.info("BGE repro java.io.tmpdir={}", dir.getAbsolutePath());
    }

    private static final class RunConfig {
        private final String name;
        private final boolean dspEnabled;
        private final GraphExecutionMode mode;
        private final boolean diskCacheEnabled;
        private final boolean forceRecompile;
        private final boolean cudaGraphsEnabled;
        private final boolean tritonSkipped;

        private RunConfig(String name, boolean dspEnabled, GraphExecutionMode mode,
                          boolean diskCacheEnabled, boolean forceRecompile,
                          boolean cudaGraphsEnabled, boolean tritonSkipped) {
            this.name = name;
            this.dspEnabled = dspEnabled;
            this.mode = mode;
            this.diskCacheEnabled = diskCacheEnabled;
            this.forceRecompile = forceRecompile;
            this.cudaGraphsEnabled = cudaGraphsEnabled;
            this.tritonSkipped = tritonSkipped;
        }

        @Override
        public String toString() {
            return "RunConfig{" +
                    "name='" + name + '\'' +
                    ", dspEnabled=" + dspEnabled +
                    ", mode=" + mode +
                    ", diskCacheEnabled=" + diskCacheEnabled +
                    ", forceRecompile=" + forceRecompile +
                    ", cudaGraphsEnabled=" + cudaGraphsEnabled +
                    ", tritonSkipped=" + tritonSkipped +
                    '}';
        }
    }

    private static final class OutputStats {
        private final String shape;
        private final DataType dtype;
        private final double min;
        private final double max;
        private final double maxAbs;
        private final double meanAbs;

        private OutputStats(String shape, DataType dtype, double min, double max, double maxAbs, double meanAbs) {
            this.shape = shape;
            this.dtype = dtype;
            this.min = min;
            this.max = max;
            this.maxAbs = maxAbs;
            this.meanAbs = meanAbs;
        }

        @Override
        public String toString() {
            return "shape=" + shape +
                    ", dtype=" + dtype +
                    ", min=" + min +
                    ", max=" + max +
                    ", maxAbs=" + maxAbs +
                    ", meanAbs=" + meanAbs;
        }
    }

    private static final class TransientArrays {
        private static INDArray abs(INDArray array) {
            return org.nd4j.linalg.ops.transforms.Transforms.abs(array, true);
        }
    }
}
