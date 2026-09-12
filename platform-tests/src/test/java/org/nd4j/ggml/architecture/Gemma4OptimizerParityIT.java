/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.ggml.convert.GGMLToSameDiffConverter;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Numerical parity proof for the optimizer-vs-packed-graph question.
 *
 * <p>Converts the real Gemma GGUF twice — once raw (runtime-quantized weights,
 * HALF activations) and once through {@link GraphOptimizer} with the packed-graph
 * guard bypassed via -Dnd4j.optimizer.allowPackedGraphs=true (set by this test)
 * — then runs both graphs on identical inputs and asserts elementwise parity
 * within HALF tolerance.</p>
 *
 * <p>Property-gated like Gemma4LastPositionConversionIT so it never runs in
 * ordinary CI:</p>
 * <ul>
 *   <li>-Dgemma4.parity.gguf=... (required; real GGUF path)</li>
 *   <li>-Dgemma4.parity.sdz.dir=... (optional; temp workspace)</li>
 * </ul>
 *
 * <p>Verdict semantics:
 * <ul>
 *   <li>Both graphs produce finite, matching outputs → the optimizer does NOT
 *       corrupt packed graphs; the blanket guard should be removed and replaced
 *       with narrower protection.</li>
 *   <li>Raw finite, optimized NaN/Inf → corruption proven; guard stays.</li>
 *   <li>Max-abs-delta reported either way so magnitude is on the record.</li>
 * </ul></p>
 */
@Tag("integration")
class Gemma4OptimizerParityIT {

    private static final String GGUF_PROPERTY = "gemma4.parity.gguf";
    private static final String DIR_PROPERTY = "gemma4.parity.sdz.dir";

    private static final int TEST_SEQ = 8;      // tiny prefill — parity, not perf
    private static final int TEST_BATCH = 1;

    @Test
    void optimizedPackedGraphMatchesRawNumerically(@TempDir Path tempDir) throws Exception {
        String ggufProp = System.getProperty(GGUF_PROPERTY);
        assumeTrueGguf(ggufProp);
        Path gguf = Path.of(ggufProp);
        assertTrue(Files.isRegularFile(gguf), "GGUF not found: " + gguf);

        Path workspace = System.getProperty(DIR_PROPERTY) != null
                ? Path.of(System.getProperty(DIR_PROPERTY)) : tempDir;
        Files.createDirectories(workspace);
        Path rawPath = workspace.resolve("gemma4-parity-raw.sdz");
        Path optPath = workspace.resolve("gemma4-parity-opt.sdz");

        // ── Convert once (raw), load, then re-save a second copy — both graphs
        // share identical weights so any divergence is optimizer-caused.
        GGMLToSameDiffConverter converter = new GGMLToSameDiffConverter(
                ConversionOptions.builder()
                        .quantizationMode(ConversionOptions.QuantizationMode.RUNTIME_QUANTIZED_MATMUL)
                        .targetDataType(DataType.HALF)
                        .build());
        converter.convertToSDZ(gguf.toFile(), rawPath.toFile());
        assertTrue(Files.size(rawPath) > 0, "raw conversion produced no output");

        SameDiff rawSd = SDZSerializer.load(rawPath.toFile(), false);
        SameDiff optSd = SDZSerializer.load(rawPath.toFile(), false);

        // ── Optimize the second copy. GraphOptimizer currently refuses packed
        // graphs (guard). The test needs the UNPROTECTED optimizer to prove or
        // disprove corruption, so we clear the guard by flag.
        System.setProperty("nd4j.optimizer.allowPackedGraphs", "true");
        SameDiff optimized;
        try {
            optimized = GraphOptimizer.optimize(optSd, optSd.outputs());
        } finally {
            System.clearProperty("nd4j.optimizer.allowPackedGraphs");
        }
        assertNotNull(optimized, "optimizer returned null");

        // ── Identical inputs for both graphs.
        Map<String, INDArray> inputs = buildInputs(rawSd);
        Map<String, INDArray> inputsCopy = buildInputs(optimized);

        Map<String, INDArray> rawOut = rawSd.output(inputs, rawSd.outputs());
        Map<String, INDArray> optOut = optimized.output(inputsCopy, optimized.outputs());

        assertFalse(rawOut.isEmpty(), "raw graph produced no outputs");
        assertFalse(optOut.isEmpty(), "optimized graph produced no outputs");

        // ── Compare every shared output.
        int compared = 0;
        double worstDelta = 0.0;
        String worstName = null;
        for (Map.Entry<String, INDArray> e : rawOut.entrySet()) {
            INDArray opt = optOut.get(e.getKey());
            if (opt == null) continue;
            INDArray raw = e.getValue();
            assertEquals(raw.dataType(), opt.dataType(),
                    "dtype divergence on " + e.getKey());

            INDArray rawF = raw.dataType() == DataType.FLOAT ? raw : raw.castTo(DataType.FLOAT);
            INDArray optF = opt.dataType() == DataType.FLOAT ? opt : opt.castTo(DataType.FLOAT);

            long rawNan = countNonFinite(rawF);
            long optNan = countNonFinite(optF);
            assertEquals(0, rawNan, "RAW graph has non-finite values in " + e.getKey());
            assertEquals(0, optNan,
                    "CORRUPTION PROVEN: optimized graph has non-finite values in "
                            + e.getKey() + " while raw graph is finite");

            double delta = maxAbsDelta(rawF, optF);
            if (delta > worstDelta) {
                worstDelta = delta;
                worstName = e.getKey();
            }
            compared++;
        }
        assertTrue(compared > 0, "no shared outputs to compare");

        // HALF rounding: elementwise parity within 2 ulp of half precision at
        // magnitude 1 is 2^-9 ≈ 0.002; logits magnitude grows so allow 0.05.
        assertTrue(worstDelta <= 0.05,
                "PARITY FAILURE: max abs delta " + worstDelta + " on " + worstName
                        + " exceeds HALF-rounding tolerance");
        System.out.println("PARITY OK: compared " + compared
                + " outputs, max abs delta " + worstDelta + " (" + worstName + ")");
    }

    private static void assumeTrueGguf(String prop) {
        if (prop == null || prop.isBlank()) {
            throw new org.opentest4j.TestAbortedException(
                    "gemma4.parity.gguf not set — skipping optimizer parity IT");
        }
    }

    private static Map<String, INDArray> buildInputs(SameDiff sd) {
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        INDArray ids = Nd4j.createFromArray(new long[][]{
                {2L, 101L, 9394L, 20495L, 108L, 235248L, 6853L, 108L}});
        inputs.put("input_ids", ids.reshape(TEST_BATCH, TEST_SEQ));
        inputs.put("position_offset", Nd4j.createFromArray(0L));
        inputs.put("cache_position", Nd4j.createFromArray(0L));
        inputs.put("actual_sequence_length", Nd4j.createFromArray((long) TEST_SEQ));
        INDArray mask = Nd4j.zeros(DataType.FLOAT, TEST_BATCH, 1, TEST_SEQ, TEST_SEQ);
        for (int q = 0; q < TEST_SEQ; q++) {
            for (int k = 0; k <= q; k++) mask.putScalar(new int[]{0, 0, q, k}, 1.0f);
        }
        inputs.put("_causal_mask", mask);

        // Every remaining unresolved placeholder is a KV-cache entry — give it a
        // fresh (zero) cache of the placeholder's declared dtype, filling the -1
        // leading dims (batch, past length) with the test batch and test sequence.
        for (SDVariable ph : sd.placeHolders()) {
            String name = ph.name();
            if (inputs.containsKey(name)) continue;
            if (!name.startsWith("past_key_values.")) {
                throw new IllegalStateException("Unresolved non-KV placeholder: " + name);
            }
            long[] shape = ph.getShape();
            long[] concrete = new long[shape.length];
            for (int d = 0; d < shape.length; d++) {
                concrete[d] = shape[d] < 0 ? TEST_SEQ : shape[d];
            }
            inputs.put(name, Nd4j.zeros(ph.dataType(), concrete));
        }
        return inputs;
    }

    private static long countNonFinite(INDArray a) {
        long n = 0;
        for (int i = 0; i < (int) Math.min(a.length(), 1_000_000L); i++) {
            double v = a.getDouble(i);
            if (Double.isNaN(v) || Double.isInfinite(v)) n++;
        }
        return n;
    }

    private static double maxAbsDelta(INDArray a, INDArray b) {
        double worst = 0.0;
        long n = Math.min(a.length(), 1_000_000L);
        for (int i = 0; i < n; i++) {
            double d = Math.abs(a.getDouble(i) - b.getDouble(i));
            if (d > worst) worst = d;
        }
        return worst;
    }
}
